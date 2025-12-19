"""Localization trainer for coordinate regression.

Specialized trainer for IVD localization with ConvNext models.
Uses HuggingFace Accelerate and supports wandb logging.
"""

from typing import Any, Literal

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from spine_vision.training.base import BaseTrainer, TrainingConfig, TrainingResult
from spine_vision.training.datasets.ivd_coords import (
    IDX_TO_LEVEL,
    NUM_LEVELS,
    IVDCoordsCollator,
    IVDCoordsDataset,
)
from spine_vision.training.metrics import LocalizationMetrics
from spine_vision.training.models.convnext import ConvNextLocalization
from spine_vision.training.visualization import TrainingVisualizer


class LocalizationConfig(TrainingConfig):
    """Configuration for localization training."""

    # Override task name for output path structure
    task: str = "localization"

    # Model configuration
    model_variant: Literal[
        "tiny", "small", "base", "large", "xlarge",
        "v2_tiny", "v2_small", "v2_base", "v2_large", "v2_huge"
    ] = "base"
    """ConvNext variant to use (timm models)."""

    pretrained: bool = True
    """Use ImageNet pretrained weights."""

    freeze_backbone_epochs: int = 0
    """Number of epochs to freeze backbone (0 = never freeze)."""

    dropout: float = 0.2
    """Dropout rate in head."""

    loss_type: Literal["mse", "smooth_l1", "huber"] = "smooth_l1"
    """Loss function type."""

    use_level_embedding: bool = True
    """Use level embedding in model."""

    level_embedding_dim: int = 16
    """Dimension of level embedding."""

    # Dataset configuration
    series_types: list[str] | None = None
    """Filter to specific series types (e.g., ['sag_t1', 'sag_t2'])."""

    levels: list[str] | None = None
    """Filter to specific levels (e.g., ['L4/L5', 'L5/S1'])."""

    sources: list[str] | None = None
    """Filter to specific sources."""

    image_size: tuple[int, int] = (224, 224)
    """Target image size (H, W)."""

    augment: bool = True
    """Apply data augmentation during training."""

    # Validation configuration
    pck_thresholds: list[float] = [0.02, 0.05, 0.10]
    """Thresholds for PCK metric."""

    visualize_predictions: bool = True
    """Generate prediction visualizations during validation."""

    num_visualization_samples: int = 16
    """Number of samples to visualize."""


class LocalizationTrainer(
    BaseTrainer[LocalizationConfig, ConvNextLocalization, IVDCoordsDataset]
):
    """Trainer for IVD localization with coordinate regression.

    Uses HuggingFace Accelerate for distributed training and mixed precision.
    Supports optional wandb logging for experiment tracking.
    """

    def __init__(
        self,
        config: LocalizationConfig,
        model: ConvNextLocalization | None = None,
        train_dataset: IVDCoordsDataset | None = None,
        val_dataset: IVDCoordsDataset | None = None,
    ) -> None:
        """Initialize trainer.

        If model or datasets are not provided, they will be created
        from the configuration.
        """
        # Create model if not provided
        if model is None:
            num_levels = NUM_LEVELS if config.use_level_embedding else 1
            model = ConvNextLocalization(
                variant=config.model_variant,
                num_outputs=2,
                pretrained=config.pretrained,
                dropout=config.dropout,
                freeze_backbone=config.freeze_backbone_epochs > 0,
                num_levels=num_levels,
                loss_type=config.loss_type,
                level_embedding_dim=config.level_embedding_dim,
            )

        # Create datasets if not provided
        if train_dataset is None:
            train_dataset = IVDCoordsDataset(
                data_path=config.data_path,
                split="train",
                val_ratio=config.val_split,
                series_types=config.series_types,
                levels=config.levels,
                sources=config.sources,
                image_size=config.image_size,
                augment=config.augment,
            )

        if val_dataset is None:
            val_dataset = IVDCoordsDataset(
                data_path=config.data_path,
                split="val",
                val_ratio=config.val_split,
                series_types=config.series_types,
                levels=config.levels,
                sources=config.sources,
                image_size=config.image_size,
                augment=False,
            )

        super().__init__(config, model, train_dataset, val_dataset)

        # Metrics calculator
        self.metrics = LocalizationMetrics(
            pck_thresholds=config.pck_thresholds,
            level_names=list(IDX_TO_LEVEL.values()),
        )

        # Visualizer with wandb support - save to logs/
        self.visualizer = TrainingVisualizer(
            output_path=config.logs_path,
            output_mode="html",
            use_wandb=config.use_wandb,
        )

        # Track backbone freeze state
        self._backbone_unfrozen = config.freeze_backbone_epochs == 0

    def _create_dataloader(
        self,
        dataset: IVDCoordsDataset,
        shuffle: bool = True,
    ) -> DataLoader[Any]:
        """Create DataLoader with custom collator."""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=shuffle,
            collate_fn=IVDCoordsCollator(),
        )

    def _unpack_batch(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack batch from IVDCoordsDataset."""
        return batch["image"], batch["coords"]

    def _train_step(self, batch: dict[str, Any]) -> float:
        """Training step with level embedding support."""
        inputs = batch["image"]
        targets = batch["coords"]
        level_idx = (
            batch["level_idx"]
            if self.config.use_level_embedding
            else None
        )

        self.optimizer.zero_grad()

        # Accelerator handles mixed precision and device placement
        with self.accelerator.autocast():
            predictions = self.model(inputs, level_idx)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            loss = unwrapped_model.get_loss(predictions, targets)

        self.accelerator.backward(loss)

        if self.config.grad_clip:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )

        self.optimizer.step()

        return loss.item()

    def _validate_epoch(self) -> tuple[float, dict[str, float]]:
        """Validate with localization metrics."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_predictions: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        all_levels: list[torch.Tensor] = []
        all_metadata: list[dict[str, Any]] = []

        # For visualization
        sample_images: list[np.ndarray] = []
        sample_indices: list[int] = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):  # type: ignore[union-attr]
                inputs = batch["image"]
                targets = batch["coords"]
                level_idx = (
                    batch["level_idx"]
                    if self.config.use_level_embedding
                    else None
                )

                with self.accelerator.autocast():
                    predictions = self.model(inputs, level_idx)
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    loss = unwrapped_model.get_loss(predictions, targets)

                total_loss += loss.item()
                num_batches += 1

                # Gather from all processes
                all_preds: torch.Tensor = self.accelerator.gather(predictions)  # type: ignore[assignment]
                all_tgts: torch.Tensor = self.accelerator.gather(targets)  # type: ignore[assignment]
                all_lvls: torch.Tensor = self.accelerator.gather(batch["level_idx"])  # type: ignore[assignment]

                all_predictions.append(all_preds.cpu())
                all_targets.append(all_tgts.cpu())
                all_levels.append(all_lvls.cpu())
                all_metadata.extend(batch["metadata"])

                # Collect samples for visualization (only on main process)
                if (
                    self.config.visualize_predictions
                    and self.accelerator.is_main_process
                    and len(sample_images) < self.config.num_visualization_samples
                ):
                    # Store first few batches of images (denormalized)
                    images_np = self._denormalize_images(batch["image"])
                    for img in images_np:
                        if len(sample_images) < self.config.num_visualization_samples:
                            sample_images.append(img)
                            sample_indices.append(
                                len(all_predictions) * self.config.batch_size
                                + len(sample_images)
                                - 1
                            )

        avg_loss = total_loss / num_batches

        # Compute metrics
        predictions_cat = torch.cat(all_predictions, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        levels_cat = torch.cat(all_levels, dim=0)

        metrics = self.metrics.compute(
            predictions_cat.numpy(),
            targets_cat.numpy(),
            levels_cat.numpy(),
        )

        # Generate visualizations (only on main process)
        if (
            self.config.visualize_predictions
            and sample_images
            and self.accelerator.is_main_process
        ):
            n_vis = min(len(sample_images), self.config.num_visualization_samples)
            vis_preds = predictions_cat[:n_vis].numpy()
            vis_targets = targets_cat[:n_vis].numpy()
            vis_metadata = all_metadata[:n_vis]

            self.visualizer.plot_localization_predictions(
                sample_images[:n_vis],
                vis_preds,
                vis_targets,
                vis_metadata,
                filename=f"predictions_epoch_{self.current_epoch}",
            )

        return avg_loss, metrics

    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        """Compute localization metrics."""
        return self.metrics.compute(
            predictions.numpy(),
            targets.numpy(),
        )

    def _denormalize_images(self, images: torch.Tensor) -> list[np.ndarray]:
        """Denormalize ImageNet-normalized images for visualization."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        images = images.cpu()
        images = images * std + mean
        images = torch.clamp(images, 0, 1)

        # Convert to numpy [B, H, W, C]
        images_np = images.permute(0, 2, 3, 1).numpy()
        images_np = (images_np * 255).astype(np.uint8)

        return [img for img in images_np]

    def train(self) -> TrainingResult:
        """Train with backbone unfreezing support."""
        # Check if we need to unfreeze backbone at some point
        if self.config.freeze_backbone_epochs > 0:
            logger.info(
                f"Backbone frozen for first {self.config.freeze_backbone_epochs} epochs"
            )

        result = self._train_loop()

        # Generate final visualizations (only on main process)
        if self.accelerator.is_main_process:
            self._generate_final_visualizations()

        return result

    def _train_loop(self) -> TrainingResult:
        """Main training loop with backbone unfreezing."""
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Model: {unwrapped_model.name}")
        logger.info(f"Parameters: {unwrapped_model.count_parameters():,}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train samples: {len(self.train_dataset)}")
        if self.val_dataset:
            logger.info(f"Val samples: {len(self.val_dataset)}")
        if self.config.use_wandb:
            logger.info(f"Logging to wandb project: {self.config.wandb_project}")

        # Log dataset statistics
        stats = self.train_dataset.get_stats()
        logger.info(f"Train dataset stats: {stats}")

        if self.config.checkpoint_path:
            self._load_checkpoint(self.config.checkpoint_path)

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            # Unfreeze backbone if needed
            if (
                not self._backbone_unfrozen
                and epoch >= self.config.freeze_backbone_epochs
            ):
                logger.info(f"Unfreezing backbone at epoch {epoch + 1}")
                unwrapped_model.unfreeze_backbone()
                self._backbone_unfrozen = True

            # Training
            train_loss = self._train_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # Validation
            val_loss: float | None = None
            metrics: dict[str, float] = {}
            if self.val_loader and (epoch + 1) % self.config.val_frequency == 0:
                val_loss, metrics = self._validate_epoch()
                self.history["val_loss"].append(val_loss)

                for key, value in metrics.items():
                    if key not in self.history:
                        self.history[key] = []
                    self.history[key].append(value)

            # Scheduler step
            if self.scheduler:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    if val_loss is not None:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Logging
            self._log_epoch(epoch, train_loss, val_loss, metrics)

            # Log to wandb
            wandb_metrics = {
                "train/loss": train_loss,
                "train/lr": self.optimizer.param_groups[0]["lr"],
            }
            if val_loss is not None:
                wandb_metrics["val/loss"] = val_loss
            for key, value in metrics.items():
                wandb_metrics[f"val/{key}"] = value
            self._log_to_wandb(wandb_metrics, step=epoch)

            # Checkpointing (use MED as primary metric, lower is better)
            metric_for_checkpoint = metrics.get(
                "med", val_loss if val_loss else train_loss
            )
            is_best = metric_for_checkpoint < self.best_metric - self.config.min_delta
            if is_best:
                self.best_metric = metric_for_checkpoint
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1

            if (epoch + 1) % self.config.save_frequency == 0:
                self._save_checkpoint(is_best=False)

            # Early stopping
            if (
                self.config.early_stopping
                and self.patience_counter >= self.config.patience
            ):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        assert self.config.output_path is not None
        best_checkpoint = self.config.output_path / "best_model.pt"
        if best_checkpoint.exists():
            self._load_checkpoint(best_checkpoint)

        # End wandb run
        if self._wandb_initialized:
            self.accelerator.end_training()

        return TrainingResult(
            best_epoch=self.best_epoch,
            best_metric=self.best_metric,
            final_train_loss=self.history["train_loss"][-1],
            final_val_loss=self.history["val_loss"][-1]
            if self.history["val_loss"]
            else 0.0,
            history=self.history,
            checkpoint_path=best_checkpoint,
        )

    def _generate_final_visualizations(self) -> None:
        """Generate final training visualizations."""
        # Training curves
        self.visualizer.plot_training_curves(
            self.history,
            filename="training_curves",
        )

        # Error distribution on validation set
        if self.val_loader:
            self.model.eval()
            all_predictions = []
            all_targets = []
            all_levels = []

            with torch.no_grad():
                for batch in self.val_loader:
                    inputs = batch["image"]
                    level_idx = (
                        batch["level_idx"]
                        if self.config.use_level_embedding
                        else None
                    )
                    predictions = self.model(inputs, level_idx)

                    # Gather from all processes
                    all_preds: torch.Tensor = self.accelerator.gather(predictions)  # type: ignore[assignment]
                    all_tgts: torch.Tensor = self.accelerator.gather(batch["coords"])  # type: ignore[assignment]
                    all_lvls: torch.Tensor = self.accelerator.gather(batch["level_idx"])  # type: ignore[assignment]

                    all_predictions.append(all_preds.cpu().numpy())
                    all_targets.append(all_tgts.cpu().numpy())
                    all_levels.append(all_lvls.cpu().numpy())

            predictions = np.concatenate(all_predictions, axis=0)
            targets = np.concatenate(all_targets, axis=0)
            levels = np.concatenate(all_levels, axis=0)

            self.visualizer.plot_error_distribution(
                predictions,
                targets,
                levels,
                level_names=list(IDX_TO_LEVEL.values()),
                filename="error_distribution",
            )

            # Per-level metrics
            final_metrics = self.metrics.compute(predictions, targets, levels)
            self.visualizer.plot_per_level_metrics(
                final_metrics,
                level_names=list(IDX_TO_LEVEL.values()),
                metric_prefix="med_",
                filename="per_level_med",
            )

        logger.info(
            f"Visualizations saved to: {self.config.logs_path}"
        )

    def evaluate(
        self, test_dataset: IVDCoordsDataset | None = None
    ) -> dict[str, float]:
        """Evaluate model on test set.

        Args:
            test_dataset: Test dataset. If None, creates from config.

        Returns:
            Dictionary of evaluation metrics.
        """
        if test_dataset is None:
            test_dataset = IVDCoordsDataset(
                data_path=self.config.data_path,
                split="test",
                val_ratio=self.config.val_split,
                series_types=self.config.series_types,
                levels=self.config.levels,
                sources=self.config.sources,
                image_size=self.config.image_size,
                augment=False,
            )

        test_loader = self._create_dataloader(test_dataset, shuffle=False)
        test_loader = self.accelerator.prepare(test_loader)

        self.model.eval()
        all_predictions = []
        all_targets = []
        all_levels = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["image"]
                level_idx = (
                    batch["level_idx"]
                    if self.config.use_level_embedding
                    else None
                )
                predictions = self.model(inputs, level_idx)

                # Gather from all processes
                all_preds: torch.Tensor = self.accelerator.gather(predictions)  # type: ignore[assignment]
                all_tgts: torch.Tensor = self.accelerator.gather(batch["coords"])  # type: ignore[assignment]
                all_lvls: torch.Tensor = self.accelerator.gather(batch["level_idx"])  # type: ignore[assignment]

                all_predictions.append(all_preds.cpu().numpy())
                all_targets.append(all_tgts.cpu().numpy())
                all_levels.append(all_lvls.cpu().numpy())

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        levels = np.concatenate(all_levels, axis=0)

        metrics = self.metrics.compute(predictions, targets, levels)

        logger.info("Test Results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        # Log to wandb
        if self._wandb_initialized:
            wandb_metrics = {f"test/{key}": value for key, value in metrics.items()}
            self._log_to_wandb(wandb_metrics)

        return metrics

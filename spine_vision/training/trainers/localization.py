"""Localization trainer for coordinate regression.

Specialized trainer for IVD localization with configurable backbone.
Uses HuggingFace Accelerate and supports trackio logging.

The model outputs all 5 IVD level coordinates in a single forward pass.
"""

from typing import Any, Literal

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from spine_vision.training.base import BaseTrainer, TrainingConfig, TrainingResult
from spine_vision.training.datasets.localization import (
    IDX_TO_LEVEL,
    NUM_LEVELS,
    LocalizationCollator,
    LocalizationDataset,
)
from spine_vision.training.metrics import LocalizationMetrics
from spine_vision.training.models import CoordinateRegressor
from spine_vision.training.registry import register_trainer
from spine_vision.visualization import TrainingVisualizer


class LocalizationConfig(TrainingConfig):
    """Configuration for localization training."""

    # Override task name for output path structure
    task: str = "localization"

    # Model configuration
    backbone: str = "convnext_base"
    """Backbone architecture (see BackboneFactory for options)."""

    pretrained: bool = True
    """Use ImageNet pretrained weights."""

    freeze_backbone_epochs: int = 0
    """Number of epochs to freeze backbone (0 = never freeze)."""

    dropout: float = 0.2
    """Dropout rate in head."""

    loss_type: Literal["mse", "smooth_l1", "huber"] = "smooth_l1"
    """Loss function type."""

    num_levels: int = NUM_LEVELS
    """Number of IVD levels to predict (default 5)."""

    # Dataset configuration
    series_types: list[str] | None = None
    """Filter to specific series types (e.g., ['sag_t1', 'sag_t2'])."""

    sources: list[str] | None = None
    """Filter to specific sources."""

    image_size: tuple[int, int] = (512, 512)
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


@register_trainer("localization", config_cls=LocalizationConfig)
class LocalizationTrainer(
    BaseTrainer[LocalizationConfig, CoordinateRegressor, LocalizationDataset]
):
    """Trainer for IVD localization with coordinate regression.

    Uses HuggingFace Accelerate for distributed training and mixed precision.
    Supports optional trackio logging for experiment tracking.

    The model outputs all 5 IVD level coordinates in a single forward pass,
    eliminating the need for multiple passes per image.

    Uses training hooks instead of overriding the entire train loop:
    - on_train_begin: Log dataset stats
    - on_epoch_begin: Handle backbone unfreezing
    - on_train_end: Generate final visualizations
    - get_metric_for_checkpoint: Use MED instead of loss
    """

    def __init__(
        self,
        config: LocalizationConfig,
        model: CoordinateRegressor | None = None,
        train_dataset: LocalizationDataset | None = None,
        val_dataset: LocalizationDataset | None = None,
    ) -> None:
        """Initialize trainer.

        If model or datasets are not provided, they will be created
        from the configuration.
        """
        # Create model if not provided
        if model is None:
            model = CoordinateRegressor(
                backbone=config.backbone,
                num_outputs=2,
                pretrained=config.pretrained,
                dropout=config.dropout,
                freeze_backbone=config.freeze_backbone_epochs > 0,
                num_levels=config.num_levels,
                loss_type=config.loss_type,
            )

        # Create datasets if not provided
        if train_dataset is None:
            train_dataset = LocalizationDataset(
                data_path=config.data_path,
                split="train",
                val_ratio=config.val_split,
                series_types=config.series_types,
                sources=config.sources,
                image_size=config.image_size,
                augment=config.augment,
            )

        if val_dataset is None:
            val_dataset = LocalizationDataset(
                data_path=config.data_path,
                split="val",
                val_ratio=config.val_split,
                series_types=config.series_types,
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

        # Visualizer with trackio support - save to logs/
        self.visualizer = TrainingVisualizer(
            output_path=config.logs_path,
            output_mode="image",
            use_trackio=config.use_trackio,
        )

        # Track backbone freeze state
        self._backbone_unfrozen = config.freeze_backbone_epochs == 0

    def _create_dataloader(
        self,
        dataset: LocalizationDataset,
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
            collate_fn=LocalizationCollator(),
        )

    def _unpack_batch(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack batch from LocalizationDataset."""
        return batch["image"], batch["coords"]

    def _train_step(self, batch: dict[str, Any]) -> float:
        """Training step with masked loss for valid levels."""
        inputs = batch["image"]  # [B, C, H, W]
        targets = batch["coords"]  # [B, 5, 2]
        mask = batch["mask"]  # [B, 5]

        self.optimizer.zero_grad()

        # Accelerator handles mixed precision and device placement
        with self.accelerator.autocast():
            predictions = self.model(inputs)  # [B, 5, 2]
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            loss = unwrapped_model.get_loss(predictions, targets, mask=mask)

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
        all_masks: list[torch.Tensor] = []
        all_metadata: list[dict[str, Any]] = []

        # For visualization
        sample_images: list[np.ndarray] = []

        with torch.no_grad():
            for batch in self.val_loader:  # type: ignore[union-attr]
                inputs = batch["image"]  # [B, C, H, W]
                targets = batch["coords"]  # [B, 5, 2]
                mask = batch["mask"]  # [B, 5]

                with self.accelerator.autocast():
                    predictions = self.model(inputs)  # [B, 5, 2]
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    loss = unwrapped_model.get_loss(predictions, targets, mask=mask)

                total_loss += loss.item()
                num_batches += 1

                # Gather from all processes
                all_preds: torch.Tensor = self.accelerator.gather(predictions)  # type: ignore[assignment]
                all_tgts: torch.Tensor = self.accelerator.gather(targets)  # type: ignore[assignment]
                all_msks: torch.Tensor = self.accelerator.gather(mask)  # type: ignore[assignment]

                all_predictions.append(all_preds.cpu())
                all_targets.append(all_tgts.cpu())
                all_masks.append(all_msks.cpu())
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

        avg_loss = total_loss / num_batches

        # Concatenate all results: [N, 5, 2] for preds/targets, [N, 5] for masks
        predictions_cat = torch.cat(all_predictions, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        masks_cat = torch.cat(all_masks, dim=0)

        # Flatten for metrics: extract valid predictions based on mask
        # Shape: [N, 5, 2] -> [N*5, 2] for valid entries
        flat_preds, flat_targets, flat_levels = self._flatten_with_mask(
            predictions_cat, targets_cat, masks_cat
        )

        metrics = self.metrics.compute(
            flat_preds,
            flat_targets,
            flat_levels,
        )

        # Generate visualizations (only on main process)
        if (
            self.config.visualize_predictions
            and sample_images
            and self.accelerator.is_main_process
        ):
            n_vis = min(len(sample_images), self.config.num_visualization_samples)
            # Flatten predictions for visualization: [n_vis, 5, 2] -> [n_vis*5, 2]
            vis_preds = predictions_cat[:n_vis].numpy().reshape(-1, 2)
            vis_targets = targets_cat[:n_vis].numpy().reshape(-1, 2)
            # Repeat metadata for each level
            vis_metadata = []
            for meta in all_metadata[:n_vis]:
                for level_name in IDX_TO_LEVEL.values():
                    vis_metadata.append({**meta, "level": level_name})

            self.visualizer.plot_localization_predictions(
                sample_images[:n_vis] * NUM_LEVELS,  # Repeat images for each level
                vis_preds,
                vis_targets,
                vis_metadata,
                filename=f"predictions_epoch_{self.current_epoch}",
            )

        return avg_loss, metrics

    def _flatten_with_mask(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flatten multi-level predictions using mask.

        Args:
            predictions: [N, num_levels, 2]
            targets: [N, num_levels, 2]
            masks: [N, num_levels] with 1=valid, 0=invalid

        Returns:
            Tuple of (flat_preds, flat_targets, flat_levels) as numpy arrays.
            Each has shape [M, 2] where M is total valid entries.
        """
        flat_preds = []
        flat_targets = []
        flat_levels = []

        masks_np = masks.numpy()
        preds_np = predictions.numpy()
        targets_np = targets.numpy()

        for sample_idx in range(len(predictions)):
            for level_idx in range(NUM_LEVELS):
                if masks_np[sample_idx, level_idx] > 0:
                    flat_preds.append(preds_np[sample_idx, level_idx])
                    flat_targets.append(targets_np[sample_idx, level_idx])
                    flat_levels.append(level_idx)

        return (
            np.array(flat_preds),
            np.array(flat_targets),
            np.array(flat_levels),
        )

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

    # ==================== Training Hooks ====================

    def on_train_begin(self) -> None:
        """Log dataset stats and freeze info at training start."""
        if self.config.freeze_backbone_epochs > 0:
            logger.info(
                f"Backbone frozen for first {self.config.freeze_backbone_epochs} epochs"
            )

        # Log dataset statistics
        stats = self.train_dataset.get_stats()
        logger.info(f"Train dataset stats: {stats}")

    def on_epoch_begin(self, epoch: int) -> None:
        """Handle backbone unfreezing."""
        if (
            not self._backbone_unfrozen
            and epoch >= self.config.freeze_backbone_epochs
        ):
            logger.info(f"Unfreezing backbone at epoch {epoch + 1}")
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.unfreeze_backbone()
            self._backbone_unfrozen = True

    def on_train_end(self, result: TrainingResult) -> None:
        """Generate final visualizations."""
        if self.accelerator.is_main_process:
            self._generate_final_visualizations()

    def get_metric_for_checkpoint(
        self,
        val_loss: float | None,
        metrics: dict[str, float],
    ) -> float:
        """Use MED as primary metric for checkpointing (lower is better)."""
        if "med" in metrics:
            return metrics["med"]
        if val_loss is not None:
            return val_loss
        return self.history["train_loss"][-1] if self.history["train_loss"] else float("inf")

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
            all_predictions: list[torch.Tensor] = []
            all_targets: list[torch.Tensor] = []
            all_masks: list[torch.Tensor] = []

            with torch.no_grad():
                for batch in self.val_loader:
                    inputs = batch["image"]  # [B, C, H, W]
                    predictions = self.model(inputs)  # [B, 5, 2]

                    # Gather from all processes
                    all_preds: torch.Tensor = self.accelerator.gather(predictions)  # type: ignore[assignment]
                    all_tgts: torch.Tensor = self.accelerator.gather(batch["coords"])  # type: ignore[assignment]
                    all_msks: torch.Tensor = self.accelerator.gather(batch["mask"])  # type: ignore[assignment]

                    all_predictions.append(all_preds.cpu())
                    all_targets.append(all_tgts.cpu())
                    all_masks.append(all_msks.cpu())

            predictions_cat = torch.cat(all_predictions, dim=0)
            targets_cat = torch.cat(all_targets, dim=0)
            masks_cat = torch.cat(all_masks, dim=0)

            # Flatten for metrics
            flat_preds, flat_targets, flat_levels = self._flatten_with_mask(
                predictions_cat, targets_cat, masks_cat
            )

            self.visualizer.plot_error_distribution(
                flat_preds,
                flat_targets,
                flat_levels,
                level_names=list(IDX_TO_LEVEL.values()),
                filename="error_distribution",
            )

            # Per-level metrics
            final_metrics = self.metrics.compute(flat_preds, flat_targets, flat_levels)
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
        self, test_dataset: LocalizationDataset | None = None
    ) -> dict[str, float]:
        """Evaluate model on test set.

        Args:
            test_dataset: Test dataset. If None, creates from config.

        Returns:
            Dictionary of evaluation metrics.
        """
        if test_dataset is None:
            test_dataset = LocalizationDataset(
                data_path=self.config.data_path,
                split="test",
                val_ratio=self.config.val_split,
                series_types=self.config.series_types,
                sources=self.config.sources,
                image_size=self.config.image_size,
                augment=False,
            )

        test_loader = self._create_dataloader(test_dataset, shuffle=False)
        test_loader = self.accelerator.prepare(test_loader)

        self.model.eval()
        all_predictions: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["image"]  # [B, C, H, W]
                predictions = self.model(inputs)  # [B, 5, 2]

                # Gather from all processes
                all_preds: torch.Tensor = self.accelerator.gather(predictions)  # type: ignore[assignment]
                all_tgts: torch.Tensor = self.accelerator.gather(batch["coords"])  # type: ignore[assignment]
                all_msks: torch.Tensor = self.accelerator.gather(batch["mask"])  # type: ignore[assignment]

                all_predictions.append(all_preds.cpu())
                all_targets.append(all_tgts.cpu())
                all_masks.append(all_msks.cpu())

        predictions_cat = torch.cat(all_predictions, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        masks_cat = torch.cat(all_masks, dim=0)

        # Flatten for metrics
        flat_preds, flat_targets, flat_levels = self._flatten_with_mask(
            predictions_cat, targets_cat, masks_cat
        )

        metrics = self.metrics.compute(flat_preds, flat_targets, flat_levels)

        logger.info("Test Results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        # Log to trackio
        if self._trackio_initialized:
            trackio_metrics = {f"test/{key}": value for key, value in metrics.items()}
            self._log_to_trackio(trackio_metrics)

        return metrics

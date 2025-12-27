"""Classification trainer for multi-task lumbar spine grading.

Specialized trainer for multi-task classification with dual-modality input.
Uses HuggingFace Accelerate and supports wandb logging.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from spine_vision.training.base import BaseTrainer, TrainingConfig, TrainingResult
from spine_vision.training.datasets.classification import (
    ClassificationCollator,
    ClassificationDataset,
)
from spine_vision.training.metrics import MTLClassificationMetrics
from spine_vision.training.models import MultiTaskClassifier, MTLTargets, TaskConfig
from spine_vision.training.registry import register_trainer
from spine_vision.training.visualization import TrainingVisualizer


def _create_lumbar_spine_tasks(
    label_smoothing: float = 0.1,
    class_weights: dict[str, torch.Tensor] | None = None,
) -> list[TaskConfig]:
    """Create lumbar spine classification tasks with optional class weights.

    Args:
        label_smoothing: Label smoothing for multiclass tasks.
        class_weights: Dict mapping task names to class weight tensors.

    Returns:
        List of TaskConfig for the 6 lumbar spine classification tasks.
    """
    cw = class_weights or {}

    return [
        TaskConfig(
            name="pfirrmann",
            num_classes=5,
            task_type="multiclass",
            label_smoothing=label_smoothing,
            class_weights=cw.get("pfirrmann"),
        ),
        TaskConfig(
            name="modic",
            num_classes=4,
            task_type="multiclass",
            label_smoothing=label_smoothing,
            class_weights=cw.get("modic"),
        ),
        TaskConfig(
            name="herniation",
            num_classes=2,
            task_type="multilabel",
            class_weights=cw.get("herniation"),
        ),
        TaskConfig(
            name="endplate",
            num_classes=2,
            task_type="multilabel",
            class_weights=cw.get("endplate"),
        ),
        TaskConfig(
            name="spondy",
            num_classes=1,
            task_type="binary",
            class_weights=cw.get("spondy"),
        ),
        TaskConfig(
            name="narrowing",
            num_classes=1,
            task_type="binary",
            class_weights=cw.get("narrowing"),
        ),
    ]


class ClassificationConfig(TrainingConfig):
    """Configuration for multi-task classification training."""

    task: str = "classification"
    data_path: Path = Path("data/processed/classification")
    """Classification dataset path."""

    # Model configuration
    backbone: str = "resnet50"
    """Backbone architecture (see BackboneFactory for options)."""

    pretrained: bool = True
    dropout: float = 0.3
    freeze_backbone_epochs: int = 0
    label_smoothing: float = 0.1
    use_class_weights: bool = True
    """Use class weights for imbalanced data."""

    # Dataset configuration
    levels: list[str] | None = None
    """Filter to specific IVD levels."""

    output_size: tuple[int, int] = (128, 128)
    """Final input size to model."""

    augment: bool = True

    # Visualization
    visualize_predictions: bool = True
    num_visualization_samples: int = 16


@register_trainer(
    "classification",
    config_cls=ClassificationConfig,
    description="Multi-task classification for lumbar spine grading",
)
class ClassificationTrainer(
    BaseTrainer[ClassificationConfig, MultiTaskClassifier, ClassificationDataset]
):
    """Trainer for multi-task lumbar spine classification.

    Uses MultiTaskClassifier with configurable backbone and 6 classification heads.
    Supports dual-modality input (T1 + T2 crops).

    Uses training hooks:
    - on_train_begin: Log dataset stats
    - on_epoch_begin: Handle backbone unfreezing
    - on_train_end: Generate final visualizations
    - get_metric_for_checkpoint: Use overall_accuracy (negated for lower-is-better)
    """

    def __init__(
        self,
        config: ClassificationConfig,
        model: MultiTaskClassifier | None = None,
        train_dataset: ClassificationDataset | None = None,
        val_dataset: ClassificationDataset | None = None,
    ) -> None:
        """Initialize trainer."""
        # Create datasets first to compute class weights
        if train_dataset is None:
            train_dataset = ClassificationDataset(
                data_path=config.data_path,
                split="train",
                val_ratio=config.val_split,
                levels=config.levels,
                output_size=config.output_size,
                augment=config.augment,
            )

        if val_dataset is None:
            val_dataset = ClassificationDataset(
                data_path=config.data_path,
                split="val",
                val_ratio=config.val_split,
                levels=config.levels,
                output_size=config.output_size,
                augment=False,
            )

        # Build tasks with class weights from training data
        class_weights: dict[str, torch.Tensor] | None = None
        if config.use_class_weights:
            class_weights = train_dataset.compute_class_weights()
            logger.info("Using class weights for imbalanced data")

        tasks = _create_lumbar_spine_tasks(
            label_smoothing=config.label_smoothing,
            class_weights=class_weights,
        )

        # Create model if not provided
        if model is None:
            model = MultiTaskClassifier(
                backbone=config.backbone,
                tasks=tasks,
                pretrained=config.pretrained,
                dropout=config.dropout,
                freeze_backbone=config.freeze_backbone_epochs > 0,
            )

        super().__init__(config, model, train_dataset, val_dataset)

        # Metrics
        self.metrics = MTLClassificationMetrics()

        # Visualizer
        self.visualizer = TrainingVisualizer(
            output_path=config.logs_path,
            output_mode="html",
            use_wandb=config.use_wandb,
        )

        # Track backbone freeze state
        self._backbone_unfrozen = config.freeze_backbone_epochs == 0

    def _create_dataloader(
        self,
        dataset: ClassificationDataset,
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
            collate_fn=ClassificationCollator(),
        )

    def _unpack_batch(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, MTLTargets]:
        """Unpack batch into inputs and targets."""
        return batch["image"], batch["targets"]

    def _train_step(self, batch: dict[str, Any]) -> float:
        """Single training step."""
        inputs = batch["image"]
        targets: MTLTargets = batch["targets"].to(self.accelerator.device)

        self.optimizer.zero_grad()

        with self.accelerator.autocast():
            predictions = self.model(inputs)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            loss = unwrapped_model.get_loss(predictions, targets.to_dict())

        self.accelerator.backward(loss)

        if self.config.grad_clip:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )

        self.optimizer.step()

        return loss.item()

    def _validate_epoch(self) -> tuple[float, dict[str, float]]:
        """Validate and compute metrics."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        self.metrics.reset()

        # For visualization
        sample_images: list[np.ndarray] = []

        with torch.no_grad():
            for _, batch in enumerate(self.val_loader):  # type: ignore[union-attr]
                inputs = batch["image"]
                targets: MTLTargets = batch["targets"].to(self.accelerator.device)

                with self.accelerator.autocast():
                    predictions = self.model(inputs)
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    loss = unwrapped_model.get_loss(predictions, targets.to_dict())

                total_loss += loss.item()
                num_batches += 1

                # Update metrics
                self.metrics.update(predictions, targets)

                # Collect samples for visualization
                if (
                    self.config.visualize_predictions
                    and self.accelerator.is_main_process
                    and len(sample_images) < self.config.num_visualization_samples
                ):
                    images_np = self._denormalize_images(inputs)
                    for img in images_np:
                        if len(sample_images) < self.config.num_visualization_samples:
                            sample_images.append(img)

        avg_loss = total_loss / num_batches
        metrics = self.metrics.compute()

        return avg_loss, metrics

    def _compute_metrics(
        self,
        _: torch.Tensor,
        __: torch.Tensor,
    ) -> dict[str, float]:
        """Compute metrics (placeholder for base class compatibility)."""
        return {}

    def _denormalize_images(self, images: torch.Tensor) -> list[np.ndarray]:
        """Denormalize ImageNet-normalized images."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        images = images.cpu()
        images = images * std + mean
        images = torch.clamp(images, 0, 1)

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

    def on_train_end(self, _: TrainingResult) -> None:
        """Generate final visualizations."""
        if self.accelerator.is_main_process:
            self._generate_final_visualizations()

    def get_metric_for_checkpoint(
        self,
        val_loss: float | None,
        metrics: dict[str, float],
    ) -> float:
        """Use overall accuracy for checkpointing (negated for lower-is-better)."""
        if "overall_accuracy" in metrics:
            return -metrics["overall_accuracy"]
        if val_loss is not None:
            return val_loss
        return self.history["train_loss"][-1] if self.history["train_loss"] else float("inf")

    def _generate_final_visualizations(self) -> None:
        """Generate final training visualizations."""
        self.visualizer.plot_training_curves(
            self.history,
            filename="training_curves",
        )

        logger.info(f"Visualizations saved to: {self.config.logs_path}")

    def evaluate(
        self,
        test_dataset: ClassificationDataset | None = None,
    ) -> dict[str, float]:
        """Evaluate model on test set."""
        if test_dataset is None:
            test_dataset = ClassificationDataset(
                data_path=self.config.data_path,
                split="test",
                val_ratio=self.config.val_split,
                levels=self.config.levels,
                output_size=self.config.output_size,
                augment=False,
            )

        test_loader = self._create_dataloader(test_dataset, shuffle=False)
        test_loader = self.accelerator.prepare(test_loader)

        self.model.eval()
        self.metrics.reset()

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["image"]
                targets: MTLTargets = batch["targets"]

                predictions = self.model(inputs)
                self.metrics.update(predictions, targets)

        metrics = self.metrics.compute()

        logger.info("Test Results:")
        for key, value in sorted(metrics.items()):
            logger.info(f"  {key}: {value:.4f}")

        # Log to wandb
        if self._wandb_initialized:
            wandb_metrics = {f"test/{key}": value for key, value in metrics.items()}
            self._log_to_wandb(wandb_metrics)

        return metrics

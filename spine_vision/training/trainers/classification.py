"""Classification trainer for multi-task lumbar spine grading.

Specialized trainer for multi-task classification with dual-modality input.
Uses HuggingFace Accelerate and supports trackio logging.

Supports training on all labels (multi-task) or individual labels (single-task)
via the `target_labels` configuration option.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from spine_vision.training.base import BaseTrainer, TrainingConfig, TrainingResult
from spine_vision.training.datasets.classification import (
    AVAILABLE_LABELS,
    ClassificationCollator,
    ClassificationDataset,
    DynamicTargets,
)
from spine_vision.training.metrics import MTLClassificationMetrics
from spine_vision.training.models import MultiTaskClassifier, TaskConfig
from spine_vision.training.registry import register_trainer
from spine_vision.training.visualization import (
    TrainingVisualizer,
    load_classification_original_images,
)


# Task configurations for all lumbar spine labels
_ALL_TASK_CONFIGS: dict[str, TaskConfig] = {
    "pfirrmann": TaskConfig(
        name="pfirrmann",
        num_classes=5,
        task_type="multiclass",
    ),
    "modic": TaskConfig(
        name="modic",
        num_classes=4,
        task_type="multiclass",
    ),
    "herniation": TaskConfig(
        name="herniation",
        num_classes=1,
        task_type="binary",
    ),
    "bulging": TaskConfig(
        name="bulging",
        num_classes=1,
        task_type="binary",
    ),
    "upper_endplate": TaskConfig(
        name="upper_endplate",
        num_classes=1,
        task_type="binary",
    ),
    "lower_endplate": TaskConfig(
        name="lower_endplate",
        num_classes=1,
        task_type="binary",
    ),
    "spondy": TaskConfig(
        name="spondy",
        num_classes=1,
        task_type="binary",
    ),
    "narrowing": TaskConfig(
        name="narrowing",
        num_classes=1,
        task_type="binary",
    ),
}


def _create_lumbar_spine_tasks(
    target_labels: list[str] | None = None,
    label_smoothing: float = 0.1,
    class_weights: dict[str, torch.Tensor] | None = None,
) -> list[TaskConfig]:
    """Create lumbar spine classification tasks with optional filtering.

    Args:
        target_labels: List of label names to include. If None, includes all labels.
        label_smoothing: Label smoothing for multiclass tasks.
        class_weights: Dict mapping task names to class weight tensors.

    Returns:
        List of TaskConfig for the selected lumbar spine classification tasks.

    Raises:
        ValueError: If any target_label is not a valid label name.
    """
    cw = class_weights or {}

    # Determine which labels to include
    if target_labels is None:
        labels_to_use = list(AVAILABLE_LABELS)
    else:
        # Validate target labels
        invalid_labels = set(target_labels) - set(AVAILABLE_LABELS)
        if invalid_labels:
            raise ValueError(
                f"Invalid target labels: {invalid_labels}. "
                f"Available labels: {AVAILABLE_LABELS}"
            )
        labels_to_use = target_labels

    # Build task configs for selected labels
    tasks: list[TaskConfig] = []
    for label in labels_to_use:
        base_config = _ALL_TASK_CONFIGS[label]
        tasks.append(
            TaskConfig(
                name=base_config.name,
                num_classes=base_config.num_classes,
                task_type=base_config.task_type,
                label_smoothing=label_smoothing if base_config.task_type == "multiclass" else 0.0,
                class_weights=cw.get(base_config.name),
            )
        )

    return tasks


class ClassificationConfig(TrainingConfig):
    """Configuration for multi-task classification training.

    Supports training on all labels (multi-task) or individual labels (single-task)
    via the `target_labels` option. When training single labels, only that label's
    head is created and trained.

    Example:
        # Train all labels (default multi-task)
        config = ClassificationConfig()

        # Train only Pfirrmann grade
        config = ClassificationConfig(target_labels=["pfirrmann"])

        # Train multiple specific labels
        config = ClassificationConfig(target_labels=["pfirrmann", "modic"])
    """

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

    series_types: list[str] | None = None
    """Filter to specific series types (e.g., ["sag_t2"] for T2 only).

    If None, requires both T1 and T2 images.
    Options: "sag_t1", "sag_t2".

    Examples:
        - None: Use both T1 and T2 (default, creates [T2, T1, T2] channels)
        - ["sag_t2"]: T2 only (creates [T2, T2, T2] channels)
        - ["sag_t1"]: T1 only (creates [T1, T1, T1] channels)
        - ["sag_t1", "sag_t2"]: Same as None
    """

    target_labels: list[str] | None = None
    """Filter to specific labels for training.

    If None, trains on all 8 labels (multi-task learning).
    If specified, only creates heads for the listed labels.

    Available labels:
        - pfirrmann: 5-class Pfirrmann grade (1-5)
        - modic: 4-class Modic type (0-3)
        - herniation: Binary disc herniation
        - bulging: Binary disc bulging
        - upper_endplate: Binary upper endplate defect
        - lower_endplate: Binary lower endplate defect
        - spondy: Binary spondylolisthesis
        - narrowing: Binary disc narrowing
    """

    output_size: tuple[int, int] = (128, 128)
    """Final input size to model."""

    augment: bool = True

    # Visualization
    visualize_predictions: bool = True
    num_visualization_samples: int = 16
    max_samples_per_cell: int = 4
    """Maximum samples to display per confusion matrix cell."""


@register_trainer("classification", config_cls=ClassificationConfig)
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
                series_types=config.series_types,
                target_labels=config.target_labels,
                output_size=config.output_size,
                augment=config.augment,
            )

        if val_dataset is None:
            val_dataset = ClassificationDataset(
                data_path=config.data_path,
                split="val",
                val_ratio=config.val_split,
                levels=config.levels,
                series_types=config.series_types,
                target_labels=config.target_labels,
                output_size=config.output_size,
                augment=False,
            )

        # Build tasks with class weights from training data
        class_weights: dict[str, torch.Tensor] | None = None
        if config.use_class_weights:
            class_weights = train_dataset.compute_class_weights()
            logger.info("Using class weights for imbalanced data")

        tasks = _create_lumbar_spine_tasks(
            target_labels=config.target_labels,
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

        # Determine which labels are being trained
        self._target_labels = config.target_labels or list(AVAILABLE_LABELS)

        # Metrics (only for labels being trained)
        self.metrics = MTLClassificationMetrics(target_labels=self._target_labels)

        # Visualizer
        self.visualizer = TrainingVisualizer(
            output_path=config.logs_path,
            output_mode="html",
            use_trackio=config.use_trackio,
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
    ) -> tuple[torch.Tensor, DynamicTargets]:
        """Unpack batch into inputs and targets."""
        return batch["image"], batch["targets"]

    def _train_step(self, batch: dict[str, Any]) -> float:
        """Single training step."""
        inputs = batch["image"]
        targets: DynamicTargets = batch["targets"].to(self.accelerator.device)

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

        with torch.no_grad():
            for _, batch in enumerate(self.val_loader):  # type: ignore[union-attr]
                inputs = batch["image"]
                targets: DynamicTargets = batch["targets"].to(self.accelerator.device)

                with self.accelerator.autocast():
                    predictions = self.model(inputs)
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    loss = unwrapped_model.get_loss(predictions, targets.to_dict())

                total_loss += loss.item()
                num_batches += 1

                # Update metrics
                self.metrics.update(predictions, targets)

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
        """Log dataset stats, target labels, and freeze info at training start."""
        # Log target labels
        if len(self._target_labels) == len(AVAILABLE_LABELS):
            logger.info("Training on all labels (multi-task)")
        else:
            logger.info(f"Training on selected labels: {self._target_labels}")

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

        # Visualize test samples with predictions if enabled
        if self.config.visualize_predictions:
            self._visualize_test_samples()

        logger.info(f"Visualizations saved to: {self.config.logs_path}")

    def _visualize_test_samples(self) -> None:
        """Visualize a batch of test samples with predicted labels overlaid."""
        # Create test dataset
        test_dataset = ClassificationDataset(
            data_path=self.config.data_path,
            split="test",
            val_ratio=self.config.val_split,
            levels=self.config.levels,
            series_types=self.config.series_types,
            target_labels=self.config.target_labels,
            output_size=self.config.output_size,
            augment=False,
        )

        if len(test_dataset) == 0:
            logger.warning("No test samples available for visualization")
            return

        self.evaluate(test_dataset, visualize=True)

    def evaluate(
        self,
        test_dataset: ClassificationDataset | None = None,
        visualize: bool = False,
        max_samples_per_cell: int | None = None,
    ) -> dict[str, float]:
        """Evaluate model on test set.

        Args:
            test_dataset: Optional test dataset. If None, creates from config.
            visualize: If True, generate confusion matrix with samples visualizations.
            max_samples_per_cell: Maximum samples per confusion matrix cell.
                Defaults to config.max_samples_per_cell.

        Returns:
            Dictionary of evaluation metrics.
        """
        if test_dataset is None:
            test_dataset = ClassificationDataset(
                data_path=self.config.data_path,
                split="test",
                val_ratio=self.config.val_split,
                levels=self.config.levels,
                series_types=self.config.series_types,
                target_labels=self.config.target_labels,
                output_size=self.config.output_size,
                augment=False,
            )

        test_loader = self._create_dataloader(test_dataset, shuffle=False)
        test_loader = self.accelerator.prepare(test_loader)

        self.model.eval()
        self.metrics.reset()

        # For visualization - collect metadata only, load original images later
        all_predictions: dict[str, list[np.ndarray]] = {label: [] for label in self._target_labels}
        all_targets: dict[str, list[np.ndarray]] = {label: [] for label in self._target_labels}
        all_metadata: list[dict[str, Any]] = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["image"]
                targets: DynamicTargets = batch["targets"]
                metadata_list: list[dict[str, Any]] = batch["metadata"]

                predictions = self.model(inputs)
                self.metrics.update(predictions, targets)

                # Collect metadata and predictions for visualization
                if visualize and self.accelerator.is_main_process:
                    all_metadata.extend(metadata_list)

                    for label in self._target_labels:
                        if label in predictions:
                            pred_tensor = predictions[label]
                            task_config = _ALL_TASK_CONFIGS.get(label)
                            is_binary = task_config and task_config.task_type == "binary"

                            if is_binary:
                                # Binary: apply sigmoid to convert logits to probabilities
                                pred_np = torch.sigmoid(pred_tensor).cpu().numpy()
                            elif pred_tensor.dim() == 1:
                                # 1D multiclass (unlikely but handle edge case)
                                pred_np = pred_tensor.cpu().numpy()
                            else:
                                # Multiclass: apply softmax
                                pred_np = torch.softmax(pred_tensor, dim=-1).cpu().numpy()
                            all_predictions[label].extend(pred_np)

                        if label in targets:
                            target_tensor = getattr(targets, label)
                            all_targets[label].extend(target_tensor.cpu().numpy())

        metrics = self.metrics.compute()

        logger.info("Test Results:")
        for key, value in sorted(metrics.items()):
            logger.info(f"  {key}: {value:.4f}")

        # Log to trackio
        if self._trackio_initialized:
            trackio_metrics = {f"test/{key}": value for key, value in metrics.items()}
            self._log_to_trackio(trackio_metrics)

        # Generate visualizations if requested
        if visualize and self.accelerator.is_main_process and all_metadata:
            # Load all original images for confusion analysis
            all_original_images = load_classification_original_images(
                data_path=self.config.data_path,
                metadata_list=all_metadata,
                output_size=self.config.output_size,
            )
            all_pred_arrays = {k: np.array(v) for k, v in all_predictions.items()}
            all_target_arrays = {k: np.array(v) for k, v in all_targets.items()}

            # Determine max samples per cell
            samples_per_cell = max_samples_per_cell or self.config.max_samples_per_cell

            # Plot per-label metrics
            self.visualizer.plot_classification_metrics(
                metrics=metrics,
                target_labels=self._target_labels,
                filename="test_metrics",
            )

            # Plot confusion matrices with samples for each label
            # This replaces the old test_samples_with_labels and confusion_examples
            self.visualizer.plot_confusion_matrices_with_samples(
                images=all_original_images,
                predictions=all_pred_arrays,
                targets=all_target_arrays,
                target_labels=self._target_labels,
                metadata=all_metadata,
                max_samples_per_cell=samples_per_cell,
                filename_prefix="confusion_matrix_samples",
            )

            # Plot confusion summary bar chart
            self.visualizer.plot_confusion_summary(
                predictions=all_pred_arrays,
                targets=all_target_arrays,
                target_labels=self._target_labels,
                filename="confusion_summary",
            )

            logger.info(f"Test visualizations saved to: {self.config.logs_path}")

        return metrics

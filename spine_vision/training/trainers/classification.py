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

from spine_vision.core.tasks import (
    AVAILABLE_TASK_NAMES,
    TaskConfig,
    get_strategy,
    get_task,
)
from spine_vision.training.datasets.classification import (
    ClassificationCollator,
    ClassificationDataset,
    DynamicTargets,
    create_weighted_sampler,
)
from spine_vision.training.metrics import ClassifierMetrics
from spine_vision.training.models import Classifier
from spine_vision.training.registry import register_trainer
from spine_vision.training.trainers.base import (
    BaseTrainer,
    TrainingConfig,
    TrainingResult,
    _create_worker_init_fn,
)
from spine_vision.visualization import (
    TrainingVisualizer,
    load_classification_original_images,
)


def _create_tasks_for_training(
    target_labels: list[str] | None = None,
    label_smoothing: float = 0.1,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
) -> list[TaskConfig]:
    """Create task configs for training with optional training-time overrides.

    Args:
        target_labels: List of label names to include. If None, includes all.
        label_smoothing: Label smoothing for multiclass tasks.
        use_focal_loss: Use Focal Loss for binary tasks.
        focal_gamma: Focusing parameter for Focal Loss.
        focal_alpha: Optional class weight for Focal Loss.

    Returns:
        List of TaskConfig with training-time settings applied.
    """
    if target_labels is None:
        labels_to_use = list(AVAILABLE_TASK_NAMES)
    else:
        invalid = set(target_labels) - set(AVAILABLE_TASK_NAMES)
        if invalid:
            raise ValueError(
                f"Invalid target labels: {invalid}. Available: {AVAILABLE_TASK_NAMES}"
            )
        labels_to_use = target_labels

    tasks: list[TaskConfig] = []
    for label in labels_to_use:
        task = get_task(label)
        overrides: dict[str, Any] = {}

        if task.is_multiclass:
            overrides["label_smoothing"] = label_smoothing
        elif task.is_binary:
            overrides["use_focal_loss"] = use_focal_loss
            overrides["focal_gamma"] = focal_gamma
            overrides["focal_alpha"] = focal_alpha

        tasks.append(task.with_overrides(**overrides) if overrides else task)

    return tasks


class ClassificationConfig(TrainingConfig):
    """Configuration for multi-task classification training.

    Supports training on all labels (multi-task) or individual labels (single-task)
    via the `target_labels` option.

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

    # Model configuration
    backbone: str = "resnet18"
    pretrained: bool = True
    dropout: float = 0.3
    freeze_backbone_epochs: int = 0
    label_smoothing: float = 0.1

    use_weighted_sampling: bool = True
    """Use weighted sampling to handle class imbalance."""

    sampler_label: str | None = None
    """Label for computing sample weights. If None, uses first target label."""

    # Dataset configuration
    levels: list[str] | None = None
    """Filter to specific IVD levels."""

    series_types: list[str] | None = None
    """Filter to specific series types (e.g., ["sag_t2"])."""

    target_labels: list[str] | None = None
    """Filter to specific labels for training. If None, trains on all."""

    output_size: tuple[int, int] = (256, 256)
    augment: bool = True

    # Loss configuration
    use_focal_loss: bool = False
    """Use Focal Loss for binary tasks."""

    focal_gamma: float = 2.0
    """Focusing parameter for Focal Loss."""

    focal_alpha: float | None = None
    """Optional class weight for Focal Loss."""

    # Visualization
    visualize_predictions: bool = True
    num_visualization_samples: int = 16
    max_samples_per_cell: int = 4


@register_trainer("classification", config_cls=ClassificationConfig)
class ClassificationTrainer(
    BaseTrainer[ClassificationConfig, Classifier, ClassificationDataset]
):
    """Trainer for multi-task lumbar spine classification.

    Uses Classifier with configurable backbone and classification heads.
    Supports dual-modality input (T1 + T2 crops).
    """

    def __init__(
        self,
        config: ClassificationConfig,
        model: Classifier | None = None,
        train_dataset: ClassificationDataset | None = None,
        val_dataset: ClassificationDataset | None = None,
    ) -> None:
        """Initialize trainer."""
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

        # Determine which labels are being trained
        target_labels = config.target_labels or list(AVAILABLE_TASK_NAMES)

        # Create weighted sampler for handling class imbalance
        self._sampler = None
        if config.use_weighted_sampling:
            sampler_label = config.sampler_label or target_labels[0]
            self._sampler = create_weighted_sampler(train_dataset, sampler_label)
            logger.info(f"Using weighted sampling based on '{sampler_label}' label")

        # Build tasks with training-time settings
        tasks = _create_tasks_for_training(
            target_labels=config.target_labels,
            label_smoothing=config.label_smoothing,
            use_focal_loss=config.use_focal_loss,
            focal_gamma=config.focal_gamma,
            focal_alpha=config.focal_alpha,
        )

        # Create model if not provided
        if model is None:
            model = Classifier(
                backbone=config.backbone,
                tasks=tasks,
                pretrained=config.pretrained,
                dropout=config.dropout,
                freeze_backbone=config.freeze_backbone_epochs > 0,
            )

        super().__init__(config, model, train_dataset, val_dataset)

        self._target_labels = target_labels
        self._tasks = tasks
        self.metrics = ClassifierMetrics(target_labels=self._target_labels)

        self.visualizer = TrainingVisualizer(
            output_path=config.logs_path,
            output_mode="image",
            use_trackio=config.use_trackio,
        )

        self._backbone_unfrozen = config.freeze_backbone_epochs == 0

    def _create_dataloader(
        self,
        dataset: ClassificationDataset,
        shuffle: bool = True,
    ) -> DataLoader[Any]:
        """Create DataLoader with custom collator."""
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)

        sampler = self._sampler if shuffle and self._sampler is not None else None
        effective_shuffle = False if sampler is not None else shuffle

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=effective_shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=shuffle,
            collate_fn=ClassificationCollator(),
            worker_init_fn=_create_worker_init_fn(self.config.seed),
            generator=generator if sampler is None else None,
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
                self.metrics.update(predictions, targets)

        avg_loss = total_loss / num_batches
        metrics = self.metrics.compute()

        return avg_loss, metrics

    def _compute_metrics(
        self,
        _: torch.Tensor,
        __: torch.Tensor,
    ) -> dict[str, float]:
        """Placeholder for base class compatibility."""
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
        """Log dataset stats at training start."""
        if len(self._target_labels) == len(AVAILABLE_TASK_NAMES):
            logger.info("Training on all labels (multi-task)")
        else:
            logger.info(f"Training on selected labels: {self._target_labels}")

        if self.config.freeze_backbone_epochs > 0:
            logger.info(
                f"Backbone frozen for first {self.config.freeze_backbone_epochs} epochs"
            )

        stats = self.train_dataset.get_stats()
        logger.info(f"Train dataset stats: {stats}")
        self._visualize_label_distribution()

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
        """Use F1 score for checkpointing (negated for lower-is-better)."""
        if "f1" in metrics:
            return -metrics["f1"]
        if "macro_f1" in metrics:
            return -metrics["macro_f1"]
        if val_loss is not None:
            return val_loss
        return self.history["train_loss"][-1] if self.history["train_loss"] else float("inf")

    def _generate_final_visualizations(self) -> None:
        """Generate final training visualizations."""
        self.visualizer.plot_training_curves(
            self.history,
            filename="training_curves",
        )

        if self.config.visualize_predictions:
            self._visualize_test_samples()

        logger.info(f"Visualizations saved to: {self.config.logs_path}")

    def _visualize_label_distribution(self) -> None:
        """Visualize label distribution across splits."""
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

        distributions: dict[str, dict[str, dict[int | str, int]]] = {
            "train": self.train_dataset.get_label_distribution(),
            "test": test_dataset.get_label_distribution(),
        }

        if self.val_dataset is not None:
            distributions["val"] = self.val_dataset.get_label_distribution()
            val_size = len(self.val_dataset)
        else:
            val_size = 0

        logger.info(
            f"Split sizes - Train: {len(self.train_dataset)}, "
            f"Val: {val_size}, Test: {len(test_dataset)}"
        )

        self.visualizer.plot_label_distribution(
            distributions=distributions,
            target_labels=self._target_labels,
            filename="label_distribution",
        )

    def _visualize_test_samples(self) -> None:
        """Visualize test samples with predictions."""
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
        """Evaluate model on test set."""
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

                if visualize and self.accelerator.is_main_process:
                    all_metadata.extend(metadata_list)

                    for label in self._target_labels:
                        if label in predictions:
                            pred_tensor = predictions[label]
                            task = get_task(label)
                            strategy = get_strategy(task)
                            probs = strategy.compute_probabilities(pred_tensor)
                            all_predictions[label].extend(probs.cpu().numpy())

                        if label in targets:
                            target_tensor = getattr(targets, label)
                            all_targets[label].extend(target_tensor.cpu().numpy())

        metrics = self.metrics.compute()

        logger.info("Test Results:")
        for key, value in sorted(metrics.items()):
            logger.info(f"  {key}: {value:.4f}")

        if self._trackio_initialized:
            trackio_metrics = {f"test/{key}": value for key, value in metrics.items()}
            self._log_to_trackio(trackio_metrics)

        if visualize and self.accelerator.is_main_process and all_metadata:
            all_original_images = load_classification_original_images(
                data_path=self.config.data_path,
                metadata_list=all_metadata,
                output_size=self.config.output_size,
            )
            all_pred_arrays = {k: np.array(v) for k, v in all_predictions.items()}
            all_target_arrays = {k: np.array(v) for k, v in all_targets.items()}

            samples_per_cell = max_samples_per_cell or self.config.max_samples_per_cell

            self.visualizer.plot_classification_metrics(
                metrics=metrics,
                target_labels=self._target_labels,
                filename="test_metrics",
            )

            self.visualizer.plot_confusion_matrices_with_samples(
                images=all_original_images,
                predictions=all_pred_arrays,
                targets=all_target_arrays,
                target_labels=self._target_labels,
                metadata=all_metadata,
                max_samples_per_cell=samples_per_cell,
                filename_prefix="confusion_matrix_samples",
            )

            self.visualizer.plot_confusion_summary(
                predictions=all_pred_arrays,
                targets=all_target_arrays,
                target_labels=self._target_labels,
                filename="confusion_summary",
            )

            logger.info(f"Test visualizations saved to: {self.config.logs_path}")

        return metrics

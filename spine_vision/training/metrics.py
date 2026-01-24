"""Metrics for evaluating model performance.

Provides metric calculators for different tasks with support for
per-class breakdown and aggregation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from spine_vision.datasets.labels import AVAILABLE_LABELS, LABEL_INFO

if TYPE_CHECKING:
    from spine_vision.training.models.generic import TaskConfig


@dataclass
class MetricResult:
    """Container for metric computation results."""

    name: str
    value: float
    per_class: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseMetrics(ABC):
    """Abstract base class for metrics calculators."""

    @abstractmethod
    def compute(
        self,
        predictions: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute metrics.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional arguments (e.g., metadata).

        Returns:
            Dictionary of metric name to value.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset accumulated state."""
        ...


class LocalizationMetrics(BaseMetrics):
    """Metrics for coordinate localization tasks.

    Computes:
    - Mean Euclidean Distance (MED)
    - Mean Absolute Error per coordinate (MAE_x, MAE_y)
    - Percentage of predictions within threshold (PCK)
    - Per-level breakdown
    """

    def __init__(
        self,
        pck_thresholds: list[float] | None = None,
        level_names: list[str] | None = None,
    ) -> None:
        """Initialize metrics.

        Args:
            pck_thresholds: Thresholds for PCK (percentage of correct keypoints).
                Values are relative to image size (e.g., 0.05 = 5% of image).
            level_names: Names for per-level breakdown.
        """
        self.pck_thresholds = pck_thresholds or [0.02, 0.05, 0.10]
        self.level_names = level_names or ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

        # Accumulators for running metrics
        self._predictions: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []
        self._levels: list[np.ndarray] = []

    def reset(self) -> None:
        """Reset accumulated state."""
        self._predictions = []
        self._targets = []
        self._levels = []

    def update(
        self,
        predictions: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        levels: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        """Accumulate predictions for later computation.

        Args:
            predictions: Predicted coordinates [B, 2].
            targets: Ground truth coordinates [B, 2].
            levels: Optional level indices [B].
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if levels is not None and isinstance(levels, torch.Tensor):
            levels = levels.cpu().numpy()

        self._predictions.append(predictions)
        self._targets.append(targets)
        if levels is not None:
            self._levels.append(levels)

    def compute(
        self,
        predictions: torch.Tensor | np.ndarray | None = None,
        targets: torch.Tensor | np.ndarray | None = None,
        levels: torch.Tensor | np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute all metrics.

        Can be called with explicit predictions/targets or use accumulated values.

        Args:
            predictions: Predicted coordinates [N, 2] or None to use accumulated.
            targets: Ground truth coordinates [N, 2] or None to use accumulated.
            levels: Level indices [N] for per-level breakdown.

        Returns:
            Dictionary of metrics.
        """
        # Use accumulated values if not provided
        if predictions is None and self._predictions:
            predictions = np.concatenate(self._predictions, axis=0)
            targets = np.concatenate(self._targets, axis=0)
            if self._levels:
                levels = np.concatenate(self._levels, axis=0)

        if predictions is None or targets is None:
            return {}

        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if levels is not None and isinstance(levels, torch.Tensor):
            levels = levels.cpu().numpy()

        metrics: dict[str, float] = {}

        # Euclidean distances
        distances = np.sqrt(np.sum((predictions - targets) ** 2, axis=1))
        metrics["med"] = float(np.mean(distances))
        metrics["med_std"] = float(np.std(distances))
        metrics["med_median"] = float(np.median(distances))

        # MAE per coordinate
        mae = np.abs(predictions - targets)
        metrics["mae_x"] = float(np.mean(mae[:, 0]))
        metrics["mae_y"] = float(np.mean(mae[:, 1]))
        metrics["mae"] = float(np.mean(mae))

        # PCK at various thresholds
        for thresh in self.pck_thresholds:
            pck = float(np.mean(distances < thresh) * 100)
            metrics[f"pck@{thresh:.2f}"] = pck

        # Per-level breakdown
        if levels is not None:
            for level_idx, level_name in enumerate(self.level_names):
                mask = levels == level_idx
                if np.sum(mask) > 0:
                    level_dist = distances[mask]
                    metrics[f"med_{level_name}"] = float(np.mean(level_dist))

        return metrics

    def compute_detailed(
        self,
        predictions: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        levels: torch.Tensor | np.ndarray | None = None,
    ) -> MetricResult:
        """Compute metrics with detailed breakdown.

        Args:
            predictions: Predicted coordinates [N, 2].
            targets: Ground truth coordinates [N, 2].
            levels: Level indices [N].

        Returns:
            MetricResult with per-class breakdown.
        """
        metrics = self.compute(predictions, targets, levels)

        # Extract per-level metrics
        per_class = {}
        for level_name in self.level_names:
            key = f"med_{level_name}"
            if key in metrics:
                per_class[level_name] = metrics[key]

        return MetricResult(
            name="LocalizationMetrics",
            value=metrics.get("med", 0.0),
            per_class=per_class,
            metadata=metrics,
        )


class ClassificationMetrics(BaseMetrics):
    """Metrics for classification tasks.

    Computes:
    - Accuracy
    - Balanced Accuracy
    - Per-class Precision, Recall, F1
    - Confusion Matrix
    """

    def __init__(self, num_classes: int, class_names: list[str] | None = None) -> None:
        """Initialize metrics.

        Args:
            num_classes: Number of classes.
            class_names: Optional names for each class.
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

        self._predictions: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []

    def reset(self) -> None:
        self._predictions = []
        self._targets = []

    def update(
        self,
        predictions: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
    ) -> None:
        """Accumulate predictions."""
        if isinstance(predictions, torch.Tensor):
            if predictions.dim() > 1:
                predictions = predictions.argmax(dim=1)
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        self._predictions.append(predictions)
        self._targets.append(targets)

    def compute(
        self,
        predictions: torch.Tensor | np.ndarray | None = None,
        targets: torch.Tensor | np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute all metrics."""
        if predictions is None and self._predictions:
            predictions = np.concatenate(self._predictions, axis=0)
            targets = np.concatenate(self._targets, axis=0)

        if predictions is None or targets is None:
            return {}

        if isinstance(predictions, torch.Tensor):
            if predictions.dim() > 1:
                predictions = predictions.argmax(dim=1)
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        metrics: dict[str, float] = {}

        # Accuracy
        metrics["accuracy"] = float(np.mean(predictions == targets) * 100)

        # Per-class metrics
        for class_idx, class_name in enumerate(self.class_names):
            pred_mask = predictions == class_idx
            target_mask = targets == class_idx

            tp = np.sum(pred_mask & target_mask)
            fp = np.sum(pred_mask & ~target_mask)
            fn = np.sum(~pred_mask & target_mask)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            metrics[f"precision_{class_name}"] = float(precision)
            metrics[f"recall_{class_name}"] = float(recall)
            metrics[f"f1_{class_name}"] = float(f1)

        # Balanced accuracy (mean per-class recall)
        recalls = [metrics[f"recall_{name}"] for name in self.class_names]
        metrics["balanced_accuracy"] = float(np.mean(recalls) * 100)

        # Macro F1
        f1s = [metrics[f"f1_{name}"] for name in self.class_names]
        metrics["macro_f1"] = float(np.mean(f1s))

        return metrics


class ClassifierMetrics:
    """Metrics calculator for Classifier model.

    Computes metrics for each task head based on task type from LABEL_INFO:
    - multiclass: accuracy, balanced accuracy, macro F1
    - binary: accuracy, precision, recall, F1

    Provides aggregate metrics:
    - overall_accuracy: mean accuracy across all tasks
    - macro_f1: mean F1 across all tasks (used for checkpoint selection)

    Args:
        tasks: List of TaskConfig objects from Classifier model.
        target_labels: List of task names to filter to.

    Example:
        metrics = ClassifierMetrics(target_labels=["herniation", "pfirrmann"])
        metrics.update(predictions, targets)
        result = metrics.compute()
        # result["macro_f1"] is used for checkpoint selection
    """

    def __init__(
        self,
        tasks: list[TaskConfig] | None = None,
        target_labels: list[str] | None = None,
    ) -> None:
        """Initialize metrics for specified tasks.

        Args:
            tasks: TaskConfig list from Classifier model. If provided, uses task
                types from these configs. Otherwise uses LABEL_INFO.
            target_labels: Filter to these task names. If None, uses all labels.
        """
        # Determine which labels to track
        if target_labels is None:
            labels_to_track = list(AVAILABLE_LABELS)
        else:
            labels_to_track = list(target_labels)

        # Build task type mapping from tasks or LABEL_INFO
        task_types: dict[str, str] = {}
        num_classes: dict[str, int] = {}

        if tasks is not None:
            for task in tasks:
                if task.name in labels_to_track:
                    task_types[task.name] = task.task_type
                    num_classes[task.name] = task.num_classes
        else:
            for label in labels_to_track:
                if label in LABEL_INFO:
                    task_types[label] = LABEL_INFO[label]["type"]
                    num_classes[label] = LABEL_INFO[label]["num_classes"]

        self._task_types = task_types

        # Initialize metrics based on task type
        self._multiclass_metrics: dict[str, ClassificationMetrics] = {}
        self._binary_preds: dict[str, list[np.ndarray]] = {}
        self._binary_targets: dict[str, list[np.ndarray]] = {}

        for label, task_type in task_types.items():
            if task_type == "multiclass":
                n_classes = num_classes[label]
                self._multiclass_metrics[label] = ClassificationMetrics(
                    num_classes=n_classes,
                    class_names=[f"class_{i}" for i in range(n_classes)],
                )
            elif task_type == "binary":
                self._binary_preds[label] = []
                self._binary_targets[label] = []

    def reset(self) -> None:
        """Reset all accumulators."""
        for metrics in self._multiclass_metrics.values():
            metrics.reset()
        for label in self._binary_preds:
            self._binary_preds[label] = []
            self._binary_targets[label] = []

    def update(
        self,
        predictions: Any,  # dict[str, Tensor] or object with attributes
        targets: Any,  # dict[str, Tensor] or object with attributes
    ) -> None:
        """Accumulate predictions from a batch.

        Args:
            predictions: Dict with task tensors or object with attribute access.
            targets: Dict with task tensors or object with attribute access.
        """
        # Support both dict and object access
        def get_value(obj: Any, key: str) -> torch.Tensor | None:
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        # Multiclass tasks
        for label, metrics in self._multiclass_metrics.items():
            pred = get_value(predictions, label)
            target = get_value(targets, label)
            if pred is not None and target is not None:
                pred_classes = pred.argmax(dim=1).cpu().numpy()
                metrics.update(pred_classes, target.cpu().numpy())

        # Binary tasks
        for label in self._binary_preds:
            pred = get_value(predictions, label)
            target = get_value(targets, label)
            if pred is not None and target is not None:
                self._binary_preds[label].append(
                    torch.sigmoid(pred).cpu().numpy()
                )
                self._binary_targets[label].append(target.cpu().numpy())

    @property
    def is_single_task(self) -> bool:
        """Check if this is a single-task setup."""
        return len(self._task_types) == 1

    def compute(self) -> dict[str, float]:
        """Compute all metrics.

        Returns:
            Dictionary containing:
            - {task}_accuracy: Accuracy for each task
            - {task}_precision, {task}_recall, {task}_f1: For binary tasks
            - {task}_balanced_acc: For multiclass tasks
            - overall_accuracy: Mean accuracy across all tasks
            - f1: F1 score (single-task only, for checkpoint selection)
            - macro_f1: Mean F1 across tasks (multi-task only, for checkpoint selection)
        """
        metrics: dict[str, float] = {}
        f1_scores: list[float] = []

        # Multiclass metrics
        for label, task_metrics in self._multiclass_metrics.items():
            computed = task_metrics.compute()
            if computed:
                metrics[f"{label}_accuracy"] = computed.get("accuracy", 0.0)
                metrics[f"{label}_balanced_acc"] = computed.get(
                    "balanced_accuracy", 0.0
                )
                # Use macro_f1 from ClassificationMetrics for multiclass tasks
                task_f1 = computed.get("macro_f1", 0.0)
                f1_scores.append(task_f1)

        # Binary metrics
        for label, preds_list in self._binary_preds.items():
            if not preds_list:
                continue

            preds = np.concatenate(preds_list, axis=0).flatten()
            targets = np.concatenate(self._binary_targets[label], axis=0).flatten()

            # Binary predictions at 0.5 threshold
            pred_binary = (preds > 0.5).astype(int)
            t_binary = targets.astype(int)

            # Accuracy
            acc = np.mean(pred_binary == t_binary) * 100
            metrics[f"{label}_accuracy"] = float(acc)

            # Precision, Recall, F1
            tp = np.sum((pred_binary == 1) & (t_binary == 1))
            fp = np.sum((pred_binary == 1) & (t_binary == 0))
            fn = np.sum((pred_binary == 0) & (t_binary == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            metrics[f"{label}_precision"] = float(precision)
            metrics[f"{label}_recall"] = float(recall)
            metrics[f"{label}_f1"] = float(f1)
            f1_scores.append(f1)

        # Aggregate metrics
        accs = [v for k, v in metrics.items() if k.endswith("_accuracy")]
        if accs:
            metrics["overall_accuracy"] = float(np.mean(accs))
        else:
            metrics["overall_accuracy"] = 0.0

        # F1 metric for checkpoint selection
        if f1_scores:
            if self.is_single_task:
                # Single task: use "f1" directly
                metrics["f1"] = float(f1_scores[0])
            else:
                # Multi-task: use "macro_f1" (average across tasks)
                metrics["macro_f1"] = float(np.mean(f1_scores))

        return metrics

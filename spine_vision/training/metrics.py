"""Metrics for evaluating model performance.

Provides metric calculators for different tasks with support for
per-class breakdown and aggregation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


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


class MTLClassificationMetrics:
    """Metrics calculator for multi-task classification.

    Computes metrics for each task head. Supports dynamic label selection.

    - Multiclass labels (pfirrmann, modic): accuracy, balanced accuracy
    - Binary labels: accuracy, precision, recall, F1

    Args:
        target_labels: List of label names to compute metrics for.
            If None, computes metrics for all 8 labels.
    """

    # All available labels with their types
    _MULTICLASS_LABELS = {"pfirrmann": 5, "modic": 4}
    _BINARY_LABELS = {
        "herniation", "bulging", "upper_endplate",
        "lower_endplate", "spondy", "narrowing"
    }

    def __init__(self, target_labels: list[str] | None = None) -> None:
        """Initialize metrics for specified labels.

        Args:
            target_labels: Labels to compute metrics for. If None, uses all labels.
        """
        if target_labels is None:
            target_labels = list(self._MULTICLASS_LABELS.keys()) + list(self._BINARY_LABELS)
        self._target_labels = set(target_labels)

        # Initialize multiclass metrics only for target labels
        self._multiclass_metrics: dict[str, ClassificationMetrics] = {}
        if "pfirrmann" in self._target_labels:
            self._multiclass_metrics["pfirrmann"] = ClassificationMetrics(
                num_classes=5,
                class_names=[f"Grade_{i+1}" for i in range(5)],
            )
        if "modic" in self._target_labels:
            self._multiclass_metrics["modic"] = ClassificationMetrics(
                num_classes=4,
                class_names=[f"Type_{i}" for i in range(4)],
            )

        # Initialize binary accumulators only for target labels
        self._binary_preds: dict[str, list[np.ndarray]] = {}
        self._binary_targets: dict[str, list[np.ndarray]] = {}
        for label in self._BINARY_LABELS:
            if label in self._target_labels:
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
        predictions: Any,  # dict[str, Tensor] or DynamicTargets
        targets: Any,  # dict[str, Tensor] or DynamicTargets
    ) -> None:
        """Accumulate predictions from a batch.

        Args:
            predictions: Dict with task tensors or object with attribute access.
            targets: Dict with task tensors or object with attribute access.
        """
        # Support both dict and object access
        def get_pred(key: str) -> torch.Tensor | None:
            if isinstance(predictions, dict):
                return predictions.get(key)
            return getattr(predictions, key, None)

        def get_target(key: str) -> torch.Tensor | None:
            if isinstance(targets, dict):
                return targets.get(key)
            return getattr(targets, key, None)

        # Multiclass heads
        for label, metrics in self._multiclass_metrics.items():
            pred = get_pred(label)
            target = get_target(label)
            if pred is not None and target is not None:
                pred_classes = pred.argmax(dim=1).cpu().numpy()
                metrics.update(pred_classes, target.cpu().numpy())

        # Binary heads
        for label in self._binary_preds:
            pred = get_pred(label)
            target = get_target(label)
            if pred is not None and target is not None:
                self._binary_preds[label].append(
                    torch.sigmoid(pred).cpu().numpy()
                )
                self._binary_targets[label].append(target.cpu().numpy())

    def compute(self) -> dict[str, float]:
        """Compute all metrics."""
        metrics: dict[str, float] = {}

        # Multiclass metrics
        for label, label_metrics in self._multiclass_metrics.items():
            computed = label_metrics.compute()
            if computed:
                metrics[f"{label}_accuracy"] = computed.get("accuracy", 0.0)
                metrics[f"{label}_balanced_acc"] = computed.get("balanced_accuracy", 0.0)

        # Binary metrics
        for label, preds_list in self._binary_preds.items():
            if not preds_list:
                continue

            preds = np.concatenate(preds_list, axis=0).flatten()
            targets = np.concatenate(self._binary_targets[label], axis=0).flatten()

            # Binary predictions
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

        # Overall average accuracy (only for present labels)
        accs = [v for k, v in metrics.items() if k.endswith("_accuracy")]
        if accs:
            metrics["overall_accuracy"] = float(np.mean(accs))
        else:
            metrics["overall_accuracy"] = 0.0

        return metrics


class RegressionMetrics(BaseMetrics):
    """Metrics for general regression tasks.

    Computes:
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    - R² Score
    """

    def __init__(self) -> None:
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
        if isinstance(predictions, torch.Tensor):
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
        if predictions is None and self._predictions:
            predictions = np.concatenate(self._predictions, axis=0)
            targets = np.concatenate(self._targets, axis=0)

        if predictions is None or targets is None:
            return {}

        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        # Flatten for scalar predictions
        predictions = predictions.flatten()
        targets = targets.flatten()

        mse = float(np.mean((predictions - targets) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(predictions - targets)))

        # R² score
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

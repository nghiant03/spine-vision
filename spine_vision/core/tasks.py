"""Unified task definitions and strategies for classification tasks.

Single source of truth for task types, eliminating scattered conditionals.
Uses strategy pattern for task-type-specific behavior.

Usage:
    from spine_vision.core.tasks import TASK_REGISTRY, get_task, get_strategy

    # Get task config
    task = get_task("pfirrmann")  # TaskConfig

    # Get strategy for task-type-specific behavior
    strategy = get_strategy(task)  # TaskStrategy
    loss_fn = strategy.get_loss_fn(task)
    preds = strategy.compute_predictions(logits)
    metrics = strategy.get_metrics(task)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall

TaskType = Literal["binary", "multiclass", "multilabel", "ordinal", "regression"]


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for a classification task.

    Immutable dataclass that defines task properties. Use `with_overrides()`
    to create modified copies with training-specific settings.

    Attributes:
        name: Task identifier (e.g., "pfirrmann", "herniation").
        num_classes: Number of output classes. For binary tasks, use 1.
        task_type: Classification type determining loss/metrics behavior.
        display_name: Human-readable name for visualization.
        class_names: Names for each class (for multiclass/ordinal).
        color: Color for visualization (hex code).

    Training-time attributes (set via with_overrides):
        label_smoothing: Smoothing factor for CE loss (multiclass only).
        use_focal_loss: Use Focal Loss for binary/multilabel.
        focal_gamma: Focusing parameter for Focal Loss.
        focal_alpha: Class weight for Focal Loss.
        loss_weight: Weight for this task in multi-task loss.
    """

    name: str
    num_classes: int
    task_type: TaskType
    display_name: str = ""
    class_names: tuple[str, ...] = ()
    color: str = "#1f77b4"

    # Training-time settings (immutable defaults, override via with_overrides)
    label_smoothing: float = 0.0
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float | None = None
    loss_weight: float = 1.0

    # Escape hatch for custom behavior
    custom_loss_fn: Callable[[], nn.Module] | None = field(default=None, repr=False)
    custom_metrics_fn: Callable[[], MetricCollection] | None = field(
        default=None, repr=False
    )

    def __post_init__(self) -> None:
        if not self.display_name:
            object.__setattr__(
                self, "display_name", self.name.replace("_", " ").title()
            )
        if not self.class_names and self.task_type == "multiclass":
            names = tuple(f"Class {i}" for i in range(self.num_classes))
            object.__setattr__(self, "class_names", names)

    def with_overrides(self, **kwargs: Any) -> TaskConfig:
        """Create a new TaskConfig with overridden values.

        Example:
            task = get_task("pfirrmann").with_overrides(
                label_smoothing=0.1,
                loss_weight=2.0,
            )
        """
        from dataclasses import asdict

        current = asdict(self)
        current.update(kwargs)
        return TaskConfig(**current)

    @property
    def is_binary(self) -> bool:
        return self.task_type == "binary"

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == "multiclass"


class TaskStrategy(ABC):
    """Strategy interface for task-type-specific behavior.

    Encapsulates all task-type conditionals in one place.
    """

    @abstractmethod
    def get_loss_fn(self, task: TaskConfig) -> nn.Module:
        """Get loss function for this task type."""
        ...

    @abstractmethod
    def compute_predictions(self, logits: Tensor) -> Tensor:
        """Convert logits to discrete predictions."""
        ...

    @abstractmethod
    def compute_probabilities(self, logits: Tensor) -> Tensor:
        """Convert logits to probabilities."""
        ...

    @abstractmethod
    def get_metrics(self, task: TaskConfig) -> MetricCollection:
        """Get metric collection for this task type."""
        ...

    @abstractmethod
    def format_target(self, target: Tensor) -> Tensor:
        """Format target tensor for loss computation."""
        ...


class BinaryStrategy(TaskStrategy):
    """Strategy for binary classification tasks."""

    def get_loss_fn(self, task: TaskConfig) -> nn.Module:
        if task.custom_loss_fn is not None:
            return task.custom_loss_fn()

        if task.use_focal_loss:
            from spine_vision.training.losses import FocalLoss

            return FocalLoss(gamma=task.focal_gamma, alpha=task.focal_alpha)
        return nn.BCEWithLogitsLoss()

    def compute_predictions(self, logits: Tensor) -> Tensor:
        preds = (torch.sigmoid(logits) > 0.5).int()
        if preds.shape[-1] == 1:
            preds = preds.squeeze(-1)
        return preds

    def compute_probabilities(self, logits: Tensor) -> Tensor:
        return torch.sigmoid(logits)

    def get_metrics(self, task: TaskConfig) -> MetricCollection:
        if task.custom_metrics_fn is not None:
            return task.custom_metrics_fn()

        return MetricCollection(
            {
                "accuracy": Accuracy(task="binary"),
                "precision": Precision(task="binary"),
                "recall": Recall(task="binary"),
                "f1": F1Score(task="binary"),
            }
        )

    def format_target(self, target: Tensor) -> Tensor:
        # Binary targets should be float for BCE
        if target.dtype != torch.float32:
            target = target.float()
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        return target


class MulticlassStrategy(TaskStrategy):
    """Strategy for multiclass classification tasks."""

    def get_loss_fn(self, task: TaskConfig) -> nn.Module:
        if task.custom_loss_fn is not None:
            return task.custom_loss_fn()

        return nn.CrossEntropyLoss(label_smoothing=task.label_smoothing)

    def compute_predictions(self, logits: Tensor) -> Tensor:
        return torch.argmax(logits, dim=1)

    def compute_probabilities(self, logits: Tensor) -> Tensor:
        return torch.softmax(logits, dim=1)

    def get_metrics(self, task: TaskConfig) -> MetricCollection:
        if task.custom_metrics_fn is not None:
            return task.custom_metrics_fn()

        return MetricCollection(
            {
                "accuracy": Accuracy(task="multiclass", num_classes=task.num_classes),
                "balanced_accuracy": Accuracy(
                    task="multiclass", num_classes=task.num_classes, average="macro"
                ),
                "macro_f1": F1Score(
                    task="multiclass", num_classes=task.num_classes, average="macro"
                ),
            }
        )

    def format_target(self, target: Tensor) -> Tensor:
        # Multiclass targets should be long for CE
        if target.dtype != torch.int64:
            target = target.long()
        return target


class MultilabelStrategy(TaskStrategy):
    """Strategy for multilabel classification tasks."""

    def get_loss_fn(self, task: TaskConfig) -> nn.Module:
        if task.custom_loss_fn is not None:
            return task.custom_loss_fn()

        if task.use_focal_loss:
            from spine_vision.training.losses import FocalLoss

            return FocalLoss(gamma=task.focal_gamma, alpha=task.focal_alpha)
        return nn.BCEWithLogitsLoss()

    def compute_predictions(self, logits: Tensor) -> Tensor:
        return (torch.sigmoid(logits) > 0.5).int()

    def compute_probabilities(self, logits: Tensor) -> Tensor:
        return torch.sigmoid(logits)

    def get_metrics(self, task: TaskConfig) -> MetricCollection:
        if task.custom_metrics_fn is not None:
            return task.custom_metrics_fn()

        return MetricCollection(
            {
                "accuracy": Accuracy(task="multilabel", num_labels=task.num_classes),
                "f1": F1Score(task="multilabel", num_labels=task.num_classes),
            }
        )

    def format_target(self, target: Tensor) -> Tensor:
        if target.dtype != torch.float32:
            target = target.float()
        return target


class OrdinalStrategy(TaskStrategy):
    """Strategy for ordinal classification tasks.

    Treats ordinal as multiclass for now, but can be extended
    with ordinal-specific losses (e.g., CORAL, cumulative link).
    """

    def get_loss_fn(self, task: TaskConfig) -> nn.Module:
        if task.custom_loss_fn is not None:
            return task.custom_loss_fn()

        # Default to CE, can be extended with ordinal-specific losses
        return nn.CrossEntropyLoss(label_smoothing=task.label_smoothing)

    def compute_predictions(self, logits: Tensor) -> Tensor:
        return torch.argmax(logits, dim=1)

    def compute_probabilities(self, logits: Tensor) -> Tensor:
        return torch.softmax(logits, dim=1)

    def get_metrics(self, task: TaskConfig) -> MetricCollection:
        if task.custom_metrics_fn is not None:
            return task.custom_metrics_fn()

        # Include MAE for ordinal (measures average grade error)
        from torchmetrics import MeanAbsoluteError

        return MetricCollection(
            {
                "accuracy": Accuracy(task="multiclass", num_classes=task.num_classes),
                "mae": MeanAbsoluteError(),
                "macro_f1": F1Score(
                    task="multiclass", num_classes=task.num_classes, average="macro"
                ),
            }
        )

    def format_target(self, target: Tensor) -> Tensor:
        if target.dtype != torch.int64:
            target = target.long()
        return target


class RegressionStrategy(TaskStrategy):
    """Strategy for regression tasks."""

    def get_loss_fn(self, task: TaskConfig) -> nn.Module:
        if task.custom_loss_fn is not None:
            return task.custom_loss_fn()

        return nn.MSELoss()

    def compute_predictions(self, logits: Tensor) -> Tensor:
        return logits

    def compute_probabilities(self, logits: Tensor) -> Tensor:
        # No probabilities for regression
        return logits

    def get_metrics(self, task: TaskConfig) -> MetricCollection:
        if task.custom_metrics_fn is not None:
            return task.custom_metrics_fn()

        from torchmetrics import MeanAbsoluteError, MeanSquaredError

        return MetricCollection(
            {
                "mse": MeanSquaredError(),
                "mae": MeanAbsoluteError(),
            }
        )

    def format_target(self, target: Tensor) -> Tensor:
        if target.dtype != torch.float32:
            target = target.float()
        return target


# Strategy registry
_STRATEGIES: dict[TaskType, TaskStrategy] = {
    "binary": BinaryStrategy(),
    "multiclass": MulticlassStrategy(),
    "multilabel": MultilabelStrategy(),
    "ordinal": OrdinalStrategy(),
    "regression": RegressionStrategy(),
}


def get_strategy(task: TaskConfig | TaskType) -> TaskStrategy:
    """Get the strategy for a task or task type.

    Args:
        task: TaskConfig instance or task type string.

    Returns:
        TaskStrategy for the task type.
    """
    task_type = task.task_type if isinstance(task, TaskConfig) else task
    if task_type not in _STRATEGIES:
        raise ValueError(f"Unknown task type: {task_type}")
    return _STRATEGIES[task_type]


# =============================================================================
# Task Registry - Single Source of Truth
# =============================================================================

# Lumbar spine classification tasks
TASK_REGISTRY: dict[str, TaskConfig] = {
    "pfirrmann": TaskConfig(
        name="pfirrmann",
        num_classes=5,
        task_type="multiclass",
        display_name="Pfirrmann Grade",
        class_names=("Grade I", "Grade II", "Grade III", "Grade IV", "Grade V"),
        color="#1f77b4",
    ),
    "modic": TaskConfig(
        name="modic",
        num_classes=4,
        task_type="multiclass",
        display_name="Modic Type",
        class_names=("Normal", "Type I", "Type II", "Type III"),
        color="#ff7f0e",
    ),
    "herniation": TaskConfig(
        name="herniation",
        num_classes=1,
        task_type="binary",
        display_name="Disc Herniation",
        color="#2ca02c",
    ),
    "bulging": TaskConfig(
        name="bulging",
        num_classes=1,
        task_type="binary",
        display_name="Disc Bulging",
        color="#d62728",
    ),
    "upper_endplate": TaskConfig(
        name="upper_endplate",
        num_classes=1,
        task_type="binary",
        display_name="Upper Endplate Defect",
        color="#9467bd",
    ),
    "lower_endplate": TaskConfig(
        name="lower_endplate",
        num_classes=1,
        task_type="binary",
        display_name="Lower Endplate Defect",
        color="#8c564b",
    ),
    "spondy": TaskConfig(
        name="spondy",
        num_classes=1,
        task_type="binary",
        display_name="Spondylolisthesis",
        color="#e377c2",
    ),
    "narrowing": TaskConfig(
        name="narrowing",
        num_classes=1,
        task_type="binary",
        display_name="Disc Narrowing",
        color="#7f7f7f",
    ),
}

# Convenience aliases
AVAILABLE_TASK_NAMES: tuple[str, ...] = tuple(TASK_REGISTRY.keys())


def get_task(name: str) -> TaskConfig:
    """Get a task configuration by name.

    Args:
        name: Task name (e.g., "pfirrmann", "herniation").

    Returns:
        TaskConfig for the requested task.

    Raises:
        KeyError: If task name is not found.
    """
    if name not in TASK_REGISTRY:
        raise KeyError(f"Unknown task: {name}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[name]


def get_tasks(names: list[str] | None = None) -> list[TaskConfig]:
    """Get multiple task configurations.

    Args:
        names: List of task names. If None, returns all tasks.

    Returns:
        List of TaskConfig instances.
    """
    if names is None:
        return list(TASK_REGISTRY.values())
    return [get_task(name) for name in names]


def register_task(task: TaskConfig) -> None:
    """Register a new task configuration.

    Args:
        task: TaskConfig to register.

    Raises:
        ValueError: If task name already exists.
    """
    if task.name in TASK_REGISTRY:
        raise ValueError(f"Task '{task.name}' already registered")
    TASK_REGISTRY[task.name] = task


# =============================================================================
# Helper functions
# =============================================================================


def create_loss_functions(
    tasks: list[TaskConfig],
) -> tuple[nn.ModuleDict, dict[str, float]]:
    """Create loss functions for multiple tasks.

    Args:
        tasks: List of TaskConfig instances.

    Returns:
        Tuple of (loss_functions ModuleDict, loss_weights dict).
    """
    loss_fns = nn.ModuleDict()
    loss_weights: dict[str, float] = {}

    for task in tasks:
        strategy = get_strategy(task)
        loss_fns[task.name] = strategy.get_loss_fn(task)
        loss_weights[task.name] = task.loss_weight

    return loss_fns, loss_weights


def compute_predictions_for_tasks(
    outputs: dict[str, Tensor],
    tasks: list[TaskConfig],
) -> dict[str, np.ndarray]:
    """Compute predictions for multiple tasks.

    Args:
        outputs: Dict mapping task names to logits.
        tasks: List of TaskConfig instances.

    Returns:
        Dict mapping task names to numpy prediction arrays.
    """
    predictions: dict[str, np.ndarray] = {}
    for task in tasks:
        if task.name not in outputs:
            continue
        strategy = get_strategy(task)
        preds = strategy.compute_predictions(outputs[task.name])
        predictions[task.name] = preds.cpu().numpy()
    return predictions


def compute_probabilities_for_tasks(
    outputs: dict[str, Tensor],
    tasks: list[TaskConfig],
) -> dict[str, np.ndarray]:
    """Compute probabilities for multiple tasks.

    Args:
        outputs: Dict mapping task names to logits.
        tasks: List of TaskConfig instances.

    Returns:
        Dict mapping task names to numpy probability arrays.
    """
    probabilities: dict[str, np.ndarray] = {}
    for task in tasks:
        if task.name not in outputs:
            continue
        strategy = get_strategy(task)
        probs = strategy.compute_probabilities(outputs[task.name])
        probabilities[task.name] = probs.cpu().numpy()
    return probabilities


def get_task_display_name(name: str) -> str:
    """Get display name for a task.

    Args:
        name: Task name.

    Returns:
        Human-readable display name, or the name itself if not found.
    """
    if name in TASK_REGISTRY:
        return TASK_REGISTRY[name].display_name
    return name


def get_task_color(name: str) -> str:
    """Get color for a task.

    Args:
        name: Task name.

    Returns:
        Hex color string, or default gray if not found.
    """
    if name in TASK_REGISTRY:
        return TASK_REGISTRY[name].color
    return "#333333"


def get_task_display_names() -> dict[str, str]:
    """Get display names for all tasks.

    Returns:
        Dict mapping task names to display names.
    """
    return {name: task.display_name for name, task in TASK_REGISTRY.items()}


def get_task_colors() -> dict[str, str]:
    """Get colors for all tasks.

    Returns:
        Dict mapping task names to hex color strings.
    """
    return {name: task.color for name, task in TASK_REGISTRY.items()}

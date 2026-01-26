"""Dataset classes for training.

Provides PyTorch Dataset implementations for various training tasks.
"""

from spine_vision.core.tasks import AVAILABLE_TASK_NAMES, get_task
from spine_vision.training.datasets.classification import (
    ClassificationCollator,
    ClassificationDataset,
    DynamicTargets,
    create_weighted_sampler,
)
from spine_vision.training.datasets.localization import (
    LocalizationCollator,
    LocalizationDataset,
)

__all__ = [
    "LocalizationDataset",
    "LocalizationCollator",
    "ClassificationDataset",
    "ClassificationCollator",
    "DynamicTargets",
    "create_weighted_sampler",
    "AVAILABLE_TASK_NAMES",
    "get_task",
]

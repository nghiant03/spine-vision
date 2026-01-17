"""Dataset classes for training.

Provides PyTorch Dataset implementations for various training tasks.
"""

from spine_vision.datasets.labels import AVAILABLE_LABELS, LABEL_INFO
from spine_vision.training.datasets.classification import (
    ClassificationCollator,
    ClassificationDataset,
    DynamicTargets,
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
    "AVAILABLE_LABELS",
    "LABEL_INFO",
]

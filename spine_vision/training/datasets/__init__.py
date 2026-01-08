"""Dataset classes for training.

Provides PyTorch Dataset implementations for various training tasks.
"""

from spine_vision.training.datasets.classification import (
    AVAILABLE_LABELS,
    LABEL_INFO,
    ClassificationCollator,
    ClassificationDataset,
    DynamicTargets,
)
from spine_vision.training.datasets.ivd_coords import IVDCoordsDataset

__all__ = [
    "IVDCoordsDataset",
    "ClassificationDataset",
    "ClassificationCollator",
    "DynamicTargets",
    "AVAILABLE_LABELS",
    "LABEL_INFO",
]

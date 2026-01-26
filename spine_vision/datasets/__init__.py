"""Dataset creation and conversion utilities.

This module provides functions for creating and converting datasets:
- localization: Create localization dataset
- phenikaa: Preprocess Phenikaa dataset (OCR + matching)
- classification: Create classification dataset (Phenikaa + SPIDER)
- rsna: RSNA dataset utilities

For task-related constants (labels, types, colors), use spine_vision.core.tasks.
"""

from spine_vision.datasets.base import ProcessingResult
from spine_vision.datasets.classification import (
    ClassificationDatasetConfig,
    ClassificationRecord,
    create_classification_dataset,
)
from spine_vision.datasets.levels import (
    IDX_TO_LEVEL,
    LEVEL_NAMES,
    LEVEL_TO_IDX,
    NUM_LEVELS,
)
from spine_vision.datasets.localization import (
    LocalizationDatasetConfig,
    create_localization_dataset,
)
from spine_vision.datasets.phenikaa import (
    PreprocessConfig,
    preprocess_phenikaa,
)
from spine_vision.datasets.rsna import get_series_type, load_series_mapping

__all__ = [
    # Level constants
    "IDX_TO_LEVEL",
    "LEVEL_NAMES",
    "LEVEL_TO_IDX",
    "NUM_LEVELS",
    # Base classes
    "ProcessingResult",
    # Localization dataset
    "LocalizationDatasetConfig",
    "create_localization_dataset",
    # Phenikaa dataset
    "PreprocessConfig",
    "preprocess_phenikaa",
    # Classification dataset
    "ClassificationDatasetConfig",
    "ClassificationRecord",
    "create_classification_dataset",
    # RSNA utilities
    "load_series_mapping",
    "get_series_type",
]

"""Dataset creation and conversion utilities.

This module provides classes and functions for creating and converting datasets:
- localization: Create localization dataset
- phenikaa: Preprocess Phenikaa dataset (OCR + matching)
- classification: Create classification dataset (Phenikaa + SPIDER)
- rsna: RSNA dataset utilities
"""

from spine_vision.datasets.base import (
    BaseProcessor,
    ProcessingResult,
)
from spine_vision.datasets.classification import (
    ClassificationDatasetConfig,
    ClassificationDatasetProcessor,
)
from spine_vision.datasets.classification import main as create_classification_dataset
from spine_vision.datasets.localization import (
    LocalizationDatasetConfig,
    LocalizationDatasetProcessor,
)
from spine_vision.datasets.localization import main as create_localization_dataset
from spine_vision.datasets.phenikaa import (
    PhenikkaaProcessor,
    PreprocessConfig,
)
from spine_vision.datasets.phenikaa import main as preprocess_phenikaa
from spine_vision.datasets.labels import (
    AVAILABLE_LABELS,
    IDX_TO_LEVEL,
    LABEL_COLORS,
    LABEL_DISPLAY_NAMES,
    LABEL_INFO,
    LEVEL_TO_IDX,
)
from spine_vision.datasets.rsna import get_series_type, load_series_mapping

__all__ = [
    # Label constants
    "AVAILABLE_LABELS",
    "IDX_TO_LEVEL",
    "LABEL_COLORS",
    "LABEL_DISPLAY_NAMES",
    "LABEL_INFO",
    "LEVEL_TO_IDX",
    # Base classes
    "BaseProcessor",
    "ProcessingResult",
    # Localization dataset
    "LocalizationDatasetConfig",
    "LocalizationDatasetProcessor",
    "create_localization_dataset",
    # Phenikaa dataset
    "PreprocessConfig",
    "PhenikkaaProcessor",
    "preprocess_phenikaa",
    # Classification dataset
    "ClassificationDatasetConfig",
    "ClassificationDatasetProcessor",
    "create_classification_dataset",
    # RSNA utilities
    "load_series_mapping",
    "get_series_type",
]

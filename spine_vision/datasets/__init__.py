"""Dataset creation and conversion utilities.

This module provides classes and functions for creating and converting datasets:
- nnunet: Convert datasets to nnU-Net format
- ivd_coords: Create IVD coordinates dataset
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
from spine_vision.datasets.ivd_coords import (
    IVDCoordsDatasetProcessor,
    IVDDatasetConfig,
)
from spine_vision.datasets.ivd_coords import main as create_ivd_dataset
from spine_vision.datasets.phenikaa import (
    PhenikkaaProcessor,
    PreprocessConfig,
)
from spine_vision.datasets.phenikaa import main as preprocess_phenikaa
from spine_vision.datasets.rsna import get_series_type, load_series_mapping

__all__ = [
    # Base classes
    "BaseProcessor",
    "ProcessingResult",
    # IVD coordinates dataset
    "IVDDatasetConfig",
    "IVDCoordsDatasetProcessor",
    "create_ivd_dataset",
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

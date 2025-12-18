"""Dataset creation and conversion utilities.

This module provides classes and functions for creating and converting datasets:
- nnunet: Convert datasets to nnU-Net format
- ivd_coords: Create IVD coordinates dataset
- phenikaa: Preprocess Phenikaa dataset (OCR + matching)
- rsna: RSNA dataset utilities
"""

from spine_vision.datasets.nnunet import ConvertConfig, main as convert_to_nnunet
from spine_vision.datasets.ivd_coords import IVDDatasetConfig, main as create_ivd_dataset
from spine_vision.datasets.phenikaa import PreprocessConfig, main as preprocess_phenikaa
from spine_vision.datasets.rsna import load_series_mapping, get_series_type

__all__ = [
    "ConvertConfig",
    "convert_to_nnunet",
    "IVDDatasetConfig",
    "create_ivd_dataset",
    "PreprocessConfig",
    "preprocess_phenikaa",
    "load_series_mapping",
    "get_series_type",
]

"""Dataset creation and conversion utilities.

This module provides classes and functions for creating and converting datasets:
- nnunet: Convert datasets to nnU-Net format
- ivd_coords: Create IVD coordinates dataset
- phenikaa: Preprocess Phenikaa dataset (OCR + matching)
"""

from spine_vision.datasets.nnunet import ConvertConfig, main as convert_to_nnunet
from spine_vision.datasets.ivd_coords import IVDDatasetConfig, main as create_ivd_dataset
from spine_vision.datasets.phenikaa import PreprocessConfig, main as preprocess_phenikaa

__all__ = [
    "ConvertConfig",
    "convert_to_nnunet",
    "IVDDatasetConfig",
    "create_ivd_dataset",
    "PreprocessConfig",
    "preprocess_phenikaa",
]

"""I/O utilities for medical images and tabular data."""

import numpy as np

from spine_vision.io.pdf import pdf_first_page_to_array, pdf_to_arrays, pdf_to_images
from spine_vision.io.readers import (
    read_dicom_file,
    read_dicom_series,
    read_medical_image,
)
from spine_vision.io.tabular import load_tabular_data, write_records_csv
from spine_vision.io.writers import write_medical_image


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize array to 0-255 uint8 range.

    Performs min-max normalization and converts to uint8.

    Args:
        arr: Input array of any numeric dtype.

    Returns:
        Normalized uint8 array with values in [0, 255].
    """
    arr = arr.astype(np.float32)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min > 0:
        arr = (arr - arr_min) / (arr_max - arr_min) * 255
    return arr.astype(np.uint8)


__all__ = [
    "read_medical_image",
    "read_dicom_series",
    "read_dicom_file",
    "write_medical_image",
    "load_tabular_data",
    "write_records_csv",
    "normalize_to_uint8",
    "pdf_to_images",
    "pdf_to_arrays",
    "pdf_first_page_to_array",
]

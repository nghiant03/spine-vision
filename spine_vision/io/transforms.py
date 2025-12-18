"""Image transformation utilities."""

import numpy as np


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

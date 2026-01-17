"""Base visualization utilities and constants.

Provides common utilities for all visualization modules:
- Color constants for labels
- Image loading functions
- Figure saving utilities
"""

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.figure import Figure
from PIL import Image

from spine_vision.datasets.labels import LABEL_COLORS, LABEL_DISPLAY_NAMES

# Re-export for backward compatibility
__all__ = ["LABEL_COLORS", "LABEL_DISPLAY_NAMES"]

# Standard colors for confusion categories
CONFUSION_COLORS = {
    "tp": "#2ecc71",  # Green
    "tn": "#27ae60",  # Dark green
    "fp": "#e74c3c",  # Red
    "fn": "#c0392b",  # Dark red
    "correct": "#2ecc71",
    "incorrect": "#e74c3c",
}

# Split colors for distribution plots
SPLIT_COLORS = {
    "train": "#3498db",  # Blue
    "val": "#e74c3c",  # Red
    "test": "#2ecc71",  # Green
}


def safe_to_int(value: Any) -> int:
    """Safely convert a value to int, handling numpy arrays and scalars."""
    if isinstance(value, np.ndarray):
        return int(value.item()) if value.ndim == 0 else int(value.flat[0])
    return int(value)


def extract_prediction_value(pred: Any, gt: Any) -> tuple[int, int]:
    """Extract integer prediction and ground truth values from various formats.

    Handles:
    - Binary predictions (single element arrays/lists, threshold at 0.5)
    - Multiclass predictions (arrays with probabilities, use argmax)
    - Scalar values

    Args:
        pred: Prediction value (scalar, array, or list).
        gt: Ground truth value (scalar, array, or list).

    Returns:
        Tuple of (pred_value, gt_value) as integers.
    """
    if isinstance(pred, (np.ndarray, list)):
        pred_arr = np.atleast_1d(pred)
        if len(pred_arr) == 1:
            pred_val = int(float(pred_arr[0]) > 0.5)
            gt_arr = np.atleast_1d(gt)
            gt_val = int(float(gt_arr[0])) if len(gt_arr) == 1 else safe_to_int(gt)
            return pred_val, gt_val
        pred_val = int(np.argmax(pred_arr))
        if np.isscalar(gt):
            gt_val = safe_to_int(gt)
        else:
            gt_arr = np.atleast_1d(gt)
            gt_val = int(np.argmax(gt_arr)) if len(gt_arr) > 1 else safe_to_int(gt_arr[0])
        return pred_val, gt_val
    return safe_to_int(pred), safe_to_int(gt)


def save_figure(
    fig: Figure,
    output_path: Path | None,
    filename: str,
    output_mode: Literal["browser", "html", "image"] = "image",
    dpi: int = 150,
) -> None:
    """Save figure according to output mode.

    Args:
        fig: Matplotlib figure to save.
        output_path: Directory for saving. If None, only shows in browser mode.
        filename: Output filename (without extension).
        output_mode: Output format - 'browser' shows interactively,
                     'html' and 'image' both save as PNG.
        dpi: Resolution for saved images.
    """
    if output_mode == "browser":
        plt.show()
    elif output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        path = output_path / f"{filename}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        logger.debug(f"Saved visualization: {path}")
    plt.close(fig)


def load_original_images(
    image_paths: list[Path | str],
    output_size: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    """Load original images from file paths without any transformation.

    Args:
        image_paths: List of paths to image files.
        output_size: Optional (H, W) to resize images for display consistency.

    Returns:
        List of numpy arrays [H, W, C] in RGB format, uint8.
    """
    images = []
    for path in image_paths:
        img = Image.open(path)
        if img.mode == "L":
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        if output_size is not None:
            img = img.resize((output_size[1], output_size[0]), Image.Resampling.BILINEAR)
        images.append(np.array(img))
    return images


def load_classification_original_images(
    data_path: Path,
    metadata_list: list[dict[str, Any]],
    output_size: tuple[int, int] | None = None,
    grayscale_display: bool = True,
) -> list[np.ndarray]:
    """Load original classification images from metadata.

    Args:
        data_path: Base path to classification dataset.
        metadata_list: List of metadata dicts with 'source', 'patient_id', 'ivd' keys.
        output_size: Optional (H, W) to resize images.
        grayscale_display: If True, displays T2 as grayscale.

    Returns:
        List of numpy arrays [H, W, 3] in RGB format, uint8.
    """
    images = []
    images_dir = data_path / "images"

    for meta in metadata_list:
        source = meta.get("source", "")
        patient_id = meta.get("patient_id", "")
        ivd = meta.get("ivd", "")

        t1_filename = f"{source}_{patient_id}_sag_t1_L{ivd}.png"
        t2_filename = f"{source}_{patient_id}_sag_t2_L{ivd}.png"
        t1_path = images_dir / t1_filename
        t2_path = images_dir / t2_filename

        if t2_path.exists():
            t2_img = np.array(Image.open(t2_path).convert("L"))
            if grayscale_display:
                rgb_image = np.stack([t2_img, t2_img, t2_img], axis=-1)
            else:
                t1_img = np.array(Image.open(t1_path).convert("L")) if t1_path.exists() else t2_img
                rgb_image = np.stack([t2_img, t1_img, t2_img], axis=-1)

            if output_size is not None:
                pil_img = Image.fromarray(rgb_image)
                pil_img = pil_img.resize((output_size[1], output_size[0]), Image.Resampling.BILINEAR)
                rgb_image = np.array(pil_img)
            images.append(rgb_image)
        else:
            h, w = output_size if output_size else (128, 128)
            images.append(np.zeros((h, w, 3), dtype=np.uint8))
            logger.warning(f"Image not found: {t2_path}")

    return images


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Ensure image is RGB format."""
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    return img


def create_grid_axes(
    n_items: int,
    max_cols: int = 4,
    figsize_per_item: tuple[float, float] = (3.0, 3.0),
) -> tuple[Figure, np.ndarray]:
    """Create a grid of axes for displaying multiple items.

    Args:
        n_items: Number of items to display.
        max_cols: Maximum columns in grid.
        figsize_per_item: (width, height) per subplot.

    Returns:
        Tuple of (figure, axes array flattened).
    """
    n_cols = min(max_cols, n_items)
    n_rows = (n_items + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_item[0] * n_cols, figsize_per_item[1] * n_rows),
        squeeze=False,
    )
    return fig, axes.flatten()

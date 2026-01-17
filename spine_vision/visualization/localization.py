"""Localization visualization using seaborn/matplotlib."""

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from PIL import Image

from spine_vision.visualization.base import (
    create_grid_axes,
    ensure_rgb,
    save_figure,
)


def plot_localization_predictions(
    images: list[np.ndarray],
    predictions: np.ndarray,
    targets: np.ndarray,
    metadata: list[dict[str, Any]] | None = None,
    num_samples: int = 16,
    output_path: Path | None = None,
    filename: str = "localization_predictions",
    output_mode: Literal["browser", "html", "image"] = "image",
) -> Figure:
    """Plot localization predictions overlaid on images.

    Args:
        images: List of images as numpy arrays [H, W, C] or [H, W].
        predictions: Predicted coordinates [N, 2] in relative [0, 1].
        targets: Ground truth coordinates [N, 2] in relative [0, 1].
        metadata: Optional list of metadata dicts per sample.
        num_samples: Maximum number of samples to show.
        output_path: Directory for saving.
        filename: Output filename.
        output_mode: Output format.

    Returns:
        Matplotlib Figure.
    """
    n_samples = min(len(images), num_samples)
    fig, axes = create_grid_axes(n_samples, max_cols=4, figsize_per_item=(3.0, 3.0))

    for i in range(n_samples):
        ax = axes[i]
        img = ensure_rgb(images[i])
        h, w = img.shape[:2]

        ax.imshow(img)

        # Convert coordinates to pixel space
        pred_x, pred_y = predictions[i] * [w, h]
        gt_x, gt_y = targets[i] * [w, h]

        # Ground truth (green X)
        ax.scatter([gt_x], [gt_y], c="green", marker="x", s=100, linewidths=2, label="GT" if i == 0 else None)
        # Prediction (red circle)
        ax.scatter([pred_x], [pred_y], c="red", marker="o", s=80, label="Pred" if i == 0 else None)
        # Connecting line
        ax.plot([gt_x, pred_x], [gt_y, pred_y], "y--", linewidth=1, alpha=0.7)

        title = ""
        if metadata and i < len(metadata):
            title = metadata[i].get("level", "")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Hide unused axes
    for i in range(n_samples, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Localization Predictions (Green=GT, Red=Pred)", fontsize=12, fontweight="bold")
    fig.legend(loc="upper right")
    plt.tight_layout()

    save_figure(fig, output_path, filename, output_mode)
    return fig


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    levels: np.ndarray | None = None,
    level_names: list[str] | None = None,
    output_path: Path | None = None,
    filename: str = "error_distribution",
    output_mode: Literal["browser", "html", "image"] = "image",
) -> Figure:
    """Plot error distribution analysis.

    Args:
        predictions: Predicted coordinates [N, 2].
        targets: Ground truth coordinates [N, 2].
        levels: Optional level indices [N].
        level_names: Names for each level.
        output_path: Directory for saving.
        filename: Output filename.
        output_mode: Output format.

    Returns:
        Matplotlib Figure.
    """
    errors = predictions - targets
    distances = np.sqrt(np.sum(errors**2, axis=1))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    sns.set_style("whitegrid")

    # Distance histogram
    ax = axes[0, 0]
    sns.histplot(distances, bins=50, ax=ax, color="blue")
    ax.set_xlabel("Euclidean Distance")
    ax.set_ylabel("Count")
    ax.set_title("Distance Distribution")

    # X vs Y error scatter
    ax = axes[0, 1]
    scatter = ax.scatter(errors[:, 0], errors[:, 1], c=distances, cmap="viridis", s=10, alpha=0.5)
    plt.colorbar(scatter, ax=ax, label="Distance")
    ax.set_xlabel("X Error")
    ax.set_ylabel("Y Error")
    ax.set_title("X Error vs Y Error")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    # Error by level (box plot)
    ax = axes[1, 0]
    if levels is not None and level_names:
        data = []
        labels = []
        for level_idx, level_name in enumerate(level_names):
            mask = levels == level_idx
            if np.sum(mask) > 0:
                data.append(distances[mask])
                labels.append(level_name)
        if data:
            ax.boxplot(data, labels=labels)
    else:
        ax.boxplot([distances], labels=["All"])
    ax.set_ylabel("Distance")
    ax.set_title("Error by Level")

    # Cumulative error
    ax = axes[1, 1]
    sorted_distances = np.sort(distances)
    cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances) * 100
    ax.plot(sorted_distances, cumulative, color="red", linewidth=2)
    for thresh in [0.02, 0.05, 0.10]:
        pct_below = float(np.mean(sorted_distances < thresh) * 100)
        ax.axhline(y=pct_below, color="gray", linestyle="--", alpha=0.5)
        ax.annotate(f"{pct_below:.1f}% @ {thresh}", xy=(thresh, pct_below), fontsize=8)
    ax.set_xlabel("Distance Threshold")
    ax.set_ylabel("% Below Threshold")
    ax.set_title("Cumulative Error")

    fig.suptitle("Error Distribution Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_figure(fig, output_path, filename, output_mode)
    return fig


def plot_per_level_metrics(
    metrics: dict[str, float],
    level_names: list[str],
    metric_prefix: str = "med_",
    output_path: Path | None = None,
    filename: str = "per_level_metrics",
    output_mode: Literal["browser", "html", "image"] = "image",
) -> Figure:
    """Plot per-level metric comparison.

    Args:
        metrics: Dictionary of metrics including per-level values.
        level_names: Names of levels.
        metric_prefix: Prefix for per-level metrics in dict.
        output_path: Directory for saving.
        filename: Output filename.
        output_mode: Output format.

    Returns:
        Matplotlib Figure.
    """
    values = []
    labels = []
    for level in level_names:
        key = f"{metric_prefix}{level}"
        if key in metrics:
            values.append(metrics[key])
            labels.append(level)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_style("whitegrid")

    bars = ax.bar(labels, values, color="steelblue")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}",
                ha="center", va="bottom", fontsize=9)

    if values:
        avg = float(np.mean(values))
        ax.axhline(y=avg, color="red", linestyle="--", label=f"Avg: {avg:.4f}")
        ax.legend()

    ax.set_xlabel("Level")
    ax.set_ylabel("Value")
    ax.set_title(f"Per-Level {metric_prefix.upper().rstrip('_')}")
    plt.tight_layout()

    save_figure(fig, output_path, filename, output_mode)
    return fig


def visualize_sample(
    image: np.ndarray | Image.Image,
    prediction: np.ndarray,
    target: np.ndarray,
    level: str = "",
    output_path: Path | None = None,
    filename: str = "sample",
    output_mode: Literal["browser", "html", "image"] = "image",
) -> Figure:
    """Visualize a single sample with prediction overlay.

    Args:
        image: Image as numpy array or PIL Image.
        prediction: Predicted coordinates [2] in relative [0, 1].
        target: Ground truth coordinates [2] in relative [0, 1].
        level: Level label for title.
        output_path: Directory for saving.
        filename: Output filename.
        output_mode: Output format.

    Returns:
        Matplotlib Figure.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    image = ensure_rgb(image)
    h, w = image.shape[:2]

    pred_x, pred_y = prediction * [w, h]
    gt_x, gt_y = target * [w, h]
    error = np.sqrt(np.sum((prediction - target) ** 2))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)

    # Ground truth
    ax.scatter([gt_x], [gt_y], c="green", marker="x", s=150, linewidths=3, label="GT")
    ax.annotate("GT", (gt_x, gt_y), textcoords="offset points", xytext=(0, -15),
                ha="center", fontsize=10, color="green")

    # Prediction
    ax.scatter([pred_x], [pred_y], c="red", marker="o", s=120, label="Pred")
    ax.annotate("Pred", (pred_x, pred_y), textcoords="offset points", xytext=(0, 15),
                ha="center", fontsize=10, color="red")

    # Connecting line
    ax.plot([gt_x, pred_x], [gt_y, pred_y], "y--", linewidth=2)

    ax.set_title(f"{level} (Error: {error:.4f})", fontsize=12, fontweight="bold")
    ax.axis("off")
    ax.legend(loc="upper right")

    plt.tight_layout()
    save_figure(fig, output_path, filename, output_mode)
    return fig

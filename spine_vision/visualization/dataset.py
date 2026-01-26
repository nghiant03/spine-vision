"""Dataset visualization standalone functions.

Provides seaborn/matplotlib-based visualizations for dataset statistics:
- Dataset statistics (samples by level, source, grades)
- Label distributions
- Label co-occurrence analysis
- Sample images per class
"""

from collections import defaultdict
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from PIL import Image

from spine_vision.core.tasks import (
    AVAILABLE_TASK_NAMES,
    get_task,
    get_task_color,
    get_task_display_name,
)
from spine_vision.datasets.levels import IDX_TO_LEVEL
from spine_vision.training.datasets.classification import ClassificationDataset
from spine_vision.visualization.base import save_figure


def _load_sample_image(
    data_path: Path,
    source: str,
    patient_id: str,
    ivd: int,
    display_size: tuple[int, int],
) -> np.ndarray:
    """Load a sample image from the dataset."""
    images_dir = data_path / "images"

    # Try T2 first (more commonly available)
    t2_path = images_dir / f"{source}_{patient_id}_sag_t2_L{ivd}.png"
    t1_path = images_dir / f"{source}_{patient_id}_sag_t1_L{ivd}.png"

    if t2_path.exists():
        img = Image.open(t2_path).convert("L")
    elif t1_path.exists():
        img = Image.open(t1_path).convert("L")
    else:
        # Return placeholder
        return np.zeros((display_size[0], display_size[1], 3), dtype=np.uint8)

    # Resize and convert to RGB
    img = img.resize((display_size[1], display_size[0]), Image.Resampling.BILINEAR)
    img_arr = np.array(img)
    return np.stack([img_arr, img_arr, img_arr], axis=-1)


def plot_dataset_statistics(
    dataset: ClassificationDataset,
    output_path: Path | None = None,
    output_mode: Literal["browser", "html", "image"] = "image",
) -> Figure:
    """Plot overall dataset statistics.

    Args:
        dataset: Classification dataset to analyze.
        output_path: Directory for saving. If None, only shows in browser mode.
        output_mode: Output format - 'browser', 'html', or 'image'.

    Returns:
        Matplotlib figure with dataset statistics.
    """
    stats = dataset.get_stats()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Classification Dataset Statistics (N={stats['num_samples']}, Patients={stats['num_patients']})",
        fontsize=14,
        fontweight="bold",
    )

    # IVD Level distribution
    ax1 = axes[0, 0]
    levels = stats["levels"]
    level_order = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
    level_counts = [levels.get(lvl, 0) for lvl in level_order]
    bars1 = ax1.bar(level_order, level_counts, color="#3498db", edgecolor="white")
    ax1.set_title("Samples by IVD Level")
    ax1.set_xlabel("IVD Level")
    ax1.set_ylabel("Count")
    for bar, count in zip(bars1, level_counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(count),
                 ha="center", va="bottom", fontsize=9)

    # Source distribution (pie chart)
    ax2 = axes[0, 1]
    sources = stats["sources"]
    ax2.pie(
        list(sources.values()),
        labels=list(sources.keys()),
        colors=["#2ecc71", "#e74c3c"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax2.set_title("Samples by Source")

    # Pfirrmann distribution
    ax3 = axes[1, 0]
    pfirrmann = stats["pfirrmann"]
    pfirrmann_labels = [f"Grade {i}" for i in range(1, 6)]
    pfirrmann_counts = [pfirrmann.get(i, 0) for i in range(1, 6)]
    bars3 = ax3.bar(pfirrmann_labels, pfirrmann_counts, color="#9b59b6", edgecolor="white")
    ax3.set_title("Pfirrmann Grade Distribution")
    ax3.set_xlabel("Grade")
    ax3.set_ylabel("Count")
    for bar, count in zip(bars3, pfirrmann_counts):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(count),
                 ha="center", va="bottom", fontsize=9)

    # Modic distribution
    ax4 = axes[1, 1]
    modic = stats["modic"]
    modic_labels = [f"Type {i}" for i in range(4)]
    modic_counts = [modic.get(i, 0) for i in range(4)]
    bars4 = ax4.bar(modic_labels, modic_counts, color="#e67e22", edgecolor="white")
    ax4.set_title("Modic Type Distribution")
    ax4.set_xlabel("Type")
    ax4.set_ylabel("Count")
    for bar, count in zip(bars4, modic_counts):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(count),
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_figure(fig, output_path, "dataset_statistics", output_mode)
    return fig


def plot_binary_label_distributions(
    dataset: ClassificationDataset,
    output_path: Path | None = None,
    output_mode: Literal["browser", "html", "image"] = "image",
) -> Figure:
    """Plot distributions for all binary labels.

    Args:
        dataset: Classification dataset to analyze.
        output_path: Directory for saving. If None, only shows in browser mode.
        output_mode: Output format - 'browser', 'html', or 'image'.

    Returns:
        Matplotlib figure with binary label distributions.
    """
    binary_labels = [
        label for label in AVAILABLE_TASK_NAMES if get_task(label).task_type == "binary"
    ]

    # Count positives and negatives for each label
    label_counts: dict[str, dict[int, int]] = {label: {0: 0, 1: 0} for label in binary_labels}

    # Map from label name to record key
    record_keys = {
        "herniation": "herniation",
        "bulging": "bulging",
        "upper_endplate": "upper_endplate",
        "lower_endplate": "lower_endplate",
        "spondy": "spondylolisthesis",
        "narrowing": "narrowing",
    }

    for record in dataset.records:
        for label in binary_labels:
            record_key = record_keys.get(label, label)
            value = int(record[record_key])
            label_counts[label][value] += 1

    # Create grouped bar chart
    label_names = [get_task_display_name(lbl) for lbl in binary_labels]
    neg_counts = [label_counts[lbl][0] for lbl in binary_labels]
    pos_counts = [label_counts[lbl][1] for lbl in binary_labels]

    x = np.arange(len(label_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, neg_counts, width, label="Negative (0)", color="#2ecc71", edgecolor="white")
    bars2 = ax.bar(x + width / 2, pos_counts, width, label="Positive (1)", color="#e74c3c", edgecolor="white")

    ax.set_title("Binary Label Distributions", fontsize=14, fontweight="bold")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.legend(loc="upper right")

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 5, str(int(height)),
                ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 5, str(int(height)),
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    save_figure(fig, output_path, "binary_label_distributions", output_mode)
    return fig


def plot_label_cooccurrence(
    dataset: ClassificationDataset,
    output_path: Path | None = None,
    output_mode: Literal["browser", "html", "image"] = "image",
) -> Figure:
    """Plot co-occurrence heatmap between binary labels.

    Args:
        dataset: Classification dataset to analyze.
        output_path: Directory for saving. If None, only shows in browser mode.
        output_mode: Output format - 'browser', 'html', or 'image'.

    Returns:
        Matplotlib figure with co-occurrence heatmap.
    """
    binary_labels = [
        label for label in AVAILABLE_TASK_NAMES if get_task(label).task_type == "binary"
    ]

    record_keys = {
        "herniation": "herniation",
        "bulging": "bulging",
        "upper_endplate": "upper_endplate",
        "lower_endplate": "lower_endplate",
        "spondy": "spondylolisthesis",
        "narrowing": "narrowing",
    }

    n_labels = len(binary_labels)
    cooccurrence = np.zeros((n_labels, n_labels), dtype=int)

    for record in dataset.records:
        for i, label_i in enumerate(binary_labels):
            key_i = record_keys.get(label_i, label_i)
            val_i = int(record[key_i])
            if val_i == 1:
                for j, label_j in enumerate(binary_labels):
                    key_j = record_keys.get(label_j, label_j)
                    val_j = int(record[key_j])
                    if val_j == 1:
                        cooccurrence[i, j] += 1

    label_names = [get_task_display_name(lbl) for lbl in binary_labels]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cooccurrence,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Binary Label Co-occurrence (Both Positive)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Label")
    ax.set_ylabel("Label")

    plt.tight_layout()
    save_figure(fig, output_path, "label_cooccurrence", output_mode)
    return fig


def plot_pfirrmann_by_level(
    dataset: ClassificationDataset,
    output_path: Path | None = None,
    output_mode: Literal["browser", "html", "image"] = "image",
) -> Figure:
    """Plot Pfirrmann grade distribution by IVD level.

    Args:
        dataset: Classification dataset to analyze.
        output_path: Directory for saving. If None, only shows in browser mode.
        output_mode: Output format - 'browser', 'html', or 'image'.

    Returns:
        Matplotlib figure with Pfirrmann distribution by level.
    """
    # Count Pfirrmann grades per level
    level_pfirrmann: dict[str, dict[int, int]] = {}
    levels = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

    for lvl in levels:
        level_pfirrmann[lvl] = {i: 0 for i in range(1, 6)}

    for record in dataset.records:
        level = IDX_TO_LEVEL.get(record["level_idx"], "")
        pfirrmann = record["pfirrmann"]
        if level in level_pfirrmann:
            level_pfirrmann[level][pfirrmann] += 1

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#27ae60", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    x = np.arange(len(levels))
    width = 0.15

    for grade_idx, grade in enumerate(range(1, 6)):
        counts = [level_pfirrmann[lvl][grade] for lvl in levels]
        offset = (grade_idx - 2) * width
        bars = ax.bar(x + offset, counts, width, label=f"Grade {grade}", color=colors[grade_idx], edgecolor="white")
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, str(count),
                        ha="center", va="bottom", fontsize=7)

    ax.set_title("Pfirrmann Grade Distribution by IVD Level", fontsize=14, fontweight="bold")
    ax.set_xlabel("IVD Level")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.legend(loc="upper left", ncol=5, fontsize=9)

    plt.tight_layout()
    save_figure(fig, output_path, "pfirrmann_by_level", output_mode)
    return fig


def plot_samples_per_class(
    dataset: ClassificationDataset,
    data_path: Path,
    output_path: Path | None = None,
    output_mode: Literal["browser", "html", "image"] = "image",
    samples_per_class: int = 4,
    display_size: tuple[int, int] = (128, 128),
) -> dict[str, Figure]:
    """Plot sample images for each possible value of each label.

    Args:
        dataset: Classification dataset to analyze.
        data_path: Path to dataset directory containing images.
        output_path: Directory for saving. If None, only shows in browser mode.
        output_mode: Output format - 'browser', 'html', or 'image'.
        samples_per_class: Number of sample images to show per class value.
        display_size: Size (H, W) for displayed images.

    Returns:
        Dictionary mapping label names to their matplotlib figures.
    """
    figures: dict[str, Figure] = {}

    # Map from label name to record key
    record_keys = {
        "pfirrmann": "pfirrmann",
        "modic": "modic",
        "herniation": "herniation",
        "bulging": "bulging",
        "upper_endplate": "upper_endplate",
        "lower_endplate": "lower_endplate",
        "spondy": "spondylolisthesis",
        "narrowing": "narrowing",
    }

    for label in AVAILABLE_TASK_NAMES:
        task = get_task(label)
        record_key = record_keys.get(label, label)

        if task.task_type == "multiclass":
            num_classes = task.num_classes
            # For Pfirrmann: values are 1-5, for Modic: 0-3
            if label == "pfirrmann":
                class_values = list(range(1, num_classes + 1))
                class_names = [f"Grade {i}" for i in class_values]
            else:
                class_values = list(range(num_classes))
                class_names = [f"Type {i}" for i in class_values]
        else:
            class_values = [0, 1]
            class_names = ["Negative (0)", "Positive (1)"]

        n_rows = len(class_values)
        n_cols = samples_per_class

        # Group samples by class value
        samples_by_class: dict[int, list[dict]] = defaultdict(list)
        for record in dataset.records:
            value = int(record[record_key])
            samples_by_class[value].append(record)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.5 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        display_name = get_task_display_name(label)
        color = get_task_color(label)

        fig.suptitle(f"{display_name} - Samples per Class Value", fontsize=14, fontweight="bold", color=color)

        for row_idx, (class_val, class_name) in enumerate(zip(class_values, class_names)):
            samples = samples_by_class.get(class_val, [])
            count = len(samples)

            # Randomly select samples
            if len(samples) > 0:
                np.random.seed(42)  # For reproducibility
                indices = np.random.choice(
                    len(samples), size=min(n_cols, len(samples)), replace=False
                )
                selected = [samples[i] for i in indices]
            else:
                selected = []

            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]

                if col_idx < len(selected):
                    record = selected[col_idx]
                    img = _load_sample_image(
                        data_path,
                        record["source"],
                        record["patient_id"],
                        record["ivd_level"],
                        display_size,
                    )

                    ax.imshow(img)
                    # Add level info
                    level = IDX_TO_LEVEL.get(record["level_idx"], "")
                    ax.text(
                        display_size[1] // 2,
                        display_size[0] - 10,
                        level,
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color="white",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
                    )
                else:
                    # Empty placeholder
                    placeholder = np.ones((display_size[0], display_size[1], 3), dtype=np.uint8) * 200
                    ax.imshow(placeholder)

                ax.axis("off")

                # Add row label (class name) on the first column
                if col_idx == 0:
                    ax.set_ylabel(f"{class_name}\n(n={count})", fontsize=10, rotation=0, labelpad=60, va="center")
                    ax.yaxis.set_label_position("left")

        plt.tight_layout()
        save_figure(fig, output_path, f"samples_{label}", output_mode)
        figures[label] = fig

    return figures

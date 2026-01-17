"""Classification visualization using seaborn/matplotlib."""

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from spine_vision.datasets.labels import LABEL_COLORS, LABEL_DISPLAY_NAMES
from spine_vision.visualization.base import (
    CONFUSION_COLORS,
    SPLIT_COLORS,
    create_grid_axes,
    ensure_rgb,
    extract_prediction_value,
    save_figure,
)


def plot_classification_predictions(
    images: list[np.ndarray],
    predictions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    metadata: list[dict[str, Any]] | None = None,
    num_samples: int = 16,
    output_path: Path | None = None,
    filename: str = "classification_predictions",
    output_mode: Literal["browser", "html", "image"] = "image",
) -> Figure:
    """Plot classification predictions overlaid on images.

    Args:
        images: List of images as numpy arrays [H, W, C] or [H, W].
        predictions: Dict mapping label names to predicted values [N] or [N, C].
        targets: Dict mapping label names to ground truth values [N] or [N, C].
        metadata: Optional list of metadata dicts per sample.
        num_samples: Maximum number of samples to show.
        output_path: Directory for saving.
        filename: Output filename.
        output_mode: Output format.

    Returns:
        Matplotlib Figure.
    """
    n_samples = min(len(images), num_samples)
    fig, axes = create_grid_axes(n_samples, max_cols=4, figsize_per_item=(3.5, 3.5))
    labels = list(predictions.keys())

    for i in range(n_samples):
        ax = axes[i]
        img = ensure_rgb(images[i])
        h, w = img.shape[:2]

        ax.imshow(img)

        annotations = []
        all_correct = True

        for label in labels:
            pred_val, gt_val = extract_prediction_value(predictions[label][i], targets[label][i])
            is_correct = pred_val == gt_val
            if not is_correct:
                all_correct = False
            display_name = LABEL_DISPLAY_NAMES.get(label, label)
            status = "\u2713" if is_correct else "\u2717"
            annotations.append(f"{display_name}: {pred_val} ({gt_val}) {status}")

        border_color = CONFUSION_COLORS["correct"] if all_correct else CONFUSION_COLORS["incorrect"]

        # Add colored border
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

        title_parts = []
        if metadata and i < len(metadata):
            level = metadata[i].get("level", "")
            if level:
                title_parts.append(level)

        subtitle = " | ".join(annotations[:3])
        if len(annotations) > 3:
            subtitle += f" +{len(annotations) - 3}"

        title = " ".join(title_parts) if title_parts else f"Sample {i + 1}"
        ax.set_title(f"{title}\n{subtitle}", fontsize=8)
        ax.axis("off")

    for i in range(n_samples, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Classification Predictions (Green=Correct, Red=Incorrect)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    save_figure(fig, output_path, filename, output_mode)
    return fig


def plot_classification_metrics(
    metrics: dict[str, float],
    target_labels: list[str] | None = None,
    output_path: Path | None = None,
    filename: str = "classification_metrics",
    output_mode: Literal["browser", "html", "image"] = "image",
) -> Figure:
    """Plot per-label classification metrics as bar charts.

    Args:
        metrics: Dictionary of metrics (e.g., pfirrmann_accuracy, modic_f1).
        target_labels: List of labels to include.
        output_path: Directory for saving.
        filename: Output filename.
        output_mode: Output format.

    Returns:
        Matplotlib Figure.
    """
    metric_types = ["accuracy", "f1", "precision", "recall"]
    labels = target_labels or list(LABEL_DISPLAY_NAMES.keys())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    sns.set_style("whitegrid")

    for idx, metric_type in enumerate(metric_types):
        ax = axes[idx]
        values = []
        label_names = []
        colors = []

        for label in labels:
            key = f"{label}_{metric_type}"
            if key in metrics:
                values.append(metrics[key])
                label_names.append(LABEL_DISPLAY_NAMES.get(label, label))
                colors.append(LABEL_COLORS.get(label, "#333333"))

        if values:
            bars = ax.bar(label_names, values, color=colors)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Value")
            ax.set_title(metric_type.title())
            ax.tick_params(axis="x", rotation=45)

    # Build title with overall metrics
    overall_metrics = {k: v for k, v in metrics.items() if k.startswith("overall_")}
    if overall_metrics:
        title = "Per-Label Classification Metrics | " + " | ".join(
            f"{k.replace('overall_', '').title()}: {v:.3f}" for k, v in overall_metrics.items()
        )
    else:
        title = "Per-Label Classification Metrics"

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    save_figure(fig, output_path, filename, output_mode)
    return fig


def plot_confusion_matrices(
    confusion_matrices: dict[str, np.ndarray],
    class_names: dict[str, list[str]] | None = None,
    output_path: Path | None = None,
    filename: str = "confusion_matrices",
    output_mode: Literal["browser", "html", "image"] = "image",
) -> Figure:
    """Plot confusion matrices for classification labels.

    Args:
        confusion_matrices: Dict mapping label names to confusion matrices [C, C].
        class_names: Dict mapping label names to class name lists.
        output_path: Directory for saving.
        filename: Output filename.
        output_mode: Output format.

    Returns:
        Matplotlib Figure.
    """
    labels = list(confusion_matrices.keys())
    n_labels = len(labels)

    if n_labels == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No confusion matrices to plot", ha="center", va="center")
        ax.axis("off")
        return fig

    n_cols = min(3, n_labels)
    n_rows = (n_labels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    for idx, label in enumerate(labels):
        ax = axes[idx]
        cm = confusion_matrices[label]
        n_classes = cm.shape[0]

        # Normalize
        cm_normalized = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

        names = class_names.get(label, [str(i) for i in range(n_classes)]) if class_names else [str(i) for i in range(n_classes)]

        sns.heatmap(cm_normalized, annot=cm, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=names, yticklabels=names, cbar=idx == 0)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(LABEL_DISPLAY_NAMES.get(label, label))

    for idx in range(n_labels, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_figure(fig, output_path, filename, output_mode)
    return fig


def plot_confusion_matrix_with_samples(
    images: list[np.ndarray],
    predictions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    target_label: str,
    metadata: list[dict[str, Any]] | None = None,
    class_names: list[str] | None = None,
    max_samples_per_cell: int = 4,
    output_path: Path | None = None,
    filename: str | None = None,
    output_mode: Literal["browser", "html", "image"] = "image",
) -> Figure:
    """Plot confusion matrix with sample images from each cell.

    Args:
        images: List of images as numpy arrays.
        predictions: Dict mapping label names to predicted values.
        targets: Dict mapping label names to ground truth values.
        target_label: Which label to analyze.
        metadata: Optional list of metadata dicts per sample.
        class_names: Optional list of class names for display.
        max_samples_per_cell: Maximum samples per cell.
        output_path: Directory for saving.
        filename: Output filename.
        output_mode: Output format.

    Returns:
        Matplotlib Figure.
    """
    if target_label not in predictions:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"Label '{target_label}' not found", ha="center", va="center")
        ax.axis("off")
        return fig

    pred_arr = np.atleast_1d(predictions[target_label])
    gt_arr = np.atleast_1d(targets[target_label])

    n_samples = len(images)
    pred_classes = np.zeros(n_samples, dtype=int)
    gt_classes = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        pred_classes[i], gt_classes[i] = extract_prediction_value(
            pred_arr[i] if pred_arr.ndim > 1 else pred_arr[i],
            gt_arr[i] if gt_arr.ndim > 1 else gt_arr[i],
        )

    unique_classes = sorted(set(pred_classes) | set(gt_classes))
    n_classes = len(unique_classes)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}

    # Build confusion matrix and cell samples
    cm = np.zeros((n_classes, n_classes), dtype=int)
    cell_samples: dict[tuple[int, int], list[int]] = {}

    for i in range(n_samples):
        gt_idx = class_to_idx[gt_classes[i]]
        pred_idx = class_to_idx[pred_classes[i]]
        cm[gt_idx, pred_idx] += 1
        key = (gt_idx, pred_idx)
        if key not in cell_samples:
            cell_samples[key] = []
        cell_samples[key].append(i)

    display_names = class_names if class_names else [str(c) for c in unique_classes]
    non_empty_cells = [(gt_idx, pred_idx) for (gt_idx, pred_idx), samples in cell_samples.items() if samples]

    if not non_empty_cells:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"No samples found for '{target_label}'", ha="center", va="center")
        ax.axis("off")
        return fig

    display_name = LABEL_DISPLAY_NAMES.get(target_label, target_label)
    n_cell_rows = len(non_empty_cells)
    n_rows = 1 + n_cell_rows

    # Create figure with gridspec for flexible layout
    fig = plt.figure(figsize=(max(8, max_samples_per_cell * 2), 3 + n_cell_rows * 2))
    gs = fig.add_gridspec(n_rows, 1, height_ratios=[2] + [1] * n_cell_rows, hspace=0.3)

    # Confusion matrix heatmap
    ax_cm = fig.add_subplot(gs[0])
    cm_normalized = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    sns.heatmap(cm_normalized, annot=cm, fmt="d", cmap="Blues", ax=ax_cm,
                xticklabels=display_names, yticklabels=display_names)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    ax_cm.set_title(f"{display_name} Confusion Matrix")

    # Sample images for each cell
    for cell_row_idx, (gt_idx, pred_idx) in enumerate(sorted(non_empty_cells)):
        sample_indices = cell_samples[(gt_idx, pred_idx)]
        np.random.shuffle(sample_indices)
        selected_indices = sample_indices[:max_samples_per_cell]

        is_correct = gt_idx == pred_idx
        border_color = CONFUSION_COLORS["correct"] if is_correct else CONFUSION_COLORS["incorrect"]

        inner_gs = gs[cell_row_idx + 1].subgridspec(1, max_samples_per_cell, wspace=0.05)

        gt_name = display_names[gt_idx]
        pred_name = display_names[pred_idx]
        status = "Correct" if is_correct else "Misclassified"

        for col_idx in range(max_samples_per_cell):
            ax = fig.add_subplot(inner_gs[col_idx])

            if col_idx < len(selected_indices):
                sample_idx = selected_indices[col_idx]
                img = ensure_rgb(images[sample_idx])
                ax.imshow(img)

                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(2)
            else:
                ax.set_facecolor("#f0f0f0")

            ax.axis("off")
            if col_idx == 0:
                n_cell = len(cell_samples[(gt_idx, pred_idx)])
                ax.set_title(f"GT={gt_name}\u2192Pred={pred_name} ({status}, n={n_cell})", fontsize=9, loc="left")

    fig.suptitle(f"Confusion Matrix with Samples - {display_name}", fontsize=12, fontweight="bold")

    output_filename = filename or f"confusion_matrix_samples_{target_label}"
    save_figure(fig, output_path, output_filename, output_mode)
    return fig


def plot_test_samples_with_labels(
    images: list[np.ndarray],
    predictions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    metadata: list[dict[str, Any]] | None = None,
    num_samples: int = 16,
    output_path: Path | None = None,
    filename: str = "test_samples",
    output_mode: Literal["browser", "html", "image"] = "image",
) -> Figure:
    """Plot test samples with predicted and ground truth labels.

    Args:
        images: List of images as numpy arrays.
        predictions: Dict mapping label names to predicted values.
        targets: Dict mapping label names to ground truth values.
        metadata: Optional list of metadata dicts per sample.
        num_samples: Maximum number of samples to show.
        output_path: Directory for saving.
        filename: Output filename.
        output_mode: Output format.

    Returns:
        Matplotlib Figure.
    """
    n_samples = min(len(images), num_samples)
    fig, axes = create_grid_axes(n_samples, max_cols=4, figsize_per_item=(3.5, 4.0))
    labels = list(predictions.keys())

    for i in range(n_samples):
        ax = axes[i]
        img = ensure_rgb(images[i])
        h, w = img.shape[:2]

        ax.imshow(img)

        pred_lines = []
        gt_lines = []
        n_correct = 0

        for label in labels:
            pred_val, gt_val = extract_prediction_value(predictions[label][i], targets[label][i])
            if pred_val == gt_val:
                n_correct += 1
            display_name = LABEL_DISPLAY_NAMES.get(label, label)[:3]
            pred_lines.append(f"{display_name}:{pred_val}")
            gt_lines.append(f"{display_name}:{gt_val}")

        accuracy = n_correct / len(labels) if labels else 0
        acc_color = "green" if accuracy >= 0.8 else ("orange" if accuracy >= 0.5 else "red")

        # Add text annotations
        pred_text = "Pred: " + " ".join(pred_lines[:4])
        gt_text = "GT: " + " ".join(gt_lines[:4])

        ax.text(5, 15, pred_text, fontsize=8, color="white",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
        ax.text(5, h - 10, gt_text, fontsize=8, color="white",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))

        # Build title
        title_parts = []
        if metadata and i < len(metadata):
            level = metadata[i].get("level", "")
            patient = metadata[i].get("patient_id", "")
            if level:
                title_parts.append(level)
            if patient:
                title_parts.append(f"({patient[:8]})")
        title_parts.append(f"Acc: {accuracy:.0%}")

        ax.set_title(" ".join(title_parts), fontsize=9, color=acc_color, fontweight="bold")

        for spine in ax.spines.values():
            spine.set_edgecolor(acc_color)
            spine.set_linewidth(3)
        ax.axis("off")

    for i in range(n_samples, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"Test Samples with Labels ({n_samples} samples)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    save_figure(fig, output_path, filename, output_mode)
    return fig


def plot_confusion_examples(
    images: list[np.ndarray],
    predictions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    metadata: list[dict[str, Any]] | None = None,
    target_label: str = "pfirrmann",
    num_samples_per_category: int = 4,
    output_path: Path | None = None,
    filename: str | None = None,
    output_mode: Literal["browser", "html", "image"] = "image",
) -> Figure:
    """Plot TP, TN, FP, FN examples for classification.

    Args:
        images: List of images as numpy arrays.
        predictions: Dict mapping label names to predicted values.
        targets: Dict mapping label names to ground truth values.
        metadata: Optional list of metadata dicts per sample.
        target_label: Which label to analyze.
        num_samples_per_category: Max samples per category.
        output_path: Directory for saving.
        filename: Output filename.
        output_mode: Output format.

    Returns:
        Matplotlib Figure.
    """
    if target_label not in predictions:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"Label '{target_label}' not found", ha="center", va="center")
        ax.axis("off")
        return fig

    pred_arr = np.atleast_1d(predictions[target_label])
    gt_arr = np.atleast_1d(targets[target_label])

    n_samples = len(images)
    pred_classes = np.zeros(n_samples, dtype=int)
    gt_classes = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        pred_classes[i], gt_classes[i] = extract_prediction_value(
            pred_arr[i] if pred_arr.ndim > 1 else pred_arr[i],
            gt_arr[i] if gt_arr.ndim > 1 else gt_arr[i],
        )

    unique_classes = np.unique(np.concatenate([pred_classes, gt_classes]))
    is_binary = len(unique_classes) <= 2

    categories: list[tuple[str, np.ndarray, str]] = []

    if is_binary:
        tp_mask = (pred_classes == 1) & (gt_classes == 1)
        tn_mask = (pred_classes == 0) & (gt_classes == 0)
        fp_mask = (pred_classes == 1) & (gt_classes == 0)
        fn_mask = (pred_classes == 0) & (gt_classes == 1)
        categories = [
            ("TP (Pred=1, GT=1)", tp_mask, CONFUSION_COLORS["tp"]),
            ("TN (Pred=0, GT=0)", tn_mask, CONFUSION_COLORS["tn"]),
            ("FP (Pred=1, GT=0)", fp_mask, CONFUSION_COLORS["fp"]),
            ("FN (Pred=0, GT=1)", fn_mask, CONFUSION_COLORS["fn"]),
        ]
    else:
        colors_correct = ["#2ecc71", "#27ae60", "#1abc9c", "#16a085", "#3498db"]
        colors_incorrect = ["#e74c3c", "#c0392b", "#e67e22", "#d35400", "#9b59b6"]
        for i, cls in enumerate(sorted(unique_classes)):
            correct_mask = (gt_classes == cls) & (pred_classes == cls)
            if correct_mask.sum() > 0:
                categories.append((f"GT={cls} Correct", correct_mask, colors_correct[i % len(colors_correct)]))
            incorrect_mask = (gt_classes == cls) & (pred_classes != cls)
            if incorrect_mask.sum() > 0:
                categories.append((f"GT={cls} Wrong", incorrect_mask, colors_incorrect[i % len(colors_incorrect)]))

    categories = [(name, mask, color) for name, mask, color in categories if mask.sum() > 0]
    n_rows = len(categories)

    if n_rows == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"No samples found for '{target_label}'", ha="center", va="center")
        ax.axis("off")
        return fig

    n_cols = num_samples_per_category
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)

    for row_idx, (cat_name, mask, border_color) in enumerate(categories):
        indices = np.where(mask)[0]
        np.random.shuffle(indices)
        selected = indices[:n_cols]

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]

            if col_idx < len(selected):
                sample_idx = selected[col_idx]
                img = ensure_rgb(images[sample_idx])
                ax.imshow(img)

                pred_val = pred_classes[sample_idx]
                gt_val = gt_classes[sample_idx]
                meta_text = ""
                if metadata and sample_idx < len(metadata):
                    level = metadata[sample_idx].get("level", "")
                    if level:
                        meta_text = f"{level} | "

                ax.text(0.5, 0.05, f"{meta_text}P:{pred_val} G:{gt_val}", transform=ax.transAxes,
                        fontsize=8, color="white", ha="center",
                        bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))

                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3)
            else:
                ax.set_facecolor("#f0f0f0")

            ax.axis("off")
            if col_idx == 0:
                count = mask.sum()
                ax.set_title(f"{cat_name} ({count} total)", fontsize=9, loc="left")

    display_name = LABEL_DISPLAY_NAMES.get(target_label, target_label)
    fig.suptitle(f"Confusion Examples for {display_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    output_filename = filename or f"confusion_examples_{target_label}"
    save_figure(fig, output_path, output_filename, output_mode)
    return fig


def plot_confusion_summary(
    predictions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    target_labels: list[str] | None = None,
    output_path: Path | None = None,
    filename: str = "confusion_summary",
    output_mode: Literal["browser", "html", "image"] = "image",
) -> Figure:
    """Plot summary of TP/TN/FP/FN counts across all labels.

    Args:
        predictions: Dict mapping label names to predicted values.
        targets: Dict mapping label names to ground truth values.
        target_labels: Labels to include.
        output_path: Directory for saving.
        filename: Output filename.
        output_mode: Output format.

    Returns:
        Matplotlib Figure.
    """
    labels = target_labels or list(predictions.keys())

    tp_counts = []
    tn_counts = []
    fp_counts = []
    fn_counts = []
    label_names = []

    for label in labels:
        if label not in predictions:
            continue

        pred_arr = np.atleast_1d(predictions[label])
        gt_arr = np.atleast_1d(targets[label])
        n_samples = len(pred_arr)
        pred_classes = np.zeros(n_samples, dtype=int)
        gt_classes = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            pred_classes[i], gt_classes[i] = extract_prediction_value(
                pred_arr[i] if pred_arr.ndim > 1 else pred_arr[i],
                gt_arr[i] if gt_arr.ndim > 1 else gt_arr[i],
            )

        unique_classes = np.unique(np.concatenate([pred_classes, gt_classes]))
        is_binary = len(unique_classes) <= 2

        if is_binary:
            tp = int(((pred_classes == 1) & (gt_classes == 1)).sum())
            tn = int(((pred_classes == 0) & (gt_classes == 0)).sum())
            fp = int(((pred_classes == 1) & (gt_classes == 0)).sum())
            fn = int(((pred_classes == 0) & (gt_classes == 1)).sum())
        else:
            correct = int((pred_classes == gt_classes).sum())
            incorrect = int((pred_classes != gt_classes).sum())
            tp, tn, fp, fn = correct, 0, incorrect, 0

        tp_counts.append(tp)
        tn_counts.append(tn)
        fp_counts.append(fp)
        fn_counts.append(fn)
        label_names.append(LABEL_DISPLAY_NAMES.get(label, label))

    x = np.arange(len(label_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(8, len(label_names) * 1.5), 6))
    sns.set_style("whitegrid")

    ax.bar(x - 1.5 * width, tp_counts, width, label="TP", color=CONFUSION_COLORS["tp"])
    ax.bar(x - 0.5 * width, tn_counts, width, label="TN", color=CONFUSION_COLORS["tn"])
    ax.bar(x + 0.5 * width, fp_counts, width, label="FP", color=CONFUSION_COLORS["fp"])
    ax.bar(x + 1.5 * width, fn_counts, width, label="FN", color=CONFUSION_COLORS["fn"])

    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.set_title("Confusion Summary by Label (TP/TN/FP/FN)")
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.legend(loc="upper right")

    plt.tight_layout()
    save_figure(fig, output_path, filename, output_mode)
    return fig


def plot_label_distribution(
    distributions: dict[str, dict[str, dict[int | str, int]]],
    target_labels: list[str] | None = None,
    output_path: Path | None = None,
    filename: str = "label_distribution",
    output_mode: Literal["browser", "html", "image"] = "image",
) -> Figure:
    """Plot label distribution across train/val/test splits.

    Args:
        distributions: Nested dict: {split_name: {label_name: {class: count}}}.
        target_labels: Labels to visualize.
        output_path: Directory for saving.
        filename: Output filename.
        output_mode: Output format.

    Returns:
        Matplotlib Figure.
    """
    splits = list(distributions.keys())
    if not splits:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No distributions provided", ha="center", va="center")
        ax.axis("off")
        return fig

    first_split = distributions[splits[0]]
    labels = target_labels or list(first_split.keys())

    if not labels:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No labels found", ha="center", va="center")
        ax.axis("off")
        return fig

    n_labels = len(labels)
    n_cols = min(3, n_labels)
    n_rows = (n_labels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()
    sns.set_style("whitegrid")

    for idx, label in enumerate(labels):
        ax = axes[idx]

        all_classes: set[int | str] = set()
        for split in splits:
            if label in distributions[split]:
                all_classes.update(distributions[split][label].keys())

        try:
            sorted_classes = sorted(all_classes, key=lambda x: (isinstance(x, str), x))
        except TypeError:
            sorted_classes = sorted(all_classes, key=str)

        class_names = [str(c) for c in sorted_classes]
        x = np.arange(len(class_names))
        width = 0.8 / len(splits)

        for s_idx, split in enumerate(splits):
            if label not in distributions[split]:
                continue
            label_dist = distributions[split][label]
            counts = [label_dist.get(c, 0) for c in sorted_classes]

            offset = (s_idx - len(splits) / 2 + 0.5) * width
            ax.bar(x + offset, counts, width, label=split.capitalize(),
                   color=SPLIT_COLORS.get(split, "#95a5a6"))

        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title(LABEL_DISPLAY_NAMES.get(label, label))
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        if idx == 0:
            ax.legend()

    for idx in range(n_labels, len(axes)):
        axes[idx].axis("off")

    # Build title with split totals
    split_totals = []
    for split in splits:
        total = sum(sum(label_dist.values()) for label_dist in distributions[split].values()) // len(labels)
        split_totals.append(f"{split.capitalize()}: {total}")

    fig.suptitle(f"Label Distribution by Split ({' | '.join(split_totals)})", fontsize=12, fontweight="bold")
    plt.tight_layout()

    save_figure(fig, output_path, filename, output_mode)
    return fig

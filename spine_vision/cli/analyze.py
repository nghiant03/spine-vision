"""Data analysis and visualization for classification dataset.

Generates comprehensive analysis including:
- Dataset statistics (samples, patients, class distributions)
- Sample images for each possible value of each label
- Class distribution histograms
- Co-occurrence heatmaps between labels
"""

import dataclasses
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from loguru import logger
from PIL import Image
from plotly.subplots import make_subplots

from spine_vision.core import setup_logger
from spine_vision.training.datasets.classification import (
    AVAILABLE_LABELS,
    ClassificationDataset,
    IDX_TO_LEVEL,
    LABEL_INFO,
)
from spine_vision.training.visualization import (
    LABEL_COLORS,
    LABEL_DISPLAY_NAMES,
)


@dataclasses.dataclass
class AnalyzeConfig:
    """Configuration for dataset analysis."""

    # Path to classification dataset
    data_path: Path = Path("data/processed/classification")

    # Output directory for visualizations
    output_path: Path = Path("analysis/classification")

    # Output format
    output_mode: Literal["browser", "html", "image"] = "html"

    # Number of samples to show per class value
    samples_per_class: int = 4

    # Image display size
    display_size: tuple[int, int] = (128, 128)

    # Enable verbose logging
    verbose: bool = False


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


def _save_figure(
    fig: go.Figure,
    output_path: Path,
    filename: str,
    output_mode: str,
) -> None:
    """Save figure based on output mode."""
    output_path.mkdir(parents=True, exist_ok=True)

    if output_mode == "browser":
        fig.show()
    elif output_mode == "html":
        path = output_path / f"{filename}.html"
        fig.write_html(path)
        logger.info(f"Saved: {path}")
    elif output_mode == "image":
        path = output_path / f"{filename}.png"
        try:
            fig.write_image(path)
            logger.info(f"Saved: {path}")
        except Exception as e:
            logger.warning(f"Failed to save image: {e}. Falling back to HTML.")
            path = output_path / f"{filename}.html"
            fig.write_html(path)


def plot_dataset_statistics(
    dataset: ClassificationDataset,
    output_path: Path,
    output_mode: str,
) -> go.Figure:
    """Plot overall dataset statistics."""
    stats = dataset.get_stats()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Samples by IVD Level",
            "Samples by Source",
            "Pfirrmann Grade Distribution",
            "Modic Type Distribution",
        ],
        specs=[
            [{"type": "bar"}, {"type": "pie"}],
            [{"type": "bar"}, {"type": "bar"}],
        ],
    )

    # IVD Level distribution
    levels = stats["levels"]
    level_order = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
    level_counts = [levels.get(lvl, 0) for lvl in level_order]
    fig.add_trace(
        go.Bar(
            x=level_order,
            y=level_counts,
            marker_color="#3498db",
            text=level_counts,
            textposition="auto",
        ),
        row=1,
        col=1,
    )

    # Source distribution (pie chart)
    sources = stats["sources"]
    fig.add_trace(
        go.Pie(
            labels=list(sources.keys()),
            values=list(sources.values()),
            marker_colors=["#2ecc71", "#e74c3c"],
        ),
        row=1,
        col=2,
    )

    # Pfirrmann distribution
    pfirrmann = stats["pfirrmann"]
    pfirrmann_labels = [f"Grade {i}" for i in range(1, 6)]
    pfirrmann_counts = [pfirrmann.get(i, 0) for i in range(1, 6)]
    fig.add_trace(
        go.Bar(
            x=pfirrmann_labels,
            y=pfirrmann_counts,
            marker_color="#9b59b6",
            text=pfirrmann_counts,
            textposition="auto",
        ),
        row=2,
        col=1,
    )

    # Modic distribution
    modic = stats["modic"]
    modic_labels = [f"Type {i}" for i in range(4)]
    modic_counts = [modic.get(i, 0) for i in range(4)]
    fig.add_trace(
        go.Bar(
            x=modic_labels,
            y=modic_counts,
            marker_color="#e67e22",
            text=modic_counts,
            textposition="auto",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=f"Classification Dataset Statistics (N={stats['num_samples']}, Patients={stats['num_patients']})",
        height=600,
        width=1000,
        showlegend=False,
    )

    _save_figure(fig, output_path, "dataset_statistics", output_mode)
    return fig


def plot_binary_label_distributions(
    dataset: ClassificationDataset,
    output_path: Path,
    output_mode: str,
) -> go.Figure:
    """Plot distributions for all binary labels."""
    binary_labels = [
        label for label in AVAILABLE_LABELS if LABEL_INFO[label]["type"] == "binary"
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
    fig = go.Figure()

    label_names = [LABEL_DISPLAY_NAMES.get(lbl, lbl) for lbl in binary_labels]
    neg_counts = [label_counts[lbl][0] for lbl in binary_labels]
    pos_counts = [label_counts[lbl][1] for lbl in binary_labels]

    fig.add_trace(
        go.Bar(
            name="Negative (0)",
            x=label_names,
            y=neg_counts,
            marker_color="#2ecc71",
            text=neg_counts,
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Positive (1)",
            x=label_names,
            y=pos_counts,
            marker_color="#e74c3c",
            text=pos_counts,
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Binary Label Distributions",
        barmode="group",
        xaxis_title="Label",
        yaxis_title="Count",
        height=500,
        width=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    _save_figure(fig, output_path, "binary_label_distributions", output_mode)
    return fig


def plot_label_cooccurrence(
    dataset: ClassificationDataset,
    output_path: Path,
    output_mode: str,
) -> go.Figure:
    """Plot co-occurrence heatmap between binary labels."""
    binary_labels = [
        label for label in AVAILABLE_LABELS if LABEL_INFO[label]["type"] == "binary"
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

    label_names = [LABEL_DISPLAY_NAMES.get(lbl, lbl) for lbl in binary_labels]

    fig = go.Figure(
        data=go.Heatmap(
            z=cooccurrence,
            x=label_names,  # type: ignore[arg-type]
            y=label_names,  # type: ignore[arg-type]
            colorscale="Blues",
            text=cooccurrence,
            texttemplate="%{text}",
            hovertemplate="(%{x}, %{y}): %{z}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Binary Label Co-occurrence (Both Positive)",
        height=600,
        width=700,
        xaxis_title="Label",
        yaxis_title="Label",
    )

    _save_figure(fig, output_path, "label_cooccurrence", output_mode)
    return fig


def plot_samples_per_class(
    dataset: ClassificationDataset,
    data_path: Path,
    output_path: Path,
    output_mode: str,
    samples_per_class: int,
    display_size: tuple[int, int],
) -> dict[str, go.Figure]:
    """Plot sample images for each possible value of each label."""
    figures: dict[str, go.Figure] = {}

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

    for label in AVAILABLE_LABELS:
        info = LABEL_INFO[label]
        record_key = record_keys.get(label, label)

        if info["type"] == "multiclass":
            num_classes = info["num_classes"]
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

        # Create subplot titles
        subplot_titles = []
        for cv, cn in zip(class_values, class_names):
            count = len(samples_by_class.get(cv, []))
            subplot_titles.extend([f"{cn} (n={count})"] + [""] * (n_cols - 1))

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.02,
            vertical_spacing=0.08,
        )

        for row_idx, (class_val, class_name) in enumerate(zip(class_values, class_names)):
            samples = samples_by_class.get(class_val, [])

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
                row = row_idx + 1
                col = col_idx + 1

                if col_idx < len(selected):
                    record = selected[col_idx]
                    img = _load_sample_image(
                        data_path,
                        record["source"],
                        record["patient_id"],
                        record["ivd_level"],
                        display_size,
                    )

                    fig.add_trace(go.Image(z=img), row=row, col=col)

                    # Add annotation with level info
                    level = IDX_TO_LEVEL.get(record["level_idx"], "")
                    fig.add_annotation(
                        x=display_size[1] // 2,
                        y=display_size[0] - 10,
                        text=f"{level}",
                        showarrow=False,
                        font=dict(size=10, color="white"),
                        bgcolor="rgba(0,0,0,0.7)",
                        xanchor="center",
                        row=row,
                        col=col,
                    )
                else:
                    # Empty placeholder
                    placeholder = np.ones(
                        (display_size[0], display_size[1], 3), dtype=np.uint8
                    ) * 200
                    fig.add_trace(go.Image(z=placeholder), row=row, col=col)

                fig.update_xaxes(showticklabels=False, row=row, col=col)
                fig.update_yaxes(showticklabels=False, row=row, col=col)

        display_name = LABEL_DISPLAY_NAMES.get(label, label)
        color = LABEL_COLORS.get(label, "#333333")

        fig.update_layout(
            title=f"<span style='color:{color}'><b>{display_name}</b></span> - Samples per Class Value",
            height=280 * n_rows,
            width=280 * n_cols,
            showlegend=False,
        )

        _save_figure(fig, output_path, f"samples_{label}", output_mode)
        figures[label] = fig

    return figures


def plot_pfirrmann_by_level(
    dataset: ClassificationDataset,
    output_path: Path,
    output_mode: str,
) -> go.Figure:
    """Plot Pfirrmann grade distribution by IVD level."""
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

    fig = go.Figure()

    colors = ["#27ae60", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

    for grade in range(1, 6):
        counts = [level_pfirrmann[lvl][grade] for lvl in levels]
        fig.add_trace(
            go.Bar(
                name=f"Grade {grade}",
                x=levels,
                y=counts,
                marker_color=colors[grade - 1],
                text=counts,
                textposition="auto",
            )
        )

    fig.update_layout(
        title="Pfirrmann Grade Distribution by IVD Level",
        barmode="stack",
        xaxis_title="IVD Level",
        yaxis_title="Count",
        height=500,
        width=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    _save_figure(fig, output_path, "pfirrmann_by_level", output_mode)
    return fig


def main(config: AnalyzeConfig) -> None:
    """Run dataset analysis."""
    setup_logger(verbose=config.verbose)

    logger.info(f"Loading dataset from {config.data_path}")

    # Load dataset with all splits combined
    dataset = ClassificationDataset(
        data_path=config.data_path,
        split="all",
        output_size=config.display_size,
        augment=False,
        normalize=False,
    )

    logger.info(f"Loaded {len(dataset)} samples")

    # Create output directory
    config.output_path.mkdir(parents=True, exist_ok=True)

    # Generate all visualizations
    logger.info("Generating dataset statistics...")
    plot_dataset_statistics(dataset, config.output_path, config.output_mode)

    logger.info("Generating binary label distributions...")
    plot_binary_label_distributions(dataset, config.output_path, config.output_mode)

    logger.info("Generating label co-occurrence heatmap...")
    plot_label_cooccurrence(dataset, config.output_path, config.output_mode)

    logger.info("Generating Pfirrmann by level distribution...")
    plot_pfirrmann_by_level(dataset, config.output_path, config.output_mode)

    logger.info("Generating samples per class for each label...")
    plot_samples_per_class(
        dataset,
        config.data_path,
        config.output_path,
        config.output_mode,
        config.samples_per_class,
        config.display_size,
    )

    logger.info(f"Analysis complete. Results saved to {config.output_path}")


if __name__ == "__main__":
    import tyro

    config = tyro.cli(AnalyzeConfig)
    main(config)

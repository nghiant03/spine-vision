"""Training curves visualization using seaborn."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from spine_vision.visualization.base import save_figure


def plot_training_curves(
    history: dict[str, list[float]],
    output_path: Path | None = None,
    filename: str = "training_curves",
    output_mode: Literal["browser", "html", "image"] = "image",
) -> Figure:
    """Plot training loss and metrics over epochs.

    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss', 'lr', etc.
        output_path: Directory for saving visualizations.
        filename: Output filename.
        output_mode: Output format.

    Returns:
        Matplotlib Figure.
    """
    has_loss = "train_loss" in history
    has_lr = "lr" in history
    metric_keys = [k for k in history if k not in ["train_loss", "val_loss", "lr"]]
    has_metrics = len(metric_keys) > 0

    n_rows = sum([has_loss, has_lr, has_metrics])
    if n_rows == 0:
        n_rows = 1

    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3 * n_rows), squeeze=False)
    axes = axes.flatten()
    row = 0

    sns.set_style("whitegrid")

    # Loss curves
    if has_loss:
        ax = axes[row]
        epochs = list(range(1, len(history["train_loss"]) + 1))
        ax.plot(epochs, history["train_loss"], label="Train Loss", color="blue")
        if "val_loss" in history and history["val_loss"]:
            val_epochs = list(range(1, len(history["val_loss"]) + 1))
            ax.plot(val_epochs, history["val_loss"], label="Val Loss", color="red")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.legend()
        row += 1

    # Metrics
    if has_metrics:
        ax = axes[row]
        colors = ["green", "orange", "purple", "cyan", "magenta"]
        for i, key in enumerate(metric_keys[:5]):
            if history[key]:
                epochs = list(range(1, len(history[key]) + 1))
                ax.plot(epochs, history[key], label=key, color=colors[i % len(colors)])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title("Metrics")
        ax.legend()
        row += 1

    # Learning rate
    if has_lr:
        ax = axes[row]
        epochs = list(range(1, len(history["lr"]) + 1))
        ax.plot(epochs, history["lr"], color="gray")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_yscale("log")
        ax.set_title("Learning Rate")

    fig.suptitle("Training Progress", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_figure(fig, output_path, filename, output_mode)
    return fig

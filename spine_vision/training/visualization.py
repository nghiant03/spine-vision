"""Visualization utilities for training.

Provides visualization tools for:
- Training curves (loss, metrics)
- Localization predictions with ground truth overlay
- Error distribution analysis
- Per-level performance breakdown
"""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from loguru import logger
from PIL import Image
from plotly.subplots import make_subplots


class TrainingVisualizer:
    """Visualizer for training progress and validation results.

    Generates interactive plots for training curves, predictions,
    and error analysis.
    """

    def __init__(
        self,
        output_path: Path | None = None,
        output_mode: Literal["browser", "html", "image"] = "html",
    ) -> None:
        """Initialize visualizer.

        Args:
            output_path: Directory for saving visualizations.
            output_mode: Output format ('browser', 'html', 'image').
        """
        self.output_path = output_path
        self.output_mode = output_mode

        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)

    def plot_training_curves(
        self,
        history: dict[str, list[float]],
        filename: str = "training_curves",
    ) -> go.Figure:
        """Plot training loss and metrics over epochs.

        Args:
            history: Dictionary with keys like 'train_loss', 'val_loss', 'lr', etc.
            filename: Output filename.

        Returns:
            Plotly Figure.
        """
        # Count number of subplots needed
        has_loss = "train_loss" in history
        has_lr = "lr" in history
        has_metrics = any(k not in ["train_loss", "val_loss", "lr"] for k in history)

        n_rows = sum([has_loss, has_lr, has_metrics])
        if n_rows == 0:
            n_rows = 1

        titles = []
        if has_loss:
            titles.append("Loss")
        if has_metrics:
            titles.append("Metrics")
        if has_lr:
            titles.append("Learning Rate")

        fig = make_subplots(
            rows=n_rows,
            cols=1,
            subplot_titles=titles,
            vertical_spacing=0.1,
        )

        row = 1

        # Loss curves
        if has_loss:
            epochs = list(range(1, len(history["train_loss"]) + 1))
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history["train_loss"],
                    mode="lines",
                    name="Train Loss",
                    line=dict(color="blue"),
                ),
                row=row,
                col=1,
            )
            if "val_loss" in history and history["val_loss"]:
                # Handle validation frequency by creating proper x-axis
                val_epochs = list(range(1, len(history["val_loss"]) + 1))
                fig.add_trace(
                    go.Scatter(
                        x=val_epochs,
                        y=history["val_loss"],
                        mode="lines",
                        name="Val Loss",
                        line=dict(color="red"),
                    ),
                    row=row,
                    col=1,
                )
            fig.update_yaxes(title_text="Loss", row=row, col=1)
            row += 1

        # Metrics
        if has_metrics:
            metric_keys = [k for k in history if k not in ["train_loss", "val_loss", "lr"]]
            colors = ["green", "orange", "purple", "cyan", "magenta"]
            for i, key in enumerate(metric_keys[:5]):  # Limit to 5 metrics
                if history[key]:
                    epochs = list(range(1, len(history[key]) + 1))
                    fig.add_trace(
                        go.Scatter(
                            x=epochs,
                            y=history[key],
                            mode="lines",
                            name=key,
                            line=dict(color=colors[i % len(colors)]),
                        ),
                        row=row,
                        col=1,
                    )
            fig.update_yaxes(title_text="Value", row=row, col=1)
            row += 1

        # Learning rate
        if has_lr:
            epochs = list(range(1, len(history["lr"]) + 1))
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history["lr"],
                    mode="lines",
                    name="Learning Rate",
                    line=dict(color="gray"),
                ),
                row=row,
                col=1,
            )
            fig.update_yaxes(title_text="LR", type="log", row=row, col=1)

        fig.update_layout(
            title="Training Progress",
            height=300 * n_rows,
            showlegend=True,
        )
        fig.update_xaxes(title_text="Epoch")

        self._save_figure(fig, filename)
        return fig

    def plot_localization_predictions(
        self,
        images: list[np.ndarray],
        predictions: np.ndarray,
        targets: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
        num_samples: int = 16,
        filename: str = "localization_predictions",
    ) -> go.Figure:
        """Plot localization predictions overlaid on images.

        Args:
            images: List of images as numpy arrays [H, W, C] or [H, W].
            predictions: Predicted coordinates [N, 2] in relative [0, 1].
            targets: Ground truth coordinates [N, 2] in relative [0, 1].
            metadata: Optional list of metadata dicts per sample.
            num_samples: Maximum number of samples to show.
            filename: Output filename.

        Returns:
            Plotly Figure.
        """
        n_samples = min(len(images), num_samples)
        n_cols = min(4, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            horizontal_spacing=0.02,
            vertical_spacing=0.05,
        )

        for i in range(n_samples):
            row = i // n_cols + 1
            col = i % n_cols + 1

            img = images[i]
            h, w = img.shape[:2]

            # Convert to RGB if grayscale
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            # Convert coordinates to pixel space
            pred_x, pred_y = predictions[i] * [w, h]
            gt_x, gt_y = targets[i] * [w, h]

            # Add image as heatmap
            fig.add_trace(
                go.Image(z=img),
                row=row,
                col=col,
            )

            # Add ground truth point
            fig.add_trace(
                go.Scatter(
                    x=[gt_x],
                    y=[gt_y],
                    mode="markers",
                    marker=dict(color="green", size=10, symbol="x"),
                    name="GT" if i == 0 else None,
                    showlegend=i == 0,
                ),
                row=row,
                col=col,
            )

            # Add prediction point
            fig.add_trace(
                go.Scatter(
                    x=[pred_x],
                    y=[pred_y],
                    mode="markers",
                    marker=dict(color="red", size=10, symbol="circle"),
                    name="Pred" if i == 0 else None,
                    showlegend=i == 0,
                ),
                row=row,
                col=col,
            )

            # Add connecting line
            fig.add_trace(
                go.Scatter(
                    x=[gt_x, pred_x],
                    y=[gt_y, pred_y],
                    mode="lines",
                    line=dict(color="yellow", width=1, dash="dash"),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # Add title with metadata
            title = ""
            if metadata and i < len(metadata):
                level = metadata[i].get("level", "")
                title = f"{level}"

            fig.update_xaxes(
                title_text=title,
                showticklabels=False,
                row=row,
                col=col,
            )
            fig.update_yaxes(showticklabels=False, row=row, col=col)

        fig.update_layout(
            title="Localization Predictions (Green=GT, Red=Pred)",
            height=250 * n_rows,
            width=250 * n_cols,
        )

        self._save_figure(fig, filename)
        return fig

    def plot_error_distribution(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        levels: np.ndarray | None = None,
        level_names: list[str] | None = None,
        filename: str = "error_distribution",
    ) -> go.Figure:
        """Plot error distribution analysis.

        Args:
            predictions: Predicted coordinates [N, 2].
            targets: Ground truth coordinates [N, 2].
            levels: Optional level indices [N].
            level_names: Names for each level.
            filename: Output filename.

        Returns:
            Plotly Figure.
        """
        # Compute errors
        errors = predictions - targets
        distances = np.sqrt(np.sum(errors**2, axis=1))

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Distance Distribution",
                "X Error vs Y Error",
                "Error by Level",
                "Cumulative Error",
            ],
        )

        # Distance histogram
        fig.add_trace(
            go.Histogram(
                x=distances,
                nbinsx=50,
                name="Distance",
                marker_color="blue",
            ),
            row=1,
            col=1,
        )
        fig.update_xaxes(title_text="Euclidean Distance", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)

        # X vs Y error scatter
        fig.add_trace(
            go.Scatter(
                x=errors[:, 0],
                y=errors[:, 1],
                mode="markers",
                marker=dict(
                    color=distances,
                    colorscale="Viridis",
                    size=5,
                    opacity=0.5,
                ),
                name="Errors",
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="X Error", row=1, col=2)
        fig.update_yaxes(title_text="Y Error", row=1, col=2)

        # Error by level (box plot)
        if levels is not None and level_names:
            for level_idx, level_name in enumerate(level_names):
                mask = levels == level_idx
                if np.sum(mask) > 0:
                    fig.add_trace(
                        go.Box(
                            y=distances[mask],
                            name=level_name,
                        ),
                        row=2,
                        col=1,
                    )
        else:
            fig.add_trace(
                go.Box(y=distances, name="All"),
                row=2,
                col=1,
            )
        fig.update_yaxes(title_text="Distance", row=2, col=1)

        # Cumulative error (sorted)
        sorted_distances = np.sort(distances)
        cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances) * 100
        fig.add_trace(
            go.Scatter(
                x=sorted_distances,
                y=cumulative,
                mode="lines",
                name="Cumulative",
                line=dict(color="red"),
            ),
            row=2,
            col=2,
        )
        # Add reference lines for common thresholds
        for thresh in [0.02, 0.05, 0.10]:
            pct_below = float(np.mean(sorted_distances < thresh) * 100)
            fig.add_hline(
                y=pct_below,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"{pct_below:.1f}% @ {thresh}",
                row=2,
                col=2,
            )
        fig.update_xaxes(title_text="Distance Threshold", row=2, col=2)
        fig.update_yaxes(title_text="% Below Threshold", row=2, col=2)

        fig.update_layout(
            title="Error Distribution Analysis",
            height=600,
            width=900,
            showlegend=True,
        )

        self._save_figure(fig, filename)
        return fig

    def plot_per_level_metrics(
        self,
        metrics: dict[str, float],
        level_names: list[str],
        metric_prefix: str = "med_",
        filename: str = "per_level_metrics",
    ) -> go.Figure:
        """Plot per-level metric comparison.

        Args:
            metrics: Dictionary of metrics including per-level values.
            level_names: Names of levels.
            metric_prefix: Prefix for per-level metrics in dict.
            filename: Output filename.

        Returns:
            Plotly Figure.
        """
        values = []
        labels = []
        for level in level_names:
            key = f"{metric_prefix}{level}"
            if key in metrics:
                values.append(metrics[key])
                labels.append(level)

        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    text=[f"{v:.4f}" for v in values],
                    textposition="auto",
                    marker_color="steelblue",
                )
            ]
        )

        # Add average line
        if values:
            avg = float(np.mean(values))
            fig.add_hline(
                y=avg,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Avg: {avg:.4f}",
            )

        fig.update_layout(
            title=f"Per-Level {metric_prefix.upper().rstrip('_')}",
            xaxis_title="Level",
            yaxis_title="Value",
            height=400,
        )

        self._save_figure(fig, filename)
        return fig

    def visualize_sample(
        self,
        image: np.ndarray | Image.Image,
        prediction: np.ndarray,
        target: np.ndarray,
        level: str = "",
        filename: str = "sample",
    ) -> go.Figure:
        """Visualize a single sample with prediction overlay.

        Args:
            image: Image as numpy array or PIL Image.
            prediction: Predicted coordinates [2] in relative [0, 1].
            target: Ground truth coordinates [2] in relative [0, 1].
            level: Level label for title.
            filename: Output filename.

        Returns:
            Plotly Figure.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        h, w = image.shape[:2]

        # Convert to RGB if grayscale
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        pred_x, pred_y = prediction * [w, h]
        gt_x, gt_y = target * [w, h]

        # Compute error
        error = np.sqrt(np.sum((prediction - target) ** 2))

        fig = go.Figure()

        fig.add_trace(go.Image(z=image))

        # Ground truth
        fig.add_trace(
            go.Scatter(
                x=[gt_x],
                y=[gt_y],
                mode="markers+text",
                marker=dict(color="green", size=15, symbol="x"),
                text=["GT"],
                textposition="top center",
                name="Ground Truth",
            )
        )

        # Prediction
        fig.add_trace(
            go.Scatter(
                x=[pred_x],
                y=[pred_y],
                mode="markers+text",
                marker=dict(color="red", size=15, symbol="circle"),
                text=["Pred"],
                textposition="bottom center",
                name="Prediction",
            )
        )

        # Connecting line
        fig.add_trace(
            go.Scatter(
                x=[gt_x, pred_x],
                y=[gt_y, pred_y],
                mode="lines",
                line=dict(color="yellow", width=2, dash="dash"),
                name=f"Error: {error:.4f}",
            )
        )

        fig.update_layout(
            title=f"Sample Visualization - {level} (Error: {error:.4f})",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False, scaleanchor="x"),
            height=500,
            width=500,
        )

        self._save_figure(fig, filename)
        return fig

    def _save_figure(self, fig: go.Figure, filename: str) -> None:
        """Save figure according to output mode."""
        if self.output_mode == "browser":
            fig.show()
        elif self.output_mode == "html" and self.output_path:
            path = self.output_path / f"{filename}.html"
            fig.write_html(path)
            logger.debug(f"Saved visualization: {path}")
        elif self.output_mode == "image" and self.output_path:
            path = self.output_path / f"{filename}.png"
            try:
                fig.write_image(path)
                logger.debug(f"Saved image: {path}")
            except Exception as e:
                logger.warning(f"Failed to save image: {e}. Falling back to HTML.")
                path = self.output_path / f"{filename}.html"
                fig.write_html(path)

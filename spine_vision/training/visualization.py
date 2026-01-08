"""Visualization utilities for training.

Provides visualization tools for:
- Training curves (loss, metrics)
- Localization predictions with ground truth overlay
- Classification predictions with label overlay
- Error distribution analysis
- Per-level performance breakdown
- Test sample visualization with labels
- Classification confusion analysis (TP/TN/FP/FN examples)

Supports optional wandb logging for experiment tracking.
"""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from loguru import logger
from PIL import Image
from plotly.subplots import make_subplots


# Label display names and color mapping for classification visualization
LABEL_DISPLAY_NAMES: dict[str, str] = {
    "pfirrmann": "Pfirrmann",
    "modic": "Modic",
    "herniation": "Herniation",
    "bulging": "Bulging",
    "upper_endplate": "Upper Endplate",
    "lower_endplate": "Lower Endplate",
    "spondy": "Spondylolisthesis",
    "narrowing": "Narrowing",
}

# Color palette for labels (hex colors for plotly)
LABEL_COLORS: dict[str, str] = {
    "pfirrmann": "#1f77b4",  # Blue
    "modic": "#ff7f0e",  # Orange
    "herniation": "#2ca02c",  # Green
    "bulging": "#d62728",  # Red
    "upper_endplate": "#9467bd",  # Purple
    "lower_endplate": "#8c564b",  # Brown
    "spondy": "#e377c2",  # Pink
    "narrowing": "#7f7f7f",  # Gray
}


def _safe_to_int(value: Any) -> int:
    """Safely convert a value to int, handling numpy arrays and scalars."""
    if isinstance(value, np.ndarray):
        return int(value.item()) if value.ndim == 0 else int(value.flat[0])
    return int(value)


def _extract_prediction_value(pred: Any, gt: Any) -> tuple[int, int]:
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
    # Binary case: single element
    if isinstance(pred, (np.ndarray, list)):
        pred_arr = np.atleast_1d(pred)
        if len(pred_arr) == 1:
            # Binary: threshold at 0.5
            pred_val = int(float(pred_arr[0]) > 0.5)
            gt_arr = np.atleast_1d(gt)
            gt_val = int(float(gt_arr[0])) if len(gt_arr) == 1 else _safe_to_int(gt)
            return pred_val, gt_val
        # Multiclass: argmax
        pred_val = int(np.argmax(pred_arr))
        if np.isscalar(gt):
            gt_val = _safe_to_int(gt)
        else:
            gt_arr = np.atleast_1d(gt)
            gt_val = int(np.argmax(gt_arr)) if len(gt_arr) > 1 else _safe_to_int(gt_arr[0])
        return pred_val, gt_val

    # Scalar case
    return _safe_to_int(pred), _safe_to_int(gt)


def load_original_images(
    image_paths: list[Path | str],
    output_size: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    """Load original images from file paths without any transformation.

    This helper is useful for visualization purposes where you want to show
    the original images (before normalization/augmentation) alongside
    model predictions.

    Args:
        image_paths: List of paths to image files.
        output_size: Optional (H, W) to resize images for display consistency.
            If None, keeps original size.

    Returns:
        List of numpy arrays [H, W, C] in RGB format, uint8.
    """
    images = []
    for path in image_paths:
        img = Image.open(path)

        # Convert to RGB
        if img.mode == "L":
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Resize if specified
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

    Loads T2 crops from the classification dataset for visualization.
    By default uses grayscale display (T2 repeated 3x) for natural appearance.
    Set grayscale_display=False to use [T2, T1, T2] model input format.

    Args:
        data_path: Base path to classification dataset (containing images/ folder).
        metadata_list: List of metadata dicts from ClassificationDataset samples.
            Each dict should have 'source', 'patient_id', 'ivd' keys.
        output_size: Optional (H, W) to resize images for display consistency.
        grayscale_display: If True, displays T2 as grayscale (natural appearance).
            If False, uses [T2, T1, T2] format matching model input.

    Returns:
        List of numpy arrays [H, W, 3] in RGB format, uint8.
    """
    images = []
    images_dir = data_path / "images"

    for meta in metadata_list:
        source = meta.get("source", "")
        patient_id = meta.get("patient_id", "")
        ivd = meta.get("ivd", "")

        # Construct filenames
        t1_filename = f"{source}_{patient_id}_sag_t1_L{ivd}.png"
        t2_filename = f"{source}_{patient_id}_sag_t2_L{ivd}.png"

        t1_path = images_dir / t1_filename
        t2_path = images_dir / t2_filename

        # Load images
        if t2_path.exists():
            t2_img = np.array(Image.open(t2_path).convert("L"))

            if grayscale_display:
                # Grayscale display: T2 repeated 3x for natural appearance
                rgb_image = np.stack([t2_img, t2_img, t2_img], axis=-1)
            else:
                # Model input format: [T2, T1, T2]
                if t1_path.exists():
                    t1_img = np.array(Image.open(t1_path).convert("L"))
                else:
                    t1_img = t2_img  # Fallback to T2 if T1 missing
                rgb_image = np.stack([t2_img, t1_img, t2_img], axis=-1)

            # Resize if specified
            if output_size is not None:
                pil_img = Image.fromarray(rgb_image)
                pil_img = pil_img.resize(
                    (output_size[1], output_size[0]), Image.Resampling.BILINEAR
                )
                rgb_image = np.array(pil_img)

            images.append(rgb_image)
        else:
            # Fallback: create placeholder
            h, w = output_size if output_size else (128, 128)
            placeholder = np.zeros((h, w, 3), dtype=np.uint8)
            images.append(placeholder)
            logger.warning(f"Image not found: {t2_path}")

    return images


class TrainingVisualizer:
    """Visualizer for training progress and validation results.

    Generates interactive plots for training curves, predictions,
    and error analysis. Optionally logs to wandb.
    """

    def __init__(
        self,
        output_path: Path | None = None,
        output_mode: Literal["browser", "html", "image"] = "html",
        use_wandb: bool = False,
    ) -> None:
        """Initialize visualizer.

        Args:
            output_path: Directory for saving visualizations.
            output_mode: Output format ('browser', 'html', 'image').
            use_wandb: If True, also log visualizations to wandb.
        """
        self.output_path = output_path
        self.output_mode = output_mode
        self.use_wandb = use_wandb
        self._wandb: Any = None

        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)

        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                logger.warning("wandb not installed. Disabling wandb logging.")
                self.use_wandb = False

    def plot_training_curves(
        self,
        history: dict[str, list[float]],
        filename: str = "training_curves",
        log_to_wandb: bool | None = None,
    ) -> go.Figure:
        """Plot training loss and metrics over epochs.

        Args:
            history: Dictionary with keys like 'train_loss', 'val_loss', 'lr', etc.
            filename: Output filename.
            log_to_wandb: Override default wandb logging setting.

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
            metric_keys = [
                k for k in history if k not in ["train_loss", "val_loss", "lr"]
            ]
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

        self._save_figure(fig, filename, log_to_wandb)
        return fig

    def plot_localization_predictions(
        self,
        images: list[np.ndarray],
        predictions: np.ndarray,
        targets: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
        num_samples: int = 16,
        filename: str = "localization_predictions",
        log_to_wandb: bool | None = None,
    ) -> go.Figure:
        """Plot localization predictions overlaid on images.

        Args:
            images: List of images as numpy arrays [H, W, C] or [H, W].
            predictions: Predicted coordinates [N, 2] in relative [0, 1].
            targets: Ground truth coordinates [N, 2] in relative [0, 1].
            metadata: Optional list of metadata dicts per sample.
            num_samples: Maximum number of samples to show.
            filename: Output filename.
            log_to_wandb: Override default wandb logging setting.

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

        self._save_figure(fig, filename, log_to_wandb)

        # Also log individual images to wandb if enabled
        if self._should_log_to_wandb(log_to_wandb) and self._wandb is not None:
            self._log_prediction_images_to_wandb(
                images[:n_samples],
                predictions[:n_samples],
                targets[:n_samples],
                metadata[:n_samples] if metadata else None,
            )

        return fig

    def _log_prediction_images_to_wandb(
        self,
        images: list[np.ndarray],
        predictions: np.ndarray,
        targets: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Log prediction images to wandb."""
        if self._wandb is None:
            return

        wandb_images = []
        for i, img in enumerate(images):
            h, w = img.shape[:2]
            pred_x, pred_y = predictions[i] * [w, h]
            gt_x, gt_y = targets[i] * [w, h]

            caption = ""
            if metadata and i < len(metadata):
                caption = metadata[i].get("level", "")

            # Create wandb image with bounding boxes
            boxes = [
                {
                    "position": {"middle": [float(gt_x), float(gt_y)], "width": 10, "height": 10},
                    "class_id": 0,
                    "box_caption": "GT",
                },
                {
                    "position": {"middle": [float(pred_x), float(pred_y)], "width": 10, "height": 10},
                    "class_id": 1,
                    "box_caption": "Pred",
                },
            ]

            wandb_img = self._wandb.Image(
                img,
                caption=caption,
                boxes={"predictions": {"box_data": boxes, "class_labels": {0: "GT", 1: "Pred"}}},
            )
            wandb_images.append(wandb_img)

        self._wandb.log({"predictions": wandb_images})

    def plot_error_distribution(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        levels: np.ndarray | None = None,
        level_names: list[str] | None = None,
        filename: str = "error_distribution",
        log_to_wandb: bool | None = None,
    ) -> go.Figure:
        """Plot error distribution analysis.

        Args:
            predictions: Predicted coordinates [N, 2].
            targets: Ground truth coordinates [N, 2].
            levels: Optional level indices [N].
            level_names: Names for each level.
            filename: Output filename.
            log_to_wandb: Override default wandb logging setting.

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
        cumulative = (
            np.arange(1, len(sorted_distances) + 1) / len(sorted_distances) * 100
        )
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

        self._save_figure(fig, filename, log_to_wandb)

        # Log summary stats to wandb
        if self._should_log_to_wandb(log_to_wandb) and self._wandb is not None:
            self._wandb.log({
                "error/mean_distance": float(np.mean(distances)),
                "error/std_distance": float(np.std(distances)),
                "error/median_distance": float(np.median(distances)),
                "error/max_distance": float(np.max(distances)),
            })

        return fig

    def plot_per_level_metrics(
        self,
        metrics: dict[str, float],
        level_names: list[str],
        metric_prefix: str = "med_",
        filename: str = "per_level_metrics",
        log_to_wandb: bool | None = None,
    ) -> go.Figure:
        """Plot per-level metric comparison.

        Args:
            metrics: Dictionary of metrics including per-level values.
            level_names: Names of levels.
            metric_prefix: Prefix for per-level metrics in dict.
            filename: Output filename.
            log_to_wandb: Override default wandb logging setting.

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

        self._save_figure(fig, filename, log_to_wandb)

        # Log per-level metrics to wandb
        if self._should_log_to_wandb(log_to_wandb) and self._wandb is not None:
            wandb_metrics = {f"per_level/{label}": val for label, val in zip(labels, values)}
            if values:
                wandb_metrics["per_level/average"] = float(np.mean(values))
            self._wandb.log(wandb_metrics)

        return fig

    def visualize_sample(
        self,
        image: np.ndarray | Image.Image,
        prediction: np.ndarray,
        target: np.ndarray,
        level: str = "",
        filename: str = "sample",
        log_to_wandb: bool | None = None,
    ) -> go.Figure:
        """Visualize a single sample with prediction overlay.

        Args:
            image: Image as numpy array or PIL Image.
            prediction: Predicted coordinates [2] in relative [0, 1].
            target: Ground truth coordinates [2] in relative [0, 1].
            level: Level label for title.
            filename: Output filename.
            log_to_wandb: Override default wandb logging setting.

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

        self._save_figure(fig, filename, log_to_wandb)
        return fig

    def log_table(
        self,
        data: dict[str, list[Any]],
        table_name: str = "results",
    ) -> None:
        """Log a table to wandb.

        Args:
            data: Dictionary where keys are column names and values are column data.
            table_name: Name for the wandb table.
        """
        if self._wandb is not None and self.use_wandb:
            table = self._wandb.Table(columns=list(data.keys()))
            n_rows = len(next(iter(data.values())))
            for i in range(n_rows):
                row = [data[col][i] for col in data.keys()]
                table.add_data(*row)
            self._wandb.log({table_name: table})

    def _should_log_to_wandb(self, override: bool | None) -> bool:
        """Determine if should log to wandb."""
        if override is not None:
            return override and self._wandb is not None
        return self.use_wandb and self._wandb is not None

    def _save_figure(
        self,
        fig: go.Figure,
        filename: str,
        log_to_wandb: bool | None = None,
    ) -> None:
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

        # Log to wandb if enabled
        if self._should_log_to_wandb(log_to_wandb) and self._wandb is not None:
            self._wandb.log({filename: fig})

    # ==================== Classification Visualization ====================

    def plot_classification_predictions(
        self,
        images: list[np.ndarray],
        predictions: dict[str, np.ndarray],
        targets: dict[str, np.ndarray],
        metadata: list[dict[str, Any]] | None = None,
        num_samples: int = 16,
        filename: str = "classification_predictions",
        log_to_wandb: bool | None = None,
    ) -> go.Figure:
        """Plot classification predictions overlaid on images.

        Shows images with ground truth and predicted labels as text overlays.
        Highlights correct predictions in green and incorrect ones in red.

        Args:
            images: List of images as numpy arrays [H, W, C] or [H, W].
            predictions: Dict mapping label names to predicted values [N] or [N, C].
            targets: Dict mapping label names to ground truth values [N] or [N, C].
            metadata: Optional list of metadata dicts per sample (level, patient_id).
            num_samples: Maximum number of samples to show.
            filename: Output filename.
            log_to_wandb: Override default wandb logging setting.

        Returns:
            Plotly Figure.
        """
        n_samples = min(len(images), num_samples)
        n_cols = min(4, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            horizontal_spacing=0.03,
            vertical_spacing=0.08,
        )

        labels = list(predictions.keys())

        for i in range(n_samples):
            row = i // n_cols + 1
            col = i % n_cols + 1

            img = images[i]
            h, w = img.shape[:2]

            # Convert to RGB if grayscale
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            # Add image
            fig.add_trace(go.Image(z=img), row=row, col=col)

            # Build annotation text
            annotations = []
            all_correct = True

            for label in labels:
                pred = predictions[label][i]
                gt = targets[label][i]

                pred_val, gt_val = _extract_prediction_value(pred, gt)

                is_correct = pred_val == gt_val
                if not is_correct:
                    all_correct = False

                display_name = LABEL_DISPLAY_NAMES.get(label, label)
                status = "✓" if is_correct else "✗"
                annotations.append(f"{display_name}: {pred_val} ({gt_val}) {status}")

            # Determine title color based on overall correctness
            title_color = "green" if all_correct else "red"

            # Add metadata to title
            title_parts = []
            if metadata and i < len(metadata):
                level = metadata[i].get("level", "")
                if level:
                    title_parts.append(level)

            # Build subtitle with label predictions
            subtitle = " | ".join(annotations[:3])  # Show first 3 labels
            if len(annotations) > 3:
                subtitle += f" +{len(annotations) - 3}"

            title = " ".join(title_parts) if title_parts else f"Sample {i + 1}"

            fig.update_xaxes(
                title_text=f"<b>{title}</b><br><sub>{subtitle}</sub>",
                showticklabels=False,
                row=row,
                col=col,
            )
            fig.update_yaxes(showticklabels=False, row=row, col=col)

            # Add a colored border annotation to indicate correct/incorrect
            fig.add_shape(
                type="rect",
                x0=0,
                y0=0,
                x1=w,
                y1=h,
                line=dict(color=title_color, width=3),
                row=row,
                col=col,
            )

        fig.update_layout(
            title="Classification Predictions (Pred (GT) - Green=Correct, Red=Incorrect)",
            height=300 * n_rows,
            width=280 * n_cols,
            showlegend=False,
        )

        self._save_figure(fig, filename, log_to_wandb)

        # Also log individual images to wandb if enabled
        if self._should_log_to_wandb(log_to_wandb) and self._wandb is not None:
            self._log_classification_images_to_wandb(
                images[:n_samples],
                {k: v[:n_samples] for k, v in predictions.items()},
                {k: v[:n_samples] for k, v in targets.items()},
                metadata[:n_samples] if metadata else None,
            )

        return fig

    def _log_classification_images_to_wandb(
        self,
        images: list[np.ndarray],
        predictions: dict[str, np.ndarray],
        targets: dict[str, np.ndarray],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Log classification prediction images to wandb."""
        if self._wandb is None:
            return

        wandb_images = []
        labels = list(predictions.keys())

        for i, img in enumerate(images):
            # Build caption with predictions
            caption_parts = []

            if metadata and i < len(metadata):
                level = metadata[i].get("level", "")
                if level:
                    caption_parts.append(f"Level: {level}")

            for label in labels:
                pred = predictions[label][i]
                gt = targets[label][i]

                pred_val, gt_val = _extract_prediction_value(pred, gt)

                display_name = LABEL_DISPLAY_NAMES.get(label, label)
                status = "✓" if pred_val == gt_val else "✗"
                caption_parts.append(f"{display_name}: {pred_val}/{gt_val} {status}")

            caption = " | ".join(caption_parts)
            wandb_img = self._wandb.Image(img, caption=caption)
            wandb_images.append(wandb_img)

        self._wandb.log({"classification_predictions": wandb_images})

    def plot_classification_metrics(
        self,
        metrics: dict[str, float],
        target_labels: list[str] | None = None,
        filename: str = "classification_metrics",
        log_to_wandb: bool | None = None,
    ) -> go.Figure:
        """Plot per-label classification metrics.

        Creates bar charts for accuracy, F1, and other metrics grouped by label.

        Args:
            metrics: Dictionary of metrics (e.g., pfirrmann_accuracy, modic_f1).
            target_labels: List of labels to include. If None, auto-detect from metrics.
            filename: Output filename.
            log_to_wandb: Override default wandb logging setting.

        Returns:
            Plotly Figure.
        """
        # Group metrics by type (accuracy, f1, precision, recall)
        metric_types = ["accuracy", "f1", "precision", "recall"]
        labels = target_labels or list(LABEL_DISPLAY_NAMES.keys())

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[t.title() for t in metric_types],
            horizontal_spacing=0.1,
            vertical_spacing=0.15,
        )

        for idx, metric_type in enumerate(metric_types):
            row = idx // 2 + 1
            col = idx % 2 + 1

            values = []
            label_names = []

            for label in labels:
                key = f"{label}_{metric_type}"
                if key in metrics:
                    values.append(metrics[key])
                    label_names.append(LABEL_DISPLAY_NAMES.get(label, label))

            if values:
                colors = [LABEL_COLORS.get(label, "#333333") for label in labels[:len(values)]]

                fig.add_trace(
                    go.Bar(
                        x=label_names,
                        y=values,
                        text=[f"{v:.3f}" for v in values],
                        textposition="auto",
                        marker_color=colors,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

                fig.update_yaxes(range=[0, 1], row=row, col=col)

        # Add overall metrics if available
        overall_metrics = {k: v for k, v in metrics.items() if k.startswith("overall_")}
        if overall_metrics:
            title = "Per-Label Classification Metrics | "
            title += " | ".join([f"{k.replace('overall_', '').title()}: {v:.3f}" for k, v in overall_metrics.items()])
        else:
            title = "Per-Label Classification Metrics"

        fig.update_layout(
            title=title,
            height=500,
            width=900,
        )

        self._save_figure(fig, filename, log_to_wandb)

        # Log to wandb
        if self._should_log_to_wandb(log_to_wandb) and self._wandb is not None:
            self._wandb.log({f"metrics/{k}": v for k, v in metrics.items()})

        return fig

    def plot_confusion_matrices(
        self,
        confusion_matrices: dict[str, np.ndarray],
        class_names: dict[str, list[str]] | None = None,
        filename: str = "confusion_matrices",
        log_to_wandb: bool | None = None,
    ) -> go.Figure:
        """Plot confusion matrices for classification labels.

        Args:
            confusion_matrices: Dict mapping label names to confusion matrices [C, C].
            class_names: Dict mapping label names to class name lists.
            filename: Output filename.
            log_to_wandb: Override default wandb logging setting.

        Returns:
            Plotly Figure.
        """
        labels = list(confusion_matrices.keys())
        n_labels = len(labels)

        if n_labels == 0:
            logger.warning("No confusion matrices to plot")
            return go.Figure()

        n_cols = min(3, n_labels)
        n_rows = (n_labels + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[LABEL_DISPLAY_NAMES.get(lbl, lbl) for lbl in labels],
            horizontal_spacing=0.1,
            vertical_spacing=0.15,
        )

        for idx, label in enumerate(labels):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            cm = confusion_matrices[label]
            n_classes = cm.shape[0]

            # Normalize confusion matrix
            cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero

            # Get class names
            if class_names and label in class_names:
                names = class_names[label]
            else:
                names = [str(i) for i in range(n_classes)]

            # Create heatmap
            fig.add_trace(
                go.Heatmap(
                    z=cm_normalized,
                    x=names,  # type: ignore[arg-type]  # Plotly accepts strings for categorical
                    y=names,  # type: ignore[arg-type]  # Plotly accepts strings for categorical
                    colorscale="Blues",
                    showscale=idx == 0,  # Only show colorbar for first
                    text=cm,  # Show raw counts as text
                    texttemplate="%{text}",
                    hovertemplate="Pred: %{x}<br>True: %{y}<br>Count: %{text}<br>Rate: %{z:.2f}<extra></extra>",
                ),
                row=row,
                col=col,
            )

            fig.update_xaxes(title_text="Predicted", row=row, col=col)
            fig.update_yaxes(title_text="True", row=row, col=col)

        fig.update_layout(
            title="Confusion Matrices",
            height=350 * n_rows,
            width=350 * n_cols,
        )

        self._save_figure(fig, filename, log_to_wandb)

        # Log to wandb
        if self._should_log_to_wandb(log_to_wandb) and self._wandb is not None:
            # Log as wandb confusion matrix artifact
            for label, cm in confusion_matrices.items():
                names = class_names.get(label, [str(i) for i in range(cm.shape[0])]) if class_names else [str(i) for i in range(cm.shape[0])]
                self._wandb.log({f"confusion_matrix/{label}": self._wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=list(range(cm.shape[0])) * int(cm.sum()),  # placeholder
                    preds=list(range(cm.shape[1])) * int(cm.sum()),  # placeholder
                    class_names=names,
                )})

        return fig

    def plot_confusion_matrix_with_samples(
        self,
        images: list[np.ndarray],
        predictions: dict[str, np.ndarray],
        targets: dict[str, np.ndarray],
        target_label: str,
        metadata: list[dict[str, Any]] | None = None,
        class_names: list[str] | None = None,
        max_samples_per_cell: int = 4,
        filename: str | None = None,
        log_to_wandb: bool | None = None,
    ) -> go.Figure:
        """Plot confusion matrix with sample images from each cell.

        Creates a visualization showing the confusion matrix as a heatmap with
        sample images from each cell (true_class, pred_class) displayed in a grid.
        Only cells with data are shown.

        Args:
            images: List of ORIGINAL images as numpy arrays [H, W, C] or [H, W].
            predictions: Dict mapping label names to predicted values [N] or [N, C].
            targets: Dict mapping label names to ground truth values [N] or [N, C].
            target_label: Which label to analyze (e.g., "pfirrmann", "herniation").
            metadata: Optional list of metadata dicts per sample.
            class_names: Optional list of class names for display.
            max_samples_per_cell: Maximum samples to show per confusion matrix cell.
            filename: Output filename. Defaults to "confusion_matrix_samples_{target_label}".
            log_to_wandb: Override default wandb logging setting.

        Returns:
            Plotly Figure with confusion matrix and sample images.
        """
        if target_label not in predictions:
            logger.warning(f"Label '{target_label}' not found in predictions")
            return go.Figure()

        pred_arr = np.atleast_1d(predictions[target_label])
        gt_arr = np.atleast_1d(targets[target_label])

        # Convert to class indices
        n_samples = len(images)
        pred_classes = np.zeros(n_samples, dtype=int)
        gt_classes = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            pred_classes[i], gt_classes[i] = _extract_prediction_value(
                pred_arr[i] if pred_arr.ndim > 1 else pred_arr[i],
                gt_arr[i] if gt_arr.ndim > 1 else gt_arr[i],
            )

        # Get unique classes and build confusion matrix
        unique_classes = sorted(set(pred_classes) | set(gt_classes))
        n_classes = len(unique_classes)
        class_to_idx = {c: i for i, c in enumerate(unique_classes)}

        # Build confusion matrix
        cm = np.zeros((n_classes, n_classes), dtype=int)
        # Build sample indices for each cell: cell_samples[gt_idx][pred_idx] = list of sample indices
        cell_samples: dict[tuple[int, int], list[int]] = {}

        for i in range(n_samples):
            gt_idx = class_to_idx[gt_classes[i]]
            pred_idx = class_to_idx[pred_classes[i]]
            cm[gt_idx, pred_idx] += 1
            key = (gt_idx, pred_idx)
            if key not in cell_samples:
                cell_samples[key] = []
            cell_samples[key].append(i)

        # Get class names for display
        if class_names is not None:
            display_names = class_names
        else:
            display_names = [str(c) for c in unique_classes]

        # Find non-empty cells
        non_empty_cells = [(gt_idx, pred_idx) for (gt_idx, pred_idx), samples in cell_samples.items() if samples]

        if not non_empty_cells:
            logger.warning(f"No samples found for label '{target_label}'")
            return go.Figure()

        # Layout: 2 columns - confusion matrix on left, sample images on right
        display_name = LABEL_DISPLAY_NAMES.get(target_label, target_label)
        n_cell_rows = len(non_empty_cells)

        # Build subplot titles: confusion matrix + sample cells
        subplot_titles = [f"{display_name} Confusion Matrix"]
        for gt_idx, pred_idx in sorted(non_empty_cells):
            gt_name = display_names[gt_idx]
            pred_name = display_names[pred_idx]
            n_cell_samples = len(cell_samples[(gt_idx, pred_idx)])
            status = "Correct" if gt_idx == pred_idx else "Misclassified"
            subplot_titles.append(f"GT={gt_name} → Pred={pred_name} ({status}, n={n_cell_samples})")

        # Create 2-column layout: col1 = confusion matrix (spans all rows), col2 = sample images
        # Use row_heights to ensure proper spacing
        specs: list[list[dict[str, str | bool | int | float] | None]] = []
        row_heights: list[float] = []

        for i in range(n_cell_rows):
            if i == 0:
                # First row: confusion matrix (spanning all rows) on left, first sample on right
                specs.append([{"type": "heatmap", "rowspan": n_cell_rows}, {"type": "image"}])
            else:
                # Remaining rows: None for left (spanned), sample image on right
                specs.append([None, {"type": "image"}])
            row_heights.append(1.0)

        # Handle edge case: if no cells, just show confusion matrix
        if n_cell_rows == 0:
            specs = [[{"type": "heatmap"}]]
            row_heights = [1.0]
            n_cols = 1
        else:
            n_cols = 2

        fig = make_subplots(
            rows=max(1, n_cell_rows),
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
            specs=specs,
            row_heights=row_heights,
            column_widths=[0.4, 0.6] if n_cols == 2 else [1.0],
        )

        # Add confusion matrix heatmap (always at row=1, col=1)
        cm_normalized = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

        fig.add_trace(
            go.Heatmap(
                z=cm_normalized,
                x=display_names,  # type: ignore[arg-type]
                y=display_names,  # type: ignore[arg-type]
                colorscale="Blues",
                showscale=True,
                text=cm,
                texttemplate="%{text}",
                hovertemplate="Pred: %{x}<br>True: %{y}<br>Count: %{text}<br>Rate: %{z:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.update_xaxes(title_text="Predicted", row=1, col=1)
        fig.update_yaxes(title_text="True", row=1, col=1)

        # Add sample images for each non-empty cell (in column 2)
        for cell_row_idx, (gt_idx, pred_idx) in enumerate(sorted(non_empty_cells)):
            row = cell_row_idx + 1  # Rows start at 1

            sample_indices = cell_samples[(gt_idx, pred_idx)]
            np.random.shuffle(sample_indices)
            selected_indices = sample_indices[:max_samples_per_cell]

            # Create a composite image by concatenating samples horizontally
            sample_images = []
            for sample_idx in selected_indices:
                img = images[sample_idx]
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                sample_images.append(img)

            if sample_images:
                # Resize all images to same size for concatenation
                target_h = min(img.shape[0] for img in sample_images)
                target_w = min(img.shape[1] for img in sample_images)

                resized_images = []
                for img in sample_images:
                    from PIL import Image as PILImage
                    pil_img = PILImage.fromarray(img)
                    pil_img = pil_img.resize((target_w, target_h), PILImage.Resampling.BILINEAR)
                    resized_images.append(np.array(pil_img))

                # Concatenate horizontally with small gaps
                gap_width = 4
                gap_color = np.array([255, 255, 255], dtype=np.uint8)
                gap = np.full((target_h, gap_width, 3), gap_color, dtype=np.uint8)

                composite_parts = []
                for i, img in enumerate(resized_images):
                    composite_parts.append(img)
                    if i < len(resized_images) - 1:
                        composite_parts.append(gap)

                composite_image = np.concatenate(composite_parts, axis=1)

                # Determine border color based on correctness
                is_correct = gt_idx == pred_idx
                border_color = np.array([46, 204, 113], dtype=np.uint8) if is_correct else np.array([231, 76, 60], dtype=np.uint8)

                # Add border
                border_width = 4
                bordered_image = np.pad(
                    composite_image,
                    ((border_width, border_width), (border_width, border_width), (0, 0)),
                    mode='constant',
                    constant_values=0,
                )
                # Set border color
                bordered_image[:border_width, :] = border_color
                bordered_image[-border_width:, :] = border_color
                bordered_image[:, :border_width] = border_color
                bordered_image[:, -border_width:] = border_color

                fig.add_trace(go.Image(z=bordered_image), row=row, col=2)

            fig.update_xaxes(showticklabels=False, row=row, col=2)
            fig.update_yaxes(showticklabels=False, row=row, col=2)

        # Calculate appropriate figure dimensions
        sample_row_height = 120
        total_height = max(400, n_cell_rows * sample_row_height)

        fig.update_layout(
            title=f"Confusion Matrix with Samples - {display_name}",
            height=total_height,
            width=max(900, max_samples_per_cell * 150 + 400),
            showlegend=False,
        )

        output_filename = filename or f"confusion_matrix_samples_{target_label}"
        self._save_figure(fig, output_filename, log_to_wandb)

        # Log to wandb
        if self._should_log_to_wandb(log_to_wandb) and self._wandb is not None:
            self._wandb.log({
                f"confusion_matrix_samples/{target_label}": fig,
            })

        return fig

    def plot_confusion_matrices_with_samples(
        self,
        images: list[np.ndarray],
        predictions: dict[str, np.ndarray],
        targets: dict[str, np.ndarray],
        target_labels: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
        class_names: dict[str, list[str]] | None = None,
        max_samples_per_cell: int = 4,
        filename_prefix: str = "confusion_matrix_samples",
        log_to_wandb: bool | None = None,
    ) -> dict[str, go.Figure]:
        """Plot confusion matrices with samples for multiple labels.

        Generates one figure per label, each showing the confusion matrix
        as a heatmap with sample images from each cell.

        Args:
            images: List of images as numpy arrays [H, W, C] or [H, W].
            predictions: Dict mapping label names to predicted values.
            targets: Dict mapping label names to ground truth values.
            target_labels: Labels to visualize. If None, uses all in predictions.
            metadata: Optional list of metadata dicts per sample.
            class_names: Dict mapping label names to class name lists.
            max_samples_per_cell: Maximum samples per confusion matrix cell.
            filename_prefix: Prefix for output filenames.
            log_to_wandb: Override default wandb logging setting.

        Returns:
            Dictionary mapping label names to their Plotly figures.
        """
        labels = target_labels or list(predictions.keys())
        figures: dict[str, go.Figure] = {}

        for label in labels:
            if label not in predictions:
                logger.warning(f"Label '{label}' not found in predictions, skipping")
                continue

            label_class_names = class_names.get(label) if class_names else None

            fig = self.plot_confusion_matrix_with_samples(
                images=images,
                predictions=predictions,
                targets=targets,
                target_label=label,
                metadata=metadata,
                class_names=label_class_names,
                max_samples_per_cell=max_samples_per_cell,
                filename=f"{filename_prefix}_{label}",
                log_to_wandb=log_to_wandb,
            )
            figures[label] = fig

        return figures

    def plot_test_samples_with_labels(
        self,
        images: list[np.ndarray],
        predictions: dict[str, np.ndarray],
        targets: dict[str, np.ndarray],
        metadata: list[dict[str, Any]] | None = None,
        num_samples: int = 16,
        filename: str = "test_samples",
        log_to_wandb: bool | None = None,
    ) -> go.Figure:
        """Plot test samples with predicted and ground truth labels overlaid.

        Creates a grid of test images with label information displayed as
        annotations. Useful for visualizing model performance on test set.

        Args:
            images: List of ORIGINAL images as numpy arrays [H, W, C] or [H, W].
                For best visualization, use original images before transformation.
                Use `load_original_images()` or `load_classification_original_images()`
                helpers to load images from file paths.
            predictions: Dict mapping label names to predicted values [N] or [N, C].
            targets: Dict mapping label names to ground truth values [N] or [N, C].
            metadata: Optional list of metadata dicts per sample.
            num_samples: Maximum number of samples to show.
            filename: Output filename.
            log_to_wandb: Override default wandb logging setting.

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
            vertical_spacing=0.12,
        )

        labels = list(predictions.keys())

        for i in range(n_samples):
            row = i // n_cols + 1
            col = i % n_cols + 1

            img = images[i]
            h, w = img.shape[:2]

            # Convert to RGB if grayscale
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            # Add image
            fig.add_trace(go.Image(z=img), row=row, col=col)

            # Build prediction summary
            pred_lines: list[str] = []
            gt_lines: list[str] = []
            n_correct = 0
            n_total = len(labels)

            for label in labels:
                pred = predictions[label][i]
                gt = targets[label][i]

                pred_val, gt_val = _extract_prediction_value(pred, gt)

                if pred_val == gt_val:
                    n_correct += 1

                display_name = LABEL_DISPLAY_NAMES.get(label, label)[:3]  # Abbreviate
                pred_lines.append(f"{display_name}:{pred_val}")
                gt_lines.append(f"{display_name}:{gt_val}")

            accuracy = n_correct / n_total if n_total > 0 else 0
            acc_color = "green" if accuracy >= 0.8 else ("orange" if accuracy >= 0.5 else "red")

            # Add text annotations for predictions
            y_offset = h - 10
            pred_text = " ".join(pred_lines[:4])
            gt_text = " ".join(gt_lines[:4])

            fig.add_annotation(
                x=5,
                y=15,
                text=f"<b>Pred:</b> {pred_text}",
                showarrow=False,
                font=dict(size=10, color="white"),
                bgcolor="rgba(0,0,0,0.7)",
                xanchor="left",
                row=row,
                col=col,
            )

            fig.add_annotation(
                x=5,
                y=y_offset,
                text=f"<b>GT:</b> {gt_text}",
                showarrow=False,
                font=dict(size=10, color="white"),
                bgcolor="rgba(0,0,0,0.7)",
                xanchor="left",
                row=row,
                col=col,
            )

            # Build title
            title_parts = []
            if metadata and i < len(metadata):
                level = metadata[i].get("level", "")
                patient = metadata[i].get("patient_id", "")
                if level:
                    title_parts.append(level)
                if patient:
                    title_parts.append(f"({patient[:8]})")  # Truncate patient ID

            title_parts.append(f"Acc: {accuracy:.0%}")
            title = " ".join(title_parts)

            fig.update_xaxes(
                title_text=f"<span style='color:{acc_color}'><b>{title}</b></span>",
                showticklabels=False,
                row=row,
                col=col,
            )
            fig.update_yaxes(showticklabels=False, row=row, col=col)

            # Add colored border based on accuracy
            fig.add_shape(
                type="rect",
                x0=0,
                y0=0,
                x1=w,
                y1=h,
                line=dict(color=acc_color, width=3),
                row=row,
                col=col,
            )

        fig.update_layout(
            title=f"Test Samples with Labels ({n_samples} samples)",
            height=300 * n_rows,
            width=280 * n_cols,
            showlegend=False,
        )

        self._save_figure(fig, filename, log_to_wandb)

        # Log to wandb
        if self._should_log_to_wandb(log_to_wandb) and self._wandb is not None:
            self._log_classification_images_to_wandb(
                images[:n_samples],
                {k: v[:n_samples] for k, v in predictions.items()},
                {k: v[:n_samples] for k, v in targets.items()},
                metadata[:n_samples] if metadata else None,
            )

        return fig

    def plot_confusion_examples(
        self,
        images: list[np.ndarray],
        predictions: dict[str, np.ndarray],
        targets: dict[str, np.ndarray],
        metadata: list[dict[str, Any]] | None = None,
        target_label: str = "pfirrmann",
        num_samples_per_category: int = 4,
        filename: str | None = None,
        log_to_wandb: bool | None = None,
    ) -> go.Figure:
        """Plot TP, TN, FP, FN examples for binary or multiclass classification.

        Creates a grid showing examples stratified by confusion matrix categories:
        - For binary: TP, TN, FP, FN (4 rows)
        - For multiclass: One row per unique (GT_class, is_correct) combination

        Ensures balanced representation of all confusion categories to give
        a comprehensive view of model performance.

        Args:
            images: List of ORIGINAL images as numpy arrays [H, W, C] or [H, W].
                These should be the original images BEFORE any transformation
                (normalization, augmentation, etc.) for proper visualization.
                Use `load_original_images()` helper or pass untransformed images.
            predictions: Dict mapping label names to predicted values [N] or [N, C].
            targets: Dict mapping label names to ground truth values [N] or [N, C].
            metadata: Optional list of metadata dicts per sample.
            target_label: Which label to analyze (e.g., "pfirrmann", "herniation").
            num_samples_per_category: Max samples to show per category.
            filename: Output filename. Defaults to "confusion_examples_{target_label}".
            log_to_wandb: Override default wandb logging setting.

        Returns:
            Plotly Figure with stratified confusion examples.
        """
        if target_label not in predictions:
            logger.warning(f"Label '{target_label}' not found in predictions")
            return go.Figure()

        pred_arr = np.atleast_1d(predictions[target_label])
        gt_arr = np.atleast_1d(targets[target_label])

        # Convert to class indices
        n_samples = len(images)
        pred_classes = np.zeros(n_samples, dtype=int)
        gt_classes = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            pred_classes[i], gt_classes[i] = _extract_prediction_value(
                pred_arr[i] if pred_arr.ndim > 1 else pred_arr[i],
                gt_arr[i] if gt_arr.ndim > 1 else gt_arr[i],
            )

        # Determine if binary or multiclass
        unique_classes = np.unique(np.concatenate([pred_classes, gt_classes]))
        is_binary = len(unique_classes) <= 2

        # Build categories list with masks
        categories: list[tuple[str, np.ndarray, str]] = []

        if is_binary:
            # Binary: standard TP, TN, FP, FN
            tp_mask = (pred_classes == 1) & (gt_classes == 1)
            tn_mask = (pred_classes == 0) & (gt_classes == 0)
            fp_mask = (pred_classes == 1) & (gt_classes == 0)
            fn_mask = (pred_classes == 0) & (gt_classes == 1)

            categories = [
                ("TP (Pred=1, GT=1)", tp_mask, "#2ecc71"),  # Green
                ("TN (Pred=0, GT=0)", tn_mask, "#27ae60"),  # Dark green
                ("FP (Pred=1, GT=0)", fp_mask, "#e74c3c"),  # Red
                ("FN (Pred=0, GT=1)", fn_mask, "#c0392b"),  # Dark red
            ]
        else:
            # Multiclass: create categories for each GT class (correct + incorrect)
            # This ensures we see examples for each ground truth class
            colors_correct = ["#2ecc71", "#27ae60", "#1abc9c", "#16a085", "#3498db"]
            colors_incorrect = ["#e74c3c", "#c0392b", "#e67e22", "#d35400", "#9b59b6"]

            for i, cls in enumerate(sorted(unique_classes)):
                # Correct predictions for this class
                correct_mask = (gt_classes == cls) & (pred_classes == cls)
                if correct_mask.sum() > 0:
                    color_idx = i % len(colors_correct)
                    categories.append(
                        (f"GT={cls} Correct", correct_mask, colors_correct[color_idx])
                    )

                # Incorrect predictions for this class (misclassified)
                incorrect_mask = (gt_classes == cls) & (pred_classes != cls)
                if incorrect_mask.sum() > 0:
                    color_idx = i % len(colors_incorrect)
                    categories.append(
                        (f"GT={cls} Wrong", incorrect_mask, colors_incorrect[color_idx])
                    )

        # Filter out empty categories
        categories = [(name, mask, color) for name, mask, color in categories if mask.sum() > 0]
        n_rows = len(categories)

        if n_rows == 0:
            logger.warning(f"No samples found for any category in label '{target_label}'")
            return go.Figure()

        n_cols = num_samples_per_category

        # Build subplot titles
        subplot_titles = []
        for cat_name, mask, _ in categories:
            count = mask.sum()
            subplot_titles.extend([f"{cat_name} ({count} total)"] + [""] * (n_cols - 1))

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.02,
            vertical_spacing=0.08,
        )

        for row_idx, (cat_name, mask, border_color) in enumerate(categories):
            indices = np.where(mask)[0]
            np.random.shuffle(indices)
            selected = indices[:n_cols]

            for col_idx, sample_idx in enumerate(selected):
                row = row_idx + 1
                col = col_idx + 1

                img = images[sample_idx]
                h, w = img.shape[:2]

                # Convert to RGB if grayscale
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)

                fig.add_trace(go.Image(z=img), row=row, col=col)

                # Build annotation
                pred_val = pred_classes[sample_idx]
                gt_val = gt_classes[sample_idx]

                # Add metadata info if available
                meta_text = ""
                if metadata and sample_idx < len(metadata):
                    level = metadata[sample_idx].get("level", "")
                    if level:
                        meta_text = f"{level} | "

                annotation = f"{meta_text}Pred: {pred_val} | GT: {gt_val}"

                fig.add_annotation(
                    x=w // 2,
                    y=h - 10,
                    text=annotation,
                    showarrow=False,
                    font=dict(size=10, color="white"),
                    bgcolor="rgba(0,0,0,0.7)",
                    xanchor="center",
                    row=row,
                    col=col,
                )

                # Add colored border
                fig.add_shape(
                    type="rect",
                    x0=0,
                    y0=0,
                    x1=w,
                    y1=h,
                    line=dict(color=border_color, width=4),
                    row=row,
                    col=col,
                )

                fig.update_xaxes(showticklabels=False, row=row, col=col)
                fig.update_yaxes(showticklabels=False, row=row, col=col)

            # Fill remaining columns with empty
            for col_idx in range(len(selected), n_cols):
                fig.update_xaxes(visible=False, row=row_idx + 1, col=col_idx + 1)
                fig.update_yaxes(visible=False, row=row_idx + 1, col=col_idx + 1)

        display_name = LABEL_DISPLAY_NAMES.get(target_label, target_label)
        fig.update_layout(
            title=f"Confusion Examples for {display_name}",
            height=280 * n_rows,
            width=280 * n_cols,
            showlegend=False,
        )

        output_filename = filename or f"confusion_examples_{target_label}"
        self._save_figure(fig, output_filename, log_to_wandb)

        # Log summary stats to wandb
        if self._should_log_to_wandb(log_to_wandb) and self._wandb is not None:
            # Compute binary-style stats for wandb logging
            if is_binary:
                tp_mask = (pred_classes == 1) & (gt_classes == 1)
                tn_mask = (pred_classes == 0) & (gt_classes == 0)
                fp_mask = (pred_classes == 1) & (gt_classes == 0)
                fn_mask = (pred_classes == 0) & (gt_classes == 1)
            else:
                tp_mask = pred_classes == gt_classes
                tn_mask = np.zeros(n_samples, dtype=bool)
                fp_mask = pred_classes != gt_classes
                fn_mask = np.zeros(n_samples, dtype=bool)

            stats: dict[str, int | float] = {
                f"confusion/{target_label}_tp": int(tp_mask.sum()),
                f"confusion/{target_label}_tn": int(tn_mask.sum()),
                f"confusion/{target_label}_fp": int(fp_mask.sum()),
                f"confusion/{target_label}_fn": int(fn_mask.sum()),
            }
            if tp_mask.sum() + fp_mask.sum() > 0:
                stats[f"confusion/{target_label}_precision"] = float(
                    tp_mask.sum() / (tp_mask.sum() + fp_mask.sum())
                )
            if tp_mask.sum() + fn_mask.sum() > 0:
                stats[f"confusion/{target_label}_recall"] = float(
                    tp_mask.sum() / (tp_mask.sum() + fn_mask.sum())
                )
            self._wandb.log(stats)

        return fig

    def plot_confusion_examples_all_labels(
        self,
        images: list[np.ndarray],
        predictions: dict[str, np.ndarray],
        targets: dict[str, np.ndarray],
        metadata: list[dict[str, Any]] | None = None,
        target_labels: list[str] | None = None,
        num_samples_per_category: int = 4,
        filename_prefix: str = "confusion_examples",
        log_to_wandb: bool | None = None,
    ) -> dict[str, go.Figure]:
        """Plot confusion examples for multiple labels.

        Generates one figure per label, each showing TP/TN/FP/FN examples.

        Args:
            images: List of images as numpy arrays [H, W, C] or [H, W].
            predictions: Dict mapping label names to predicted values.
            targets: Dict mapping label names to ground truth values.
            metadata: Optional list of metadata dicts per sample.
            target_labels: Labels to visualize. If None, uses all in predictions.
            num_samples_per_category: Max samples per TP/TN/FP/FN category.
            filename_prefix: Prefix for output filenames.
            log_to_wandb: Override default wandb logging setting.

        Returns:
            Dictionary mapping label names to their Plotly figures.
        """
        labels = target_labels or list(predictions.keys())
        figures: dict[str, go.Figure] = {}

        for label in labels:
            if label not in predictions:
                logger.warning(f"Label '{label}' not found in predictions, skipping")
                continue

            fig = self.plot_confusion_examples(
                images=images,
                predictions=predictions,
                targets=targets,
                metadata=metadata,
                target_label=label,
                num_samples_per_category=num_samples_per_category,
                filename=f"{filename_prefix}_{label}",
                log_to_wandb=log_to_wandb,
            )
            figures[label] = fig

        return figures

    def plot_confusion_summary(
        self,
        predictions: dict[str, np.ndarray],
        targets: dict[str, np.ndarray],
        target_labels: list[str] | None = None,
        filename: str = "confusion_summary",
        log_to_wandb: bool | None = None,
    ) -> go.Figure:
        """Plot summary of TP/TN/FP/FN counts across all labels.

        Creates a grouped bar chart showing the distribution of prediction
        outcomes for each label.

        Args:
            predictions: Dict mapping label names to predicted values.
            targets: Dict mapping label names to ground truth values.
            target_labels: Labels to include. If None, uses all in predictions.
            filename: Output filename.
            log_to_wandb: Override default wandb logging setting.

        Returns:
            Plotly Figure with stacked/grouped bars.
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
                pred_classes[i], gt_classes[i] = _extract_prediction_value(
                    pred_arr[i] if pred_arr.ndim > 1 else pred_arr[i],
                    gt_arr[i] if gt_arr.ndim > 1 else gt_arr[i],
                )

            # Determine binary vs multiclass
            unique_classes = np.unique(np.concatenate([pred_classes, gt_classes]))
            is_binary = len(unique_classes) <= 2

            if is_binary:
                tp = int(((pred_classes == 1) & (gt_classes == 1)).sum())
                tn = int(((pred_classes == 0) & (gt_classes == 0)).sum())
                fp = int(((pred_classes == 1) & (gt_classes == 0)).sum())
                fn = int(((pred_classes == 0) & (gt_classes == 1)).sum())
            else:
                # For multiclass: correct = TP+TN, incorrect = FP+FN
                correct = int((pred_classes == gt_classes).sum())
                incorrect = int((pred_classes != gt_classes).sum())
                tp = correct
                tn = 0
                fp = incorrect
                fn = 0

            tp_counts.append(tp)
            tn_counts.append(tn)
            fp_counts.append(fp)
            fn_counts.append(fn)
            label_names.append(LABEL_DISPLAY_NAMES.get(label, label))

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                name="TP",
                x=label_names,
                y=tp_counts,
                marker_color="#2ecc71",
            )
        )
        fig.add_trace(
            go.Bar(
                name="TN",
                x=label_names,
                y=tn_counts,
                marker_color="#27ae60",
            )
        )
        fig.add_trace(
            go.Bar(
                name="FP",
                x=label_names,
                y=fp_counts,
                marker_color="#e74c3c",
            )
        )
        fig.add_trace(
            go.Bar(
                name="FN",
                x=label_names,
                y=fn_counts,
                marker_color="#c0392b",
            )
        )

        fig.update_layout(
            title="Confusion Summary by Label (TP/TN/FP/FN)",
            barmode="group",
            xaxis_title="Label",
            yaxis_title="Count",
            height=500,
            width=max(600, 100 * len(label_names)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        self._save_figure(fig, filename, log_to_wandb)

        # Log to wandb
        if self._should_log_to_wandb(log_to_wandb) and self._wandb is not None:
            for i, label in enumerate(labels):
                if label not in predictions:
                    continue
                self._wandb.log({
                    f"confusion_summary/{label}_tp": tp_counts[i],
                    f"confusion_summary/{label}_tn": tn_counts[i],
                    f"confusion_summary/{label}_fp": fp_counts[i],
                    f"confusion_summary/{label}_fn": fn_counts[i],
                })

        return fig

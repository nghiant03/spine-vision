"""Visualizer classes for training and dataset analysis.

Provides wrapper classes for visualization functionality:
- BaseVisualizer: Shared configuration and utility methods
- TrainingVisualizer: Training progress, predictions, and error analysis
- DatasetVisualizer: Dataset statistics and distributions
"""

from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger
from matplotlib.figure import Figure
from PIL import Image

from spine_vision.core.tasks import get_task_display_name
from spine_vision.training.datasets.classification import ClassificationDataset
from spine_vision.visualization.base import extract_prediction_value, save_figure
from spine_vision.visualization.classification import (
    plot_classification_metrics,
    plot_classification_predictions,
    plot_confusion_examples,
    plot_confusion_matrix_with_samples,
    plot_confusion_summary,
    plot_label_distribution,
    plot_test_samples_with_labels,
)
from spine_vision.visualization.dataset import (
    plot_binary_label_distributions,
    plot_dataset_statistics,
    plot_label_cooccurrence,
    plot_pfirrmann_by_level,
    plot_samples_per_class,
)
from spine_vision.visualization.localization import (
    plot_error_distribution,
    plot_localization_predictions,
    plot_per_level_metrics,
    visualize_sample,
)
from spine_vision.visualization.training import plot_training_curves


class BaseVisualizer:
    """Base class for all visualizers.

    Provides shared configuration and utility methods for visualization classes.
    """

    output_path: Path | None
    output_mode: Literal["browser", "html", "image"]

    def __init__(
        self,
        output_path: Path | None = None,
        output_mode: Literal["browser", "html", "image"] = "image",
    ) -> None:
        """Initialize base visualizer.

        Args:
            output_path: Directory for saving visualizations.
            output_mode: Output format.
        """
        self.output_path = output_path
        self.output_mode = output_mode

        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)

    def _save_figure(
        self,
        fig: Figure,
        filename: str,
        dpi: int = 150,
    ) -> None:
        """Save figure using configured output mode."""
        save_figure(fig, self.output_path, filename, self.output_mode, dpi)


class TrainingVisualizer(BaseVisualizer):
    """Visualizer for training progress and validation results.

    Generates static plots for training curves, predictions,
    and error analysis. Optionally logs to trackio.
    """

    use_trackio: bool
    _trackio: Any

    def __init__(
        self,
        output_path: Path | None = None,
        output_mode: Literal["browser", "html", "image"] = "image",
        use_trackio: bool = False,
    ) -> None:
        """Initialize visualizer."""
        super().__init__(output_path, output_mode)
        self.use_trackio = use_trackio
        self._trackio = None

        if use_trackio:
            try:
                import trackio
                self._trackio = trackio
            except ImportError:
                logger.warning("trackio not installed. Disabling trackio logging.")
                self.use_trackio = False

    def _should_log_to_trackio(self, override: bool | None) -> bool:
        """Determine if should log to trackio."""
        if override is not None:
            return override and self._trackio is not None
        return self.use_trackio and self._trackio is not None

    def _log_figure_to_trackio(self, fig: Figure, name: str) -> None:
        """Log matplotlib figure to trackio as an image."""
        if self._trackio is None:
            return
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        pil_img = Image.open(buf)
        img = np.array(pil_img.convert("RGB"))
        buf.close()
        self._trackio.log({name: self._trackio.Image(img)})

    # === Localization methods ===

    def plot_training_curves(
        self,
        history: dict[str, list[float]],
        filename: str = "training_curves",
        log_to_trackio: bool | None = None,
    ) -> Figure:
        """Plot training loss and metrics over epochs."""
        fig = plot_training_curves(
            history=history,
            output_path=self.output_path,
            filename=filename,
            output_mode=self.output_mode,
        )
        if self._should_log_to_trackio(log_to_trackio):
            self._log_figure_to_trackio(fig, filename)
        return fig

    def plot_localization_predictions(
        self,
        images: list[np.ndarray],
        predictions: np.ndarray,
        targets: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
        num_samples: int = 16,
        filename: str = "localization_predictions",
        log_to_trackio: bool | None = None,
    ) -> Figure:
        """Plot localization predictions overlaid on images."""
        fig = plot_localization_predictions(
            images=images,
            predictions=predictions,
            targets=targets,
            metadata=metadata,
            num_samples=num_samples,
            output_path=self.output_path,
            filename=filename,
            output_mode=self.output_mode,
        )
        if self._should_log_to_trackio(log_to_trackio) and self._trackio is not None:
            self._log_prediction_images_to_trackio(
                images[:num_samples],
                predictions[:num_samples],
                targets[:num_samples],
                metadata[:num_samples] if metadata else None,
            )
        return fig

    def _log_prediction_images_to_trackio(
        self,
        images: list[np.ndarray],
        predictions: np.ndarray,
        targets: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Log prediction images to trackio."""
        if self._trackio is None:
            return

        trackio_images = []
        for i, img in enumerate(images):
            h, w = img.shape[:2]
            pred_x, pred_y = predictions[i] * [w, h]
            gt_x, gt_y = targets[i] * [w, h]

            caption = ""
            if metadata and i < len(metadata):
                caption = metadata[i].get("level", "")

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

            trackio_img = self._trackio.Image(
                img,
                caption=caption,
                boxes={"predictions": {"box_data": boxes, "class_labels": {0: "GT", 1: "Pred"}}},
            )
            trackio_images.append(trackio_img)

        self._trackio.log({"predictions": trackio_images})

    def plot_error_distribution(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        levels: np.ndarray | None = None,
        level_names: list[str] | None = None,
        filename: str = "error_distribution",
        log_to_trackio: bool | None = None,
    ) -> Figure:
        """Plot error distribution analysis."""
        fig = plot_error_distribution(
            predictions=predictions,
            targets=targets,
            levels=levels,
            level_names=level_names,
            output_path=self.output_path,
            filename=filename,
            output_mode=self.output_mode,
        )
        if self._should_log_to_trackio(log_to_trackio) and self._trackio is not None:
            distances = np.sqrt(np.sum((predictions - targets) ** 2, axis=1))
            self._trackio.log({
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
        log_to_trackio: bool | None = None,
    ) -> Figure:
        """Plot per-level metric comparison."""
        fig = plot_per_level_metrics(
            metrics=metrics,
            level_names=level_names,
            metric_prefix=metric_prefix,
            output_path=self.output_path,
            filename=filename,
            output_mode=self.output_mode,
        )
        if self._should_log_to_trackio(log_to_trackio) and self._trackio is not None:
            values = []
            labels = []
            for level in level_names:
                key = f"{metric_prefix}{level}"
                if key in metrics:
                    values.append(metrics[key])
                    labels.append(level)
            trackio_metrics = {f"per_level/{label}": val for label, val in zip(labels, values)}
            if values:
                trackio_metrics["per_level/average"] = float(np.mean(values))
            self._trackio.log(trackio_metrics)
        return fig

    def visualize_sample(
        self,
        image: np.ndarray | Image.Image,
        prediction: np.ndarray,
        target: np.ndarray,
        level: str = "",
        filename: str = "sample",
        log_to_trackio: bool | None = None,
    ) -> Figure:
        """Visualize a single sample with prediction overlay."""
        fig = visualize_sample(
            image=image,
            prediction=prediction,
            target=target,
            level=level,
            output_path=self.output_path,
            filename=filename,
            output_mode=self.output_mode,
        )
        if self._should_log_to_trackio(log_to_trackio):
            self._log_figure_to_trackio(fig, filename)
        return fig

    def log_table(
        self,
        data: dict[str, list[Any]],
        table_name: str = "results",
    ) -> None:
        """Log a table to trackio."""
        if self._trackio is not None and self.use_trackio:
            table = self._trackio.Table(columns=list(data.keys()))
            n_rows = len(next(iter(data.values())))
            for i in range(n_rows):
                row = [data[col][i] for col in data.keys()]
                table.add_data(*row)
            self._trackio.log({table_name: table})

    # === Classification methods ===

    def plot_classification_predictions(
        self,
        images: list[np.ndarray],
        predictions: dict[str, np.ndarray],
        targets: dict[str, np.ndarray],
        metadata: list[dict[str, Any]] | None = None,
        num_samples: int = 16,
        filename: str = "classification_predictions",
        log_to_trackio: bool | None = None,
    ) -> Figure:
        """Plot classification predictions overlaid on images."""
        fig = plot_classification_predictions(
            images=images,
            predictions=predictions,
            targets=targets,
            metadata=metadata,
            num_samples=num_samples,
            output_path=self.output_path,
            filename=filename,
            output_mode=self.output_mode,
        )
        if self._should_log_to_trackio(log_to_trackio) and self._trackio is not None:
            self._log_classification_images_to_trackio(
                images[:num_samples],
                {k: v[:num_samples] for k, v in predictions.items()},
                {k: v[:num_samples] for k, v in targets.items()},
                metadata[:num_samples] if metadata else None,
            )
        return fig

    def _log_classification_images_to_trackio(
        self,
        images: list[np.ndarray],
        predictions: dict[str, np.ndarray],
        targets: dict[str, np.ndarray],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Log classification prediction images to trackio."""
        if self._trackio is None:
            return

        trackio_images = []
        labels = list(predictions.keys())

        for i, img in enumerate(images):
            caption_parts = []
            if metadata and i < len(metadata):
                level = metadata[i].get("level", "")
                if level:
                    caption_parts.append(f"Level: {level}")

            for label in labels:
                pred_val, gt_val = extract_prediction_value(predictions[label][i], targets[label][i])
                display_name = get_task_display_name(label)
                status = "\u2713" if pred_val == gt_val else "\u2717"
                caption_parts.append(f"{display_name}: {pred_val}/{gt_val} {status}")

            caption = " | ".join(caption_parts)
            trackio_img = self._trackio.Image(img, caption=caption)
            trackio_images.append(trackio_img)

        self._trackio.log({"classification_predictions": trackio_images})

    def plot_classification_metrics(
        self,
        metrics: dict[str, float],
        target_labels: list[str] | None = None,
        filename: str = "classification_metrics",
        log_to_trackio: bool | None = None,
    ) -> Figure:
        """Plot per-label classification metrics."""
        fig = plot_classification_metrics(
            metrics=metrics,
            target_labels=target_labels,
            output_path=self.output_path,
            filename=filename,
            output_mode=self.output_mode,
        )
        if self._should_log_to_trackio(log_to_trackio) and self._trackio is not None:
            self._trackio.log({f"metrics/{k}": v for k, v in metrics.items()})
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
        log_to_trackio: bool | None = None,
    ) -> Figure:
        """Plot confusion matrix with sample images from each cell."""
        fig = plot_confusion_matrix_with_samples(
            images=images,
            predictions=predictions,
            targets=targets,
            target_label=target_label,
            metadata=metadata,
            class_names=class_names,
            max_samples_per_cell=max_samples_per_cell,
            output_path=self.output_path,
            filename=filename,
            output_mode=self.output_mode,
        )
        if self._should_log_to_trackio(log_to_trackio) and self._trackio is not None:
            output_filename = filename or f"confusion_matrix_samples_{target_label}"
            self._log_figure_to_trackio(fig, output_filename)
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
        log_to_trackio: bool | None = None,
    ) -> dict[str, Figure]:
        """Plot confusion matrices with samples for multiple labels."""
        labels = target_labels or list(predictions.keys())
        figures: dict[str, Figure] = {}

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
                log_to_trackio=log_to_trackio,
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
        log_to_trackio: bool | None = None,
    ) -> Figure:
        """Plot test samples with predicted and ground truth labels."""
        fig = plot_test_samples_with_labels(
            images=images,
            predictions=predictions,
            targets=targets,
            metadata=metadata,
            num_samples=num_samples,
            output_path=self.output_path,
            filename=filename,
            output_mode=self.output_mode,
        )
        if self._should_log_to_trackio(log_to_trackio) and self._trackio is not None:
            self._log_classification_images_to_trackio(
                images[:num_samples],
                {k: v[:num_samples] for k, v in predictions.items()},
                {k: v[:num_samples] for k, v in targets.items()},
                metadata[:num_samples] if metadata else None,
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
        log_to_trackio: bool | None = None,
    ) -> Figure:
        """Plot TP, TN, FP, FN examples for classification."""
        fig = plot_confusion_examples(
            images=images,
            predictions=predictions,
            targets=targets,
            metadata=metadata,
            target_label=target_label,
            num_samples_per_category=num_samples_per_category,
            output_path=self.output_path,
            filename=filename,
            output_mode=self.output_mode,
        )
        if self._should_log_to_trackio(log_to_trackio):
            output_filename = filename or f"confusion_examples_{target_label}"
            self._log_figure_to_trackio(fig, output_filename)
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
        log_to_trackio: bool | None = None,
    ) -> dict[str, Figure]:
        """Plot confusion examples for multiple labels."""
        labels = target_labels or list(predictions.keys())
        figures: dict[str, Figure] = {}

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
                log_to_trackio=log_to_trackio,
            )
            figures[label] = fig

        return figures

    def plot_confusion_summary(
        self,
        predictions: dict[str, np.ndarray],
        targets: dict[str, np.ndarray],
        target_labels: list[str] | None = None,
        filename: str = "confusion_summary",
        log_to_trackio: bool | None = None,
    ) -> Figure:
        """Plot summary of TP/TN/FP/FN counts across all labels."""
        fig = plot_confusion_summary(
            predictions=predictions,
            targets=targets,
            target_labels=target_labels,
            output_path=self.output_path,
            filename=filename,
            output_mode=self.output_mode,
        )
        if self._should_log_to_trackio(log_to_trackio):
            self._log_figure_to_trackio(fig, filename)
        return fig

    def plot_label_distribution(
        self,
        distributions: dict[str, dict[str, dict[int | str, int]]],
        target_labels: list[str] | None = None,
        filename: str = "label_distribution",
        log_to_trackio: bool | None = None,
    ) -> Figure:
        """Plot label distribution across train/val/test splits."""
        fig = plot_label_distribution(
            distributions=distributions,
            target_labels=target_labels,
            output_path=self.output_path,
            filename=filename,
            output_mode=self.output_mode,
        )
        if self._should_log_to_trackio(log_to_trackio) and self._trackio is not None:
            for split, split_dist in distributions.items():
                for label, class_counts in split_dist.items():
                    total = sum(class_counts.values())
                    for cls, count in class_counts.items():
                        self._trackio.log({
                            f"distribution/{split}/{label}/class_{cls}": count,
                            f"distribution/{split}/{label}/class_{cls}_pct": (
                                count / total * 100 if total > 0 else 0
                            ),
                        })
        return fig


class DatasetVisualizer(BaseVisualizer):
    """Visualizer for dataset statistics and analysis.

    Provides seaborn/matplotlib-based visualizations for dataset exploration.
    """

    def plot_dataset_statistics(self, dataset: ClassificationDataset) -> Figure:
        """Plot overall dataset statistics."""
        return plot_dataset_statistics(dataset, self.output_path, self.output_mode)

    def plot_binary_label_distributions(self, dataset: ClassificationDataset) -> Figure:
        """Plot distributions for all binary labels."""
        return plot_binary_label_distributions(
            dataset, self.output_path, self.output_mode
        )

    def plot_label_cooccurrence(self, dataset: ClassificationDataset) -> Figure:
        """Plot co-occurrence heatmap between binary labels."""
        return plot_label_cooccurrence(dataset, self.output_path, self.output_mode)

    def plot_pfirrmann_by_level(self, dataset: ClassificationDataset) -> Figure:
        """Plot Pfirrmann grade distribution by IVD level."""
        return plot_pfirrmann_by_level(dataset, self.output_path, self.output_mode)

    def plot_samples_per_class(
        self,
        dataset: ClassificationDataset,
        data_path: Path,
        samples_per_class: int = 4,
        display_size: tuple[int, int] = (128, 128),
    ) -> dict[str, Figure]:
        """Plot sample images for each possible value of each label."""
        return plot_samples_per_class(
            dataset,
            data_path,
            self.output_path,
            self.output_mode,
            samples_per_class,
            display_size,
        )

    def generate_all(
        self,
        dataset: ClassificationDataset,
        data_path: Path,
        samples_per_class: int = 4,
        display_size: tuple[int, int] = (128, 128),
    ) -> None:
        """Generate all dataset visualizations."""
        logger.info("Generating dataset statistics...")
        self.plot_dataset_statistics(dataset)

        logger.info("Generating binary label distributions...")
        self.plot_binary_label_distributions(dataset)

        logger.info("Generating label co-occurrence heatmap...")
        self.plot_label_cooccurrence(dataset)

        logger.info("Generating Pfirrmann by level distribution...")
        self.plot_pfirrmann_by_level(dataset)

        logger.info("Generating samples per class for each label...")
        self.plot_samples_per_class(dataset, data_path, samples_per_class, display_size)

        if self.output_path:
            logger.info(f"All visualizations saved to {self.output_path}")

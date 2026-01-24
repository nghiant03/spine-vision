"""Visualization utilities for training and dataset analysis.

Provides visualization tools for:
- Training curves (loss, metrics)
- Localization predictions with ground truth overlay
- Classification predictions with label overlay
- Error distribution analysis
- Per-level performance breakdown
- Test sample visualization with labels
- Classification confusion analysis (TP/TN/FP/FN examples)
- Dataset statistics and distributions

Uses matplotlib and seaborn for static visualizations.
Supports optional trackio logging for experiment tracking.
"""

from spine_vision.datasets.labels import LABEL_COLORS, LABEL_DISPLAY_NAMES
from spine_vision.visualization.base import (
    load_classification_original_images,
    load_original_images,
    save_figure,
)
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
from spine_vision.visualization.visualizer import (
    BaseVisualizer,
    DatasetVisualizer,
    TrainingVisualizer,
)

__all__ = [
    # Constants
    "LABEL_DISPLAY_NAMES",
    "LABEL_COLORS",
    # Base utilities
    "load_original_images",
    "load_classification_original_images",
    "save_figure",
    # Training curves
    "plot_training_curves",
    # Localization
    "plot_localization_predictions",
    "plot_error_distribution",
    "plot_per_level_metrics",
    "visualize_sample",
    # Classification
    "plot_classification_predictions",
    "plot_classification_metrics",
    "plot_confusion_matrix_with_samples",
    "plot_test_samples_with_labels",
    "plot_confusion_examples",
    "plot_confusion_summary",
    "plot_label_distribution",
    # Dataset visualization
    "plot_dataset_statistics",
    "plot_binary_label_distributions",
    "plot_label_cooccurrence",
    "plot_pfirrmann_by_level",
    "plot_samples_per_class",
    # Visualizer classes
    "BaseVisualizer",
    "TrainingVisualizer",
    "DatasetVisualizer",
]

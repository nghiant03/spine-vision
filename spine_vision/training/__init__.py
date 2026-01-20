"""Training module for spine-vision models.

This module provides extensible training infrastructure for various tasks:
- Localization (landmark detection, coordinate regression)
- Classification (single-task and multi-task lumbar spine grading)
- Segmentation

Uses HuggingFace Accelerate for distributed training and mixed precision,
with optional trackio logging for experiment tracking.

Key Features:
- Model/Trainer registries for dynamic discovery
- Configurable backbone via BackboneFactory
- Configurable head architectures via HeadConfig
- Training hooks for custom behavior without overriding train loop
- Stateful metrics for consistent evaluation

Exports:
    - Base classes: BaseTrainer, BaseModel, TrainingConfig
    - Registries: ModelRegistry, TrainerRegistry, MetricsRegistry
    - Heads: HeadConfig, HeadFactory, create_head
    - Datasets: LocalizationDataset, ClassificationDataset
    - Models: Classifier, CoordinateRegressor
    - Trainers: LocalizationTrainer, LocalizationConfig, ClassificationTrainer, ClassificationConfig
    - Metrics: LocalizationMetrics, MTLClassificationMetrics
    - Visualization: TrainingVisualizer
"""

from spine_vision.training.base import (
    BaseModel,
    BaseTrainer,
    TrainingConfig,
    TrainingResult,
)
from spine_vision.training.datasets import (
    ClassificationDataset,
    LocalizationDataset,
)
from spine_vision.training.heads import (
    BaseHead,
    HeadConfig,
    HeadFactory,
    MLPHead,
    MultiTaskHead,
    create_head,
)
from spine_vision.training.metrics import (
    BaseMetrics,
    ClassificationMetrics,
    LocalizationMetrics,
    MTLClassificationMetrics,
)
from spine_vision.training.models import (
    BACKBONES,
    BackboneFactory,
    Classifier,
    CoordinateRegressor,
    LUMBAR_SPINE_TASKS,
    MTLTargets,
    MultiTaskClassifier,  # Backward compatibility alias
    TaskConfig,
    list_backbones,
)
from spine_vision.training.registry import (
    MetricsRegistry,
    ModelRegistry,
    TrainerRegistry,
    register_metrics,
    register_model,
    register_trainer,
)
from spine_vision.training.trainers import (
    ClassificationConfig,
    ClassificationTrainer,
    LocalizationConfig,
    LocalizationTrainer,
)
from spine_vision.visualization import (
    LABEL_COLORS,
    LABEL_DISPLAY_NAMES,
    DatasetVisualizer,
    TrainingVisualizer,
    load_classification_original_images,
    load_original_images,
)

__all__ = [
    # Base classes
    "BaseModel",
    "BaseTrainer",
    "TrainingConfig",
    "TrainingResult",
    # Registries
    "ModelRegistry",
    "TrainerRegistry",
    "MetricsRegistry",
    "register_model",
    "register_trainer",
    "register_metrics",
    # Heads
    "HeadConfig",
    "HeadFactory",
    "BaseHead",
    "MLPHead",
    "MultiTaskHead",
    "create_head",
    # Datasets
    "LocalizationDataset",
    "ClassificationDataset",
    # Models
    "Classifier",
    "MultiTaskClassifier",  # Backward compatibility alias
    "CoordinateRegressor",
    "TaskConfig",
    "LUMBAR_SPINE_TASKS",
    "MTLTargets",
    "BackboneFactory",
    "BACKBONES",
    "list_backbones",
    # Trainers
    "LocalizationConfig",
    "LocalizationTrainer",
    "ClassificationConfig",
    "ClassificationTrainer",
    # Metrics
    "BaseMetrics",
    "LocalizationMetrics",
    "ClassificationMetrics",
    "MTLClassificationMetrics",
    # Visualization
    "TrainingVisualizer",
    "DatasetVisualizer",
    "LABEL_DISPLAY_NAMES",
    "LABEL_COLORS",
    "load_original_images",
    "load_classification_original_images",
]

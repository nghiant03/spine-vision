"""Training module for spine-vision models.

This module provides extensible training infrastructure for various tasks:
- Localization (landmark detection, coordinate regression)
- Classification (multi-task lumbar spine grading)
- Segmentation

Uses HuggingFace Accelerate for distributed training and mixed precision,
with optional wandb logging for experiment tracking.

Exports:
    - Base classes: BaseTrainer, BaseModel, TrainingConfig
    - Datasets: IVDCoordsDataset, ClassificationDataset
    - Models: ConvNextLocalization, ConvNextClassifier, VisionTransformerLocalization, ResNet50MTL
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
    IVDCoordsDataset,
)
from spine_vision.training.metrics import LocalizationMetrics, MTLClassificationMetrics
from spine_vision.training.models import (
    ConvNextClassifier,
    ConvNextLocalization,
    MTLPredictions,
    MTLTargets,
    ResNet50MTL,
    VisionTransformerLocalization,
)
from spine_vision.training.trainers import (
    ClassificationConfig,
    ClassificationTrainer,
    LocalizationConfig,
    LocalizationTrainer,
)
from spine_vision.training.visualization import TrainingVisualizer

__all__ = [
    # Base classes
    "BaseModel",
    "BaseTrainer",
    "TrainingConfig",
    "TrainingResult",
    # Datasets
    "IVDCoordsDataset",
    "ClassificationDataset",
    # Models
    "ConvNextLocalization",
    "ConvNextClassifier",
    "VisionTransformerLocalization",
    "ResNet50MTL",
    "MTLPredictions",
    "MTLTargets",
    # Trainers
    "LocalizationConfig",
    "LocalizationTrainer",
    "ClassificationConfig",
    "ClassificationTrainer",
    # Metrics
    "LocalizationMetrics",
    "MTLClassificationMetrics",
    # Visualization
    "TrainingVisualizer",
]

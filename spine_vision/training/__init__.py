"""Training module for spine-vision models.

This module provides extensible training infrastructure for various tasks:
- Localization (landmark detection, coordinate regression)
- Classification
- Segmentation

Exports:
    - Base classes: BaseTrainer, BaseModel, TrainingConfig
    - Datasets: IVDCoordsDataset
    - Models: ConvNextLocalization
    - Trainers: LocalizationTrainer, LocalizationConfig
    - Metrics: LocalizationMetrics
    - Visualization: TrainingVisualizer
"""

from spine_vision.training.base import (
    BaseModel,
    BaseTrainer,
    TrainingConfig,
    TrainingResult,
)
from spine_vision.training.datasets import IVDCoordsDataset
from spine_vision.training.metrics import LocalizationMetrics
from spine_vision.training.models import ConvNextLocalization
from spine_vision.training.trainers import LocalizationConfig, LocalizationTrainer
from spine_vision.training.visualization import TrainingVisualizer

__all__ = [
    # Base classes
    "BaseModel",
    "BaseTrainer",
    "TrainingConfig",
    "TrainingResult",
    # Datasets
    "IVDCoordsDataset",
    # Models
    "ConvNextLocalization",
    # Trainers
    "LocalizationConfig",
    "LocalizationTrainer",
    # Metrics
    "LocalizationMetrics",
    # Visualization
    "TrainingVisualizer",
]

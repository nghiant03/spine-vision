"""Task-specific trainers.

Provides concrete trainer implementations for different tasks.
"""

from spine_vision.training.trainers.classification import (
    ClassificationConfig,
    ClassificationTrainer,
)
from spine_vision.training.trainers.localization import (
    LocalizationConfig,
    LocalizationTrainer,
)

__all__ = [
    "LocalizationConfig",
    "LocalizationTrainer",
    "ClassificationConfig",
    "ClassificationTrainer",
]

"""Model architectures for training.

Provides implementations of various backbone models for different tasks.
Uses timm (PyTorch Image Models) for pretrained backbones.

Available models:
- ConvNextLocalization: ConvNext-based coordinate regression
- ConvNextClassifier: ConvNext-based classification
- VisionTransformerLocalization: ViT-based coordinate regression
"""

from spine_vision.training.models.convnext import (
    ConvNextClassifier,
    ConvNextLocalization,
    VisionTransformerLocalization,
)

__all__ = [
    "ConvNextLocalization",
    "ConvNextClassifier",
    "VisionTransformerLocalization",
]

"""Model architectures for training.

Provides implementations of various backbone models for different tasks.
Uses timm (PyTorch Image Models) for pretrained backbones.

Available models:
- ConvNextLocalization: ConvNext-based coordinate regression
- ConvNextClassifier: ConvNext-based classification
- VisionTransformerLocalization: ViT-based coordinate regression
- ResNet50MTL: ResNet-50 multi-task classification for lumbar spine grading
"""

from spine_vision.training.models.convnext import (
    ConvNextClassifier,
    ConvNextLocalization,
    VisionTransformerLocalization,
)
from spine_vision.training.models.resnet_mtl import (
    MTLPredictions,
    MTLTargets,
    ResNet50MTL,
)

__all__ = [
    "ConvNextLocalization",
    "ConvNextClassifier",
    "VisionTransformerLocalization",
    "ResNet50MTL",
    "MTLPredictions",
    "MTLTargets",
]

"""Backbone registry and factory for vision models.

Provides a unified interface for creating backbone feature extractors
from various architectures via timm.

Usage:
    from spine_vision.training.models.backbone import BackboneFactory, BACKBONES

    # Create backbone
    backbone, feature_dim = BackboneFactory.create("resnet50", pretrained=True)

    # List available backbones
    print(BackboneFactory.list_backbones())

    # Get feature dimension without creating
    feature_dim = BackboneFactory.get_feature_dim("resnet50")
"""

from typing import Literal

import timm
import torch.nn as nn

# Backbone name to timm model mapping
BACKBONES: dict[str, str] = {
    # ResNet family
    "resnet18": "resnet18.a1_in1k",
    "resnet34": "resnet34.a1_in1k",
    "resnet50": "resnet50.a1_in1k",
    "resnet101": "resnet101.a1_in1k",
    "resnet152": "resnet152.a1_in1k",
    # ResNet with better training recipes
    "resnet50_a2": "resnet50.a2_in1k",
    "resnet50_b": "resnet50.b1k_in1k",
    "resnet50_c": "resnet50.c1_in1k",
    "resnet50_d": "resnet50.d_in1k",
    # ResNeXt
    "resnext50": "resnext50_32x4d.a1h_in1k",
    "resnext101": "resnext101_32x8d.fb_wsl_ig1b_ft_in1k",
    # Wide ResNet
    "wide_resnet50": "wide_resnet50_2.racm_in1k",
    "wide_resnet101": "wide_resnet101_2.tv2_in1k",
    # ResNet-RS (improved training)
    "resnetrs50": "resnetrs50.tf_in1k",
    "resnetrs101": "resnetrs101.tf_in1k",
    "resnetrs152": "resnetrs152.tf_in1k",
    # ConvNeXt
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "convnext_small": "convnext_small.fb_in22k_ft_in1k",
    "convnext_base": "convnext_base.fb_in22k_ft_in1k",
    "convnext_large": "convnext_large.fb_in22k_ft_in1k",
    "convnext_xlarge": "convnext_xlarge.fb_in22k_ft_in1k",
    # ConvNeXt V2
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "convnextv2_small": "convnextv2_small.fcmae",
    "convnextv2_base": "convnextv2_base.fcmae_ft_in22k_in1k",
    "convnextv2_large": "convnextv2_large.fcmae_ft_in22k_in1k",
    "convnextv2_huge": "convnextv2_huge.fcmae_ft_in22k_in1k",
    # Vision Transformer
    "vit_tiny": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit_base": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "vit_large": "vit_large_patch16_224.augreg_in21k_ft_in1k",
    # DeiT
    "deit_tiny": "deit3_small_patch16_224.fb_in22k_ft_in1k",
    "deit_small": "deit3_small_patch16_224.fb_in22k_ft_in1k",
    "deit_base": "deit3_base_patch16_224.fb_in22k_ft_in1k",
    # Swin Transformer
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "swin_small": "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
    "swin_base": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    # EfficientNet
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "efficientnet_b1": "efficientnet_b1.ra_in1k",
    "efficientnet_b2": "efficientnet_b2.ra_in1k",
    "efficientnet_b3": "efficientnet_b3.ra_in1k",
    "efficientnet_b4": "efficientnet_b4.ra_in1k",
    # EfficientNetV2
    "efficientnetv2_s": "efficientnetv2_s.ra_in1k",
    "efficientnetv2_m": "efficientnetv2_m.ra_in1k",
    "efficientnetv2_l": "efficientnetv2_l.ra_in1k",
    # MobileNetV3
    "mobilenetv3_small": "mobilenetv3_small_100.lamb_in1k",
    "mobilenetv3_large": "mobilenetv3_large_100.ra_in1k",
}

BackboneName = Literal[
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "resnet50_a2", "resnet50_b", "resnet50_c", "resnet50_d",
    "resnext50", "resnext101",
    "wide_resnet50", "wide_resnet101",
    "resnetrs50", "resnetrs101", "resnetrs152",
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large", "convnext_xlarge",
    "convnextv2_tiny", "convnextv2_small", "convnextv2_base", "convnextv2_large", "convnextv2_huge",
    "vit_tiny", "vit_small", "vit_base", "vit_large",
    "deit_tiny", "deit_small", "deit_base",
    "swin_tiny", "swin_small", "swin_base",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4",
    "efficientnetv2_s", "efficientnetv2_m", "efficientnetv2_l",
    "mobilenetv3_small", "mobilenetv3_large",
]


class BackboneFactory:
    """Factory for creating backbone feature extractors."""

    # Cache for feature dimensions (avoid loading model just to get dim)
    _feature_dims: dict[str, int] = {}

    @classmethod
    def create(
        cls,
        name: str,
        pretrained: bool = True,
    ) -> tuple[nn.Module, int]:
        """Create a backbone feature extractor.

        Args:
            name: Backbone name (see BACKBONES for options).
            pretrained: Use pretrained weights.

        Returns:
            Tuple of (backbone module, feature dimension).

        Raises:
            ValueError: If backbone name not found.
        """
        if name not in BACKBONES:
            available = ", ".join(sorted(BACKBONES.keys()))
            raise ValueError(f"Unknown backbone: {name}. Available: {available}")

        timm_name = BACKBONES[name]
        backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        feature_dim: int = backbone.num_features  # type: ignore[assignment]

        # Cache the feature dimension
        cls._feature_dims[name] = feature_dim

        return backbone, feature_dim

    @classmethod
    def get_feature_dim(cls, name: str) -> int:
        """Get feature dimension for a backbone without creating it.

        Args:
            name: Backbone name.

        Returns:
            Feature dimension.
        """
        if name in cls._feature_dims:
            return cls._feature_dims[name]

        # Need to create the model to get dimension
        _, feature_dim = cls.create(name, pretrained=False)
        return feature_dim

    @classmethod
    def list_backbones(cls, family: str | None = None) -> list[str]:
        """List available backbone names.

        Args:
            family: Optional filter by family (resnet, convnext, vit, etc.).

        Returns:
            List of backbone names.
        """
        if family is None:
            return sorted(BACKBONES.keys())

        return sorted(
            name for name in BACKBONES.keys()
            if name.startswith(family.lower())
        )

    @classmethod
    def get_timm_name(cls, name: str) -> str:
        """Get the timm model name for a backbone.

        Args:
            name: Backbone name.

        Returns:
            Timm model name.
        """
        if name not in BACKBONES:
            raise ValueError(f"Unknown backbone: {name}")
        return BACKBONES[name]

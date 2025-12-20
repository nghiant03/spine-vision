"""ViT-based models for localization and other tasks.

Uses timm (PyTorch Image Models) for pretrained backbones with extensive
model variants and pretrained weights.
"""

from typing import Any, Literal

import timm
import torch
import torch.nn as nn
from pydantic import BaseModel


class VisionTransformerLocalization(BaseModel):
    """Vision Transformer model for coordinate regression.

    Alternative to ConvNext using ViT architecture via timm.
    """

    VARIANTS = {
        "tiny": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        "small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
        "base": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "large": "vit_large_patch16_224.augreg_in21k_ft_in1k",
        # DeiT variants
        "deit_tiny": "deit3_small_patch16_224.fb_in22k_ft_in1k",
        "deit_small": "deit3_small_patch16_224.fb_in22k_ft_in1k",
        "deit_base": "deit3_base_patch16_224.fb_in22k_ft_in1k",
        # Swin variants
        "swin_tiny": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        "swin_small": "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
        "swin_base": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    }

    def __init__(
        self,
        variant: Literal[
            "tiny", "small", "base", "large",
            "deit_tiny", "deit_small", "deit_base",
            "swin_tiny", "swin_small", "swin_base"
        ] = "base",
        num_outputs: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        num_levels: int = 1,
        loss_type: Literal["mse", "smooth_l1", "huber"] = "smooth_l1",
        level_embedding_dim: int = 16,
    ) -> None:
        """Initialize VisionTransformerLocalization.

        Args:
            variant: ViT variant (see VARIANTS for options).
            num_outputs: Number of output coordinates (default 2 for x,y).
            pretrained: Use pretrained weights.
            dropout: Dropout rate before final layer.
            freeze_backbone: Freeze backbone weights initially.
            num_levels: Number of IVD levels (for level embedding).
            loss_type: Loss function type.
            level_embedding_dim: Dimension of level embedding.
        """
        super().__init__()

        self._variant = variant
        self._num_outputs = num_outputs

        if variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown variant: {variant}. Choose from {list(self.VARIANTS.keys())}"
            )

        model_name = self.VARIANTS[variant]
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        feature_dim: int = self.backbone.num_features  # type: ignore[assignment]

        # Level embedding (optional)
        self.level_embedding: nn.Embedding | None = None
        if num_levels > 1:
            self.level_embedding = nn.Embedding(num_levels, level_embedding_dim)
            head_input_dim = feature_dim + level_embedding_dim
        else:
            head_input_dim = feature_dim

        # Regression head
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Dropout(dropout),
            nn.Linear(head_input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_outputs),
            nn.Sigmoid(),
        )

        # Loss function
        if loss_type == "mse":
            self._loss_fn = nn.MSELoss()
        elif loss_type == "smooth_l1":
            self._loss_fn = nn.SmoothL1Loss()
        elif loss_type == "huber":
            self._loss_fn = nn.HuberLoss(delta=0.1)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        if freeze_backbone:
            self.freeze_backbone()

        self._is_initialized = True

    @property
    def name(self) -> str:
        return f"ViT-{self._variant.replace('_', '-').title()}-Localization"

    def forward(
        self,
        x: torch.Tensor,
        level_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.backbone(x)

        if self.level_embedding is not None and level_idx is not None:
            level_emb = self.level_embedding(level_idx)
            features = torch.cat([features, level_emb], dim=1)

        return self.head(features)

    def get_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._loss_fn(predictions, targets)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

"""ConvNext-based models for localization and other tasks.

ConvNext provides a modern CNN architecture that can compete with transformers
while maintaining the efficiency of convolutions.
"""

from typing import Any, Literal

import torch
import torch.nn as nn
import torchvision.models as models

from spine_vision.training.base import BaseModel


class ConvNextLocalization(BaseModel):
    """ConvNext model for coordinate regression (localization).

    Predicts (x, y) coordinates for IVD localization from spine images.
    Supports multiple ConvNext variants and optional feature freezing.
    """

    VARIANTS = {
        "tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),
        "small": (models.convnext_small, models.ConvNeXt_Small_Weights.DEFAULT),
        "base": (models.convnext_base, models.ConvNeXt_Base_Weights.DEFAULT),
        "large": (models.convnext_large, models.ConvNeXt_Large_Weights.DEFAULT),
    }

    def __init__(
        self,
        variant: Literal["tiny", "small", "base", "large"] = "base",
        num_outputs: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        num_levels: int = 1,
        loss_type: Literal["mse", "smooth_l1", "huber"] = "smooth_l1",
        level_embedding_dim: int = 16,
    ) -> None:
        """Initialize ConvNextLocalization.

        Args:
            variant: ConvNext variant ('tiny', 'small', 'base', 'large').
            num_outputs: Number of output coordinates (default 2 for x,y).
            pretrained: Use ImageNet pretrained weights.
            dropout: Dropout rate before final layer.
            freeze_backbone: Freeze backbone weights initially.
            num_levels: Number of IVD levels (for level embedding).
            loss_type: Loss function type.
            level_embedding_dim: Dimension of level embedding.
        """
        super().__init__()

        self._variant = variant
        self._num_outputs = num_outputs
        self._pretrained = pretrained
        self._dropout = dropout
        self._freeze_backbone = freeze_backbone
        self._num_levels = num_levels
        self._loss_type = loss_type
        self._level_embedding_dim = level_embedding_dim

        # Load pretrained ConvNext
        if variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown variant: {variant}. Choose from {list(self.VARIANTS.keys())}"
            )

        model_fn, weights = self.VARIANTS[variant]
        self.backbone: nn.Module = model_fn(weights=weights if pretrained else None)

        # Get feature dimension from backbone
        # ConvNext uses classifier[2] as the final linear layer
        classifier = self.backbone.classifier  # type: ignore[union-attr]
        feature_dim: int = classifier[2].in_features  # type: ignore[index]

        # Replace classifier with identity to get features
        self.backbone.classifier = nn.Identity()  # type: ignore[union-attr]

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
            nn.Sigmoid(),  # Output in [0, 1] for relative coordinates
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

        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()

        self._is_initialized = True

    @property
    def name(self) -> str:
        """Human-readable name for this model."""
        return f"ConvNext-{self._variant.capitalize()}-Localization"

    def forward(
        self,
        x: torch.Tensor,
        level_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images [B, C, H, W].
            level_idx: Optional level indices [B] for level embedding.

        Returns:
            Predicted coordinates [B, num_outputs] in [0, 1].
        """
        # Extract features
        features = self.backbone(x)

        # Add level embedding if available
        if self.level_embedding is not None and level_idx is not None:
            level_emb = self.level_embedding(level_idx)
            features = torch.cat([features, level_emb], dim=1)

        # Regression head
        return self.head(features)

    def get_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute training loss.

        Args:
            predictions: Predicted coordinates [B, num_outputs].
            targets: Ground truth coordinates [B, num_outputs].
            **kwargs: Additional arguments (unused).

        Returns:
            Loss tensor (scalar).
        """
        return self._loss_fn(predictions, targets)

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without prediction head.

        Args:
            x: Input images [B, C, H, W].

        Returns:
            Feature tensor [B, feature_dim].
        """
        return self.backbone(x)


class ConvNextClassifier(BaseModel):
    """ConvNext model for classification tasks.

    Can be used for grading, condition classification, etc.
    """

    VARIANTS = ConvNextLocalization.VARIANTS

    def __init__(
        self,
        variant: Literal["tiny", "small", "base", "large"] = "base",
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        label_smoothing: float = 0.1,
    ) -> None:
        """Initialize ConvNextClassifier.

        Args:
            variant: ConvNext variant.
            num_classes: Number of output classes.
            pretrained: Use ImageNet pretrained weights.
            dropout: Dropout rate.
            freeze_backbone: Freeze backbone weights.
            label_smoothing: Label smoothing for cross-entropy.
        """
        super().__init__()

        self._variant = variant
        self._num_classes = num_classes
        self._pretrained = pretrained
        self._dropout = dropout
        self._freeze_backbone = freeze_backbone

        # Load pretrained ConvNext
        model_fn, weights = self.VARIANTS[variant]
        self.backbone: nn.Module = model_fn(weights=weights if pretrained else None)

        classifier = self.backbone.classifier  # type: ignore[union-attr]
        feature_dim: int = classifier[2].in_features  # type: ignore[index]
        self.backbone.classifier = nn.Identity()  # type: ignore[union-attr]

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

        self._loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if freeze_backbone:
            self.freeze_backbone()

        self._is_initialized = True

    @property
    def name(self) -> str:
        return f"ConvNext-{self._variant.capitalize()}-Classifier"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images [B, C, H, W].

        Returns:
            Logits [B, num_classes].
        """
        features = self.backbone(x)
        return self.head(features)

    def get_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute cross-entropy loss."""
        return self._loss_fn(predictions, targets)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

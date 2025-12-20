"""ConvNext-based models for localization and other tasks.

ConvNext provides a modern CNN architecture that can compete with transformers
while maintaining the efficiency of convolutions.

Uses timm (PyTorch Image Models) for pretrained backbones with extensive
model variants and pretrained weights.
"""

from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image

from spine_vision.training.base import BaseModel


class ConvNextLocalization(BaseModel):
    """ConvNext model for coordinate regression (localization).

    Predicts (x, y) coordinates for IVD localization from spine images.
    Supports multiple ConvNext variants via timm and optional feature freezing.
    """

    # Map variant names to timm model names
    VARIANTS = {
        "tiny": "convnext_tiny.fb_in22k_ft_in1k",
        "small": "convnext_small.fb_in22k_ft_in1k",
        "base": "convnext_base.fb_in22k_ft_in1k",
        "large": "convnext_large.fb_in22k_ft_in1k",
        "xlarge": "convnext_xlarge.fb_in22k_ft_in1k",
        # ConvNeXt V2 variants
        "v2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
        "v2_small": "convnextv2_small.fcmae",
        "v2_base": "convnextv2_base.fcmae_ft_in22k_in1k",
        "v2_large": "convnextv2_large.fcmae_ft_in22k_in1k",
        "v2_huge": "convnextv2_huge.fcmae_ft_in22k_in1k",
    }

    def __init__(
        self,
        variant: Literal[
            "tiny", "small", "base", "large", "xlarge",
            "v2_tiny", "v2_small", "v2_base", "v2_large", "v2_huge"
        ] = "base",
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
            variant: ConvNext variant (see VARIANTS for options).
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

        # Load pretrained ConvNext via timm
        if variant not in self.VARIANTS:
            raise ValueError(
                f"Unknown variant: {variant}. Choose from {list(self.VARIANTS.keys())}"
            )

        model_name = self.VARIANTS[variant]
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head, get features
        )

        # Get feature dimension from backbone
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
        return f"ConvNext-{self._variant.replace('_', '-').title()}-Localization"

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
        # Extract features via timm backbone
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

    def test_inference(
        self,
        images: Sequence[str | Path | Image.Image | np.ndarray],
        image_size: tuple[int, int] = (224, 224),
        device: str | torch.device | None = None,
        level_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Test inference with a list of images.

        Extended version that supports level indices for this localization model.

        Args:
            images: List of images - can be file paths, PIL Images, or numpy arrays.
            image_size: Target size for resizing images (H, W).
            device: Device to run inference on. If None, uses model's current device.
            level_indices: Optional list of IVD level indices (0-4) for each image.

        Returns:
            Dictionary containing:
                - predictions: Predicted coordinates as numpy array [N, 2]
                - images: Preprocessed images as numpy array (for visualization)
                - inference_time_ms: Total inference time in milliseconds
                - coords_pixel: Coordinates converted to pixel space based on image_size
        """
        import time

        from torchvision import transforms

        # Determine device
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)

        # Build preprocessing transform
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Process images
        processed_tensors: list[torch.Tensor] = []
        processed_images: list[np.ndarray] = []

        for img in images:
            # Convert to PIL Image
            if isinstance(img, (str, Path)):
                pil_img = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                pil_img = Image.fromarray(img).convert("RGB")
            elif isinstance(img, Image.Image):
                pil_img = img.convert("RGB")
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

            # Store original (resized) for visualization
            resized = pil_img.resize((image_size[1], image_size[0]))
            processed_images.append(np.array(resized))

            # Transform and add to batch
            tensor = transform(pil_img)
            processed_tensors.append(tensor)

        # Create batch
        batch = torch.stack(processed_tensors).to(device)

        # Prepare level indices if provided
        level_idx_tensor: torch.Tensor | None = None
        if level_indices is not None and self.level_embedding is not None:
            level_idx_tensor = torch.tensor(level_indices, dtype=torch.long, device=device)

        # Run inference
        self.eval()
        start_time = time.perf_counter()
        with torch.no_grad():
            predictions = self.forward(batch, level_idx_tensor)
        end_time = time.perf_counter()

        inference_time_ms = (end_time - start_time) * 1000
        predictions_np = predictions.cpu().numpy()

        # Convert to pixel coordinates
        h, w = image_size
        coords_pixel = predictions_np * np.array([w, h])

        return {
            "predictions": predictions_np,
            "coords_pixel": coords_pixel,
            "images": np.stack(processed_images),
            "inference_time_ms": inference_time_ms,
            "num_images": len(images),
            "device": str(device),
        }


class ConvNextClassifier(BaseModel):
    """ConvNext model for classification tasks.

    Can be used for grading, condition classification, etc.
    Uses timm for pretrained backbones.
    """

    VARIANTS = ConvNextLocalization.VARIANTS

    def __init__(
        self,
        variant: Literal[
            "tiny", "small", "base", "large", "xlarge",
            "v2_tiny", "v2_small", "v2_base", "v2_large", "v2_huge"
        ] = "base",
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

        # Load pretrained ConvNext via timm
        model_name = self.VARIANTS[variant]
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        feature_dim: int = self.backbone.num_features  # type: ignore[assignment]

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
        return f"ConvNext-{self._variant.replace('_', '-').title()}-Classifier"

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

    def test_inference(
        self,
        images: Sequence[str | Path | Image.Image | np.ndarray],
        image_size: tuple[int, int] = (224, 224),
        device: str | torch.device | None = None,
        return_probabilities: bool = True,
    ) -> dict[str, Any]:
        """Test inference with a list of images.

        Args:
            images: List of images - can be file paths, PIL Images, or numpy arrays.
            image_size: Target size for resizing images (H, W).
            device: Device to run inference on. If None, uses model's current device.
            return_probabilities: If True, return softmax probabilities.

        Returns:
            Dictionary containing:
                - predictions: Class predictions as numpy array [N]
                - logits: Raw logits as numpy array [N, num_classes]
                - probabilities: Softmax probabilities (if return_probabilities=True)
                - images: Preprocessed images as numpy array (for visualization)
                - inference_time_ms: Total inference time in milliseconds
        """
        result = super().test_inference(images, image_size, device)

        logits = result["predictions"]
        predictions = np.argmax(logits, axis=1)

        result["logits"] = logits
        result["predictions"] = predictions

        if return_probabilities:
            # Compute softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            result["probabilities"] = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return result


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

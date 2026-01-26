"""Base model abstract class for trainable models.

Provides the abstract interface that all trainable models should implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class BaseModel(nn.Module, ABC):
    """Abstract base class for trainable models.

    All models should inherit from this class and implement:
    - forward(): The forward pass
    - get_loss(): Compute training loss
    - predict(): Run inference (no gradients)
    - test_inference(): Test model with sample images
    """

    def __init__(self) -> None:
        super().__init__()
        self._is_initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this model."""
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs: Any) -> Any:
        """Forward pass.

        Args:
            x: Input tensor.
            **kwargs: Additional model-specific arguments.

        Returns:
            Model output (tensor or dict of tensors for multi-task models).
        """
        ...

    @abstractmethod
    def get_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute training loss.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional arguments (e.g., weights, masks).

        Returns:
            Loss tensor (scalar).
        """
        ...

    def predict(self, x: torch.Tensor, **kwargs: Any) -> Any:
        """Run inference without gradients.

        Args:
            x: Input tensor.
            **kwargs: Additional model-specific arguments.

        Returns:
            Model predictions.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x, **kwargs)

    def test_inference(
        self,
        images: Sequence[str | Path | Image.Image | np.ndarray],
        image_size: tuple[int, int] = (224, 224),
        device: str | torch.device | None = None,
    ) -> dict[str, Any]:
        """Test inference with a list of images.

        This method provides a convenient way to test the model with
        various image inputs (file paths, PIL Images, or numpy arrays).

        Args:
            images: List of images - can be file paths, PIL Images, or numpy arrays.
            image_size: Target size for resizing images (H, W).
            device: Device to run inference on. If None, uses model's current device.

        Returns:
            Dictionary containing:
                - predictions: Model predictions as numpy array
                - images: Preprocessed images as numpy array (for visualization)
                - inference_time_ms: Total inference time in milliseconds
        """
        import time

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

        # Run inference
        self.eval()
        start_time = time.perf_counter()
        with torch.no_grad():
            predictions = self.forward(batch)
        end_time = time.perf_counter()

        inference_time_ms = (end_time - start_time) * 1000

        return {
            "predictions": predictions.cpu().numpy(),
            "images": np.stack(processed_images),
            "inference_time_ms": inference_time_ms,
            "num_images": len(images),
            "device": str(device),
        }

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters (if applicable)."""
        pass

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters (if applicable)."""
        pass

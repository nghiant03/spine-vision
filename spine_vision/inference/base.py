"""Base classes for inference models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


@dataclass
class InferenceResult:
    """Container for inference results.

    Attributes:
        prediction: The model output (segmentation mask, bounding boxes, etc.).
        probabilities: Optional probability maps.
        metadata: Additional model-specific metadata.
    """

    prediction: sitk.Image | np.ndarray
    probabilities: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class InferenceModel(ABC):
    """Abstract base class for inference models.

    All inference models (segmentation, localization, etc.) should
    inherit from this class and implement the required methods.
    """

    def __init__(self, model_path: Path, device: str = "cuda:0") -> None:
        """Initialize model.

        Args:
            model_path: Path to model weights or directory.
            device: Device to run inference on.
        """
        self.model_path = model_path
        self.device = device
        self._model: Any = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this model."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Called lazily on first predict()."""
        ...

    @abstractmethod
    def predict(self, image: sitk.Image) -> InferenceResult:
        """Run inference on an image.

        Args:
            image: Input SimpleITK Image.

        Returns:
            InferenceResult containing prediction and optional metadata.
        """
        ...

    def predict_batch(self, images: list[sitk.Image]) -> list[InferenceResult]:
        """Run inference on multiple images.

        Default implementation processes sequentially. Subclasses may
        override for batch processing.

        Args:
            images: List of input images.

        Returns:
            List of InferenceResults.
        """
        return [self.predict(img) for img in images]

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded before prediction."""
        if self._model is None:
            self.load()

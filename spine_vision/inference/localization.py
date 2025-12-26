"""Localization models for bounding box and landmark detection."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from loguru import logger

from spine_vision.inference.base import InferenceModel, InferenceResult


@dataclass
class BoundingBox:
    """3D bounding box representation.

    Attributes:
        min_coords: (x, y, z) minimum corner.
        max_coords: (x, y, z) maximum corner.
        label: Optional class label.
        confidence: Optional confidence score.
    """

    min_coords: tuple[float, float, float]
    max_coords: tuple[float, float, float]
    label: int | None = None
    confidence: float | None = None

    @property
    def center(self) -> tuple[float, float, float]:
        """Get center point of bounding box."""
        result = tuple(
            (mn + mx) / 2 for mn, mx in zip(self.min_coords, self.max_coords)
        )
        return (result[0], result[1], result[2])

    @property
    def size(self) -> tuple[float, float, float]:
        """Get size (width, height, depth) of bounding box."""
        result = tuple(mx - mn for mn, mx in zip(self.min_coords, self.max_coords))
        return (result[0], result[1], result[2])


@dataclass
class Landmark:
    """3D landmark/keypoint representation.

    Attributes:
        coords: (x, y, z) coordinates.
        label: Optional label identifier.
        confidence: Optional confidence score.
    """

    coords: tuple[float, float, float]
    label: int | str | None = None
    confidence: float | None = None


class LocalizationModel(InferenceModel):
    """Base class for localization models.

    Detects bounding boxes or landmarks in medical images.
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda:0",
        confidence_threshold: float = 0.5,
    ) -> None:
        """Initialize localization model.

        Args:
            model_path: Path to model weights.
            device: Device for inference.
            confidence_threshold: Minimum confidence for detections.
        """
        super().__init__(model_path, device)
        self.confidence_threshold = confidence_threshold

    @property
    def name(self) -> str:
        return "Localization Model"

    def load(self) -> None:
        """Load model weights."""
        logger.info(f"Loading {self.name} from {self.model_path}")
        raise NotImplementedError(
            "Subclasses must implement load() with specific model loading"
        )

    def predict(self, image: sitk.Image) -> InferenceResult:
        """Run localization on an image.

        Args:
            image: Input SimpleITK Image.

        Returns:
            InferenceResult with bounding boxes in metadata.
        """
        raise NotImplementedError(
            "Subclasses must implement predict() with specific detection logic"
        )

    def detect_boxes(self, image: sitk.Image) -> list[BoundingBox]:
        """Detect bounding boxes in an image.

        Args:
            image: Input SimpleITK Image.

        Returns:
            List of detected BoundingBox objects.
        """
        result = self.predict(image)
        return result.metadata.get("boxes", [])

    def detect_landmarks(self, image: sitk.Image) -> list[Landmark]:
        """Detect landmarks in an image.

        Args:
            image: Input SimpleITK Image.

        Returns:
            List of detected Landmark objects.
        """
        result = self.predict(image)
        return result.metadata.get("landmarks", [])


def boxes_from_segmentation(
    segmentation: sitk.Image,
    labels: list[int] | None = None,
) -> list[BoundingBox]:
    """Extract bounding boxes from a segmentation mask.

    Args:
        segmentation: Segmentation mask as SimpleITK Image.
        labels: Optional list of labels to extract. If None, all non-zero labels.

    Returns:
        List of BoundingBox objects, one per label.
    """
    seg_array = sitk.GetArrayFromImage(segmentation)
    spacing = segmentation.GetSpacing()
    origin = segmentation.GetOrigin()

    if labels is None:
        labels = [int(label) for label in np.unique(seg_array) if label != 0]

    boxes = []
    for label in labels:
        coords = np.where(seg_array == label)
        if len(coords[0]) == 0:
            continue

        min_idx = np.array([c.min() for c in coords])
        max_idx = np.array([c.max() for c in coords])

        min_coords = tuple(origin[i] + min_idx[2 - i] * spacing[i] for i in range(3))
        max_coords = tuple(origin[i] + max_idx[2 - i] * spacing[i] for i in range(3))

        boxes.append(
            BoundingBox(
                min_coords=min_coords,
                max_coords=max_coords,
                label=label,
            )
        )

    return boxes

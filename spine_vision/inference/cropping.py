"""ROI cropping utilities for extracting regions from images."""

from pathlib import Path
from typing import Sequence

import numpy as np
import SimpleITK as sitk
from loguru import logger

from spine_vision.inference.base import InferenceModel, InferenceResult
from spine_vision.inference.localization import BoundingBox, boxes_from_segmentation


class ROICropper(InferenceModel):
    """Crops regions of interest from medical images.

    Can use either explicit bounding boxes or derive them from
    segmentation masks.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        device: str = "cpu",
        padding: tuple[int, int, int] = (10, 10, 10),
        labels: list[int] | None = None,
    ) -> None:
        """Initialize ROI cropper.

        Args:
            model_path: Not used for cropping, kept for interface compatibility.
            device: Not used for cropping.
            padding: (x, y, z) padding around each ROI in voxels.
            labels: Specific labels to crop. If None, crops all non-zero labels.
        """
        super().__init__(model_path or Path("."), device)
        self.padding = padding
        self.labels = labels
        self._model = True

    @property
    def name(self) -> str:
        return "ROI Cropper"

    def load(self) -> None:
        """No loading needed for cropping."""
        pass

    def predict(self, image: sitk.Image) -> InferenceResult:
        """Not applicable for standalone cropping.

        Use crop_from_segmentation() or crop_from_boxes() instead.
        """
        raise NotImplementedError(
            "ROICropper requires segmentation mask or boxes. "
            "Use crop_from_segmentation() or crop_from_boxes()."
        )

    def crop_from_segmentation(
        self,
        image: sitk.Image,
        segmentation: sitk.Image,
        labels: list[int] | None = None,
    ) -> dict[int, sitk.Image]:
        """Crop ROIs based on segmentation mask labels.

        Args:
            image: Source image to crop from.
            segmentation: Segmentation mask with integer labels.
            labels: Labels to extract. If None, uses self.labels or all labels.

        Returns:
            Dictionary mapping label -> cropped image.
        """
        labels = labels or self.labels
        boxes = boxes_from_segmentation(segmentation, labels)

        crops = {}
        for box in boxes:
            if box.label is not None:
                crop = self._crop_box(image, box)
                crops[box.label] = crop

        logger.debug(f"Cropped {len(crops)} regions from segmentation")
        return crops

    def crop_from_boxes(
        self,
        image: sitk.Image,
        boxes: Sequence[BoundingBox],
    ) -> list[sitk.Image]:
        """Crop ROIs from explicit bounding boxes.

        Args:
            image: Source image to crop from.
            boxes: List of BoundingBox objects.

        Returns:
            List of cropped images.
        """
        crops = [self._crop_box(image, box) for box in boxes]
        logger.debug(f"Cropped {len(crops)} regions from boxes")
        return crops

    def crop_single(
        self,
        image: sitk.Image,
        center: tuple[float, float, float],
        size: tuple[int, int, int],
    ) -> sitk.Image:
        """Crop a single region centered at a point.

        Args:
            image: Source image.
            center: (x, y, z) center point in physical coordinates.
            size: (width, height, depth) in voxels.

        Returns:
            Cropped image.
        """
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        img_size = image.GetSize()

        center_idx = [int((center[i] - origin[i]) / spacing[i]) for i in range(3)]

        start_idx = [max(0, center_idx[i] - size[i] // 2) for i in range(3)]
        end_idx = [min(img_size[i], center_idx[i] + size[i] // 2) for i in range(3)]

        crop_size = [end_idx[i] - start_idx[i] for i in range(3)]

        return sitk.RegionOfInterest(image, crop_size, start_idx)

    def _crop_box(self, image: sitk.Image, box: BoundingBox) -> sitk.Image:
        """Crop image to bounding box with padding."""
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        img_size = image.GetSize()

        start_idx = [
            max(0, int((box.min_coords[i] - origin[i]) / spacing[i]) - self.padding[i])
            for i in range(3)
        ]
        end_idx = [
            min(
                img_size[i],
                int((box.max_coords[i] - origin[i]) / spacing[i]) + self.padding[i],
            )
            for i in range(3)
        ]

        crop_size = [end_idx[i] - start_idx[i] for i in range(3)]

        return sitk.RegionOfInterest(image, crop_size, start_idx)


def crop_to_nonzero(
    image: sitk.Image,
    mask: sitk.Image | None = None,
    padding: tuple[int, int, int] = (0, 0, 0),
) -> sitk.Image:
    """Crop image to the non-zero region.

    Args:
        image: Image to crop.
        mask: Optional mask to determine crop region. If None, uses image.
        padding: Additional padding around non-zero region.

    Returns:
        Cropped image.
    """
    if mask is None:
        mask = image

    mask_array = sitk.GetArrayFromImage(mask)
    coords = np.where(mask_array != 0)

    if len(coords[0]) == 0:
        logger.warning("No non-zero voxels found, returning original image")
        return image

    min_idx = [max(0, c.min() - padding[2 - i]) for i, c in enumerate(coords)]
    max_idx = [c.max() + padding[2 - i] for i, c in enumerate(coords)]

    img_size = image.GetSize()
    start_idx = [min_idx[2], min_idx[1], min_idx[0]]
    end_idx = [min(img_size[i], max_idx[2 - i] + 1) for i in range(3)]

    crop_size = [end_idx[i] - start_idx[i] for i in range(3)]

    return sitk.RegionOfInterest(image, crop_size, start_idx)

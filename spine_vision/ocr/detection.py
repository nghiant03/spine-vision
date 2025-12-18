"""Text detection using PaddleOCR."""

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from paddleocr import TextDetection


class TextDetector:
    """Wrapper for PaddleOCR text detection.

    Detects text regions in images and returns bounding polygons.
    """

    def __init__(
        self,
        model_name: str = "PP-OCRv5_server_det",
        use_gpu: bool = True,
    ) -> None:
        """Initialize text detector.

        Args:
            model_name: PaddleOCR detection model name.
            use_gpu: Whether to use GPU acceleration.
        """
        logger.info(f"Loading detection model: {model_name}")
        self.model = TextDetection(model_name=model_name)
        self.use_gpu = use_gpu

    def detect(self, image_path: Path | str) -> list[np.ndarray]:
        """Detect text regions in an image.

        Args:
            image_path: Path to image file.

        Returns:
            List of polygon arrays, each with shape (4, 2) representing
            the four corners of a text region.
        """
        path_str = str(image_path) if isinstance(image_path, Path) else image_path
        result = self.model.predict(path_str)

        if not result:
            logger.debug(f"No text detected in {image_path}")
            return []

        boxes = result[0].get("dt_polys", [])
        return [np.array(box).astype(np.int32) for box in boxes]

    def detect_from_array(self, image: np.ndarray) -> list[np.ndarray]:
        """Detect text regions from a numpy array.

        Args:
            image: Image as numpy array (RGB).

        Returns:
            List of polygon arrays.
        """
        result = self.model.predict(image)

        if not result:
            return []

        boxes = result[0].get("dt_polys", [])
        return [np.array(box).astype(np.int32) for box in boxes]

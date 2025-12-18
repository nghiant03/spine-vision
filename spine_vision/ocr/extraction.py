"""Document field extraction combining detection and recognition."""

from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from spine_vision.ocr.detection import TextDetector
from spine_vision.ocr.recognition import TextRecognizer


def crop_polygon(image_np: np.ndarray, points: np.ndarray) -> Image.Image:
    """Crop and perspective-transform a polygon region from an image.

    Applies perspective transformation to extract a rectangular crop
    from a quadrilateral text region.

    Args:
        image_np: Source image as numpy array.
        points: Four corner points as (4, 2) array in order:
                [top-left, top-right, bottom-right, bottom-left].

    Returns:
        Cropped and rectified PIL Image.
    """
    points = points.astype(np.float32)
    tl, tr, br, bl = points

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(points, dst)
    warped = cv2.warpPerspective(image_np, M, (max_width, max_height))

    return Image.fromarray(warped)


class DocumentExtractor:
    """Extract text fields from medical documents.

    Combines text detection and recognition to extract all text
    from a document image.
    """

    def __init__(
        self,
        detection_model: str = "PP-OCRv5_server_det",
        recognition_model: str = "vgg_transformer",
        device: str = "cuda:0",
        use_gpu: bool = True,
    ) -> None:
        """Initialize document extractor.

        Args:
            detection_model: PaddleOCR detection model name.
            recognition_model: VietOCR recognition model name.
            device: Device for recognition model.
            use_gpu: Whether to use GPU for detection.
        """
        self.detector = TextDetector(detection_model, use_gpu)
        self.recognizer = TextRecognizer(recognition_model, device)

    def extract(self, image_path: Path) -> list[str]:
        """Extract all text lines from a document image.

        Args:
            image_path: Path to document image.

        Returns:
            List of recognized text strings, one per detected region.
        """
        boxes = self.detector.detect(image_path)

        if not boxes:
            logger.debug(f"No text detected in {image_path}")
            return []

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        text_lines = []
        for box in boxes:
            crop = crop_polygon(image_np, box)
            text = self.recognizer.recognize(crop)
            text_lines.append(text)

        return text_lines

    def extract_from_array(self, image: np.ndarray) -> list[str]:
        """Extract all text lines from an image array.

        Args:
            image: Image as numpy array (RGB).

        Returns:
            List of recognized text strings.
        """
        boxes = self.detector.detect_from_array(image)

        if not boxes:
            return []

        text_lines = []
        for box in boxes:
            crop = crop_polygon(image, box)
            text = self.recognizer.recognize(crop)
            text_lines.append(text)

        return text_lines

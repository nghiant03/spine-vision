"""Document field extraction combining detection and recognition."""

from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from spine_vision.io.pdf import pdf_first_page_to_array
from spine_vision.ocr.detection import TextDetector
from spine_vision.ocr.recognition import TextRecognizer

# Supported file extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS


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
    from document images or PDFs.
    """

    def __init__(
        self,
        detection_model: str = "PP-OCRv5_server_det",
        recognition_model: str = "vgg_transformer",
        device: str = "cuda:0",
        use_gpu: bool = True,
        pdf_dpi: int = 200,
    ) -> None:
        """Initialize document extractor.

        Args:
            detection_model: PaddleOCR detection model name.
            recognition_model: VietOCR recognition model name.
            device: Device for recognition model.
            use_gpu: Whether to use GPU for detection.
            pdf_dpi: DPI for rendering PDF pages.
        """
        self.detector = TextDetector(detection_model, use_gpu)
        self.recognizer = TextRecognizer(recognition_model, device)
        self.pdf_dpi = pdf_dpi

    def extract(self, document_path: Path) -> list[str]:
        """Extract all text lines from a document (image or PDF).

        Args:
            document_path: Path to document file (image or PDF).

        Returns:
            List of recognized text strings, one per detected region.

        Raises:
            ValueError: If file extension is not supported.
        """
        suffix = document_path.suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {suffix}. "
                f"Supported: {SUPPORTED_EXTENSIONS}"
            )

        if suffix in PDF_EXTENSIONS:
            image_np = pdf_first_page_to_array(document_path, dpi=self.pdf_dpi)
            return self._extract_from_array(image_np, str(document_path))
        else:
            return self._extract_from_image_file(document_path)

    def _extract_from_image_file(self, image_path: Path) -> list[str]:
        """Extract text from an image file.

        Args:
            image_path: Path to image file.

        Returns:
            List of recognized text strings.
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

    def _extract_from_array(
        self, image: np.ndarray, source_name: str = "array"
    ) -> list[str]:
        """Extract text from an image array.

        Args:
            image: Image as numpy array (RGB).
            source_name: Name for logging.

        Returns:
            List of recognized text strings.
        """
        boxes = self.detector.detect_from_array(image)

        if not boxes:
            logger.debug(f"No text detected in {source_name}")
            return []

        text_lines = []
        for box in boxes:
            crop = crop_polygon(image, box)
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
        return self._extract_from_array(image)

    def extract_from_pdf_crop(
        self,
        pdf_path: Path,
        crop_region: tuple[int, int, int, int],
    ) -> list[str]:
        """Extract text from a cropped region of a PDF's first page.

        Args:
            pdf_path: Path to PDF file.
            crop_region: Crop box as (x1, y1, x2, y2) in pixels.

        Returns:
            List of recognized text strings from the cropped region.
        """
        image_np = pdf_first_page_to_array(pdf_path, dpi=self.pdf_dpi)
        x1, y1, x2, y2 = crop_region
        cropped = image_np[y1:y2, x1:x2]
        return self._extract_from_array(cropped, f"{pdf_path} (cropped)")

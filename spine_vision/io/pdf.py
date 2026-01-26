"""PDF reading utilities for document processing."""

from pathlib import Path

import fitz
import numpy as np
from PIL import Image


def pdf_to_images(
    pdf_path: Path,
    dpi: int = 200,
) -> list[Image.Image]:
    """Convert PDF pages to PIL Images.

    Args:
        pdf_path: Path to PDF file.
        dpi: Resolution for rendering (default 200).

    Returns:
        List of PIL Images, one per page.
    """
    doc = fitz.open(pdf_path)
    images: list[Image.Image] = []

    zoom = dpi / 72  # 72 is default PDF DPI
    matrix = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)

    doc.close()
    return images


def pdf_to_arrays(
    pdf_path: Path,
    dpi: int = 200,
) -> list[np.ndarray]:
    """Convert PDF pages to numpy arrays.

    Args:
        pdf_path: Path to PDF file.
        dpi: Resolution for rendering (default 200).

    Returns:
        List of numpy arrays (RGB), one per page.
    """
    images = pdf_to_images(pdf_path, dpi)
    return [np.array(img) for img in images]


def pdf_first_page_to_array(
    pdf_path: Path,
    dpi: int = 200,
) -> np.ndarray:
    """Convert first PDF page to numpy array.

    Args:
        pdf_path: Path to PDF file.
        dpi: Resolution for rendering (default 200).

    Returns:
        First page as numpy array (RGB).

    Raises:
        ValueError: If PDF has no pages.
    """
    doc = fitz.open(pdf_path)

    if doc.page_count == 0:
        doc.close()
        raise ValueError(f"PDF has no pages: {pdf_path}")

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    page = doc[0]
    pix = page.get_pixmap(matrix=matrix)
    array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    doc.close()

    return array.copy()  # Copy to own memory after doc is closed

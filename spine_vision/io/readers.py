"""Medical image readers supporting multiple formats."""

from pathlib import Path
from enum import Enum, auto

import SimpleITK as sitk
from loguru import logger


class ImageFormat(Enum):
    """Supported medical image formats."""

    DICOM = auto()
    DICOM_FILE = auto()
    NIFTI = auto()
    MHA = auto()
    MHD = auto()
    NRRD = auto()
    UNKNOWN = auto()


EXTENSION_MAP: dict[str, ImageFormat] = {
    ".nii": ImageFormat.NIFTI,
    ".nii.gz": ImageFormat.NIFTI,
    ".mha": ImageFormat.MHA,
    ".mhd": ImageFormat.MHD,
    ".nrrd": ImageFormat.NRRD,
    ".dcm": ImageFormat.DICOM_FILE,
}


def detect_format(path: Path) -> ImageFormat:
    """Detect medical image format from path.

    Args:
        path: Path to image file or DICOM directory.

    Returns:
        Detected ImageFormat enum value.
    """
    if path.is_dir():
        return ImageFormat.DICOM

    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ImageFormat.NIFTI

    suffix = path.suffix.lower()
    return EXTENSION_MAP.get(suffix, ImageFormat.UNKNOWN)


def read_dicom_series(folder_path: Path) -> sitk.Image:
    """Read DICOM series from a directory.

    Args:
        folder_path: Path to directory containing DICOM files.

    Returns:
        SimpleITK Image object.

    Raises:
        ValueError: If no DICOM series found in directory.
    """
    logger.debug(f"Reading DICOM series from: {folder_path}")
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(folder_path))

    if not series_ids:
        raise ValueError(f"No DICOM series found in {folder_path}")

    dicom_names = reader.GetGDCMSeriesFileNames(str(folder_path), series_ids[0])
    reader.SetFileNames(dicom_names)
    return reader.Execute()


def read_nifti(file_path: Path) -> sitk.Image:
    """Read NIfTI file.

    Args:
        file_path: Path to .nii or .nii.gz file.

    Returns:
        SimpleITK Image object.
    """
    logger.debug(f"Reading NIfTI from: {file_path}")
    return sitk.ReadImage(str(file_path))


def read_mha(file_path: Path) -> sitk.Image:
    """Read MHA/MHD file.

    Args:
        file_path: Path to .mha or .mhd file.

    Returns:
        SimpleITK Image object.
    """
    logger.debug(f"Reading MHA/MHD from: {file_path}")
    return sitk.ReadImage(str(file_path))


def read_nrrd(file_path: Path) -> sitk.Image:
    """Read NRRD file.

    Args:
        file_path: Path to .nrrd file.

    Returns:
        SimpleITK Image object.
    """
    logger.debug(f"Reading NRRD from: {file_path}")
    return sitk.ReadImage(str(file_path))


def read_dicom_file(file_path: Path) -> sitk.Image:
    """Read a single DICOM file.

    Args:
        file_path: Path to .dcm file.

    Returns:
        SimpleITK Image object.
    """
    logger.debug(f"Reading DICOM file: {file_path}")
    return sitk.ReadImage(str(file_path))


def read_medical_image(path: Path) -> sitk.Image:
    """Read medical image with automatic format detection.

    Supports DICOM directories, NIfTI (.nii, .nii.gz), MHA, MHD, and NRRD formats.

    Args:
        path: Path to image file or DICOM directory.

    Returns:
        SimpleITK Image object.

    Raises:
        ValueError: If format is unknown or unsupported.
        FileNotFoundError: If path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    format_type = detect_format(path)
    logger.debug(f"Detected format: {format_type.name}")

    match format_type:
        case ImageFormat.DICOM:
            return read_dicom_series(path)
        case ImageFormat.DICOM_FILE:
            return read_dicom_file(path)
        case ImageFormat.NIFTI:
            return read_nifti(path)
        case ImageFormat.MHA | ImageFormat.MHD:
            return read_mha(path)
        case ImageFormat.NRRD:
            return read_nrrd(path)
        case _:
            raise ValueError(f"Unsupported format for path: {path}")

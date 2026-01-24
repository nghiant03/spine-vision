"""Create classification dataset for radiological grading prediction.

Combines Phenikaa and SPIDER datasets with IVD-level cropping based on
localization model predictions. Uses middle sagittal slice from T1/T2 sequences.

Dataset structure:
    output_path/
    ├── images/                  # Cropped IVD images (PNG)
    │   ├── phenikaa_<id>_<series>_<level>.png
    │   └── spider_<id>_<series>_<level>.png
    └── annotations.csv          # CSV with grading labels
"""

import csv
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import SimpleITK as sitk
import torch
from loguru import logger
from PIL import Image
from pydantic import BaseModel, computed_field
from tqdm.rich import tqdm
from tqdm.std import TqdmExperimentalWarning

from spine_vision.core import BaseConfig, add_file_log, setup_logger
from spine_vision.datasets.base import BaseProcessor, ProcessingResult
from spine_vision.io import normalize_to_uint8, read_medical_image

# Type alias for crop mode
CropMode = Literal["horizontal", "rotated"]

# Constants
ISOTROPIC_SPACING = (0.3, 0.3, 0.3)  # mm - Standard isotropic spacing for resampling
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization mean (RGB)
IMAGENET_STD = [0.229, 0.224, 0.225]  # ImageNet normalization std (RGB)

# Default IVD center positions in normalized coordinates (0-1) for fallback
# Approximate positions for L1/L2 through L5/S1 in typical sagittal view
DEFAULT_IVD_CENTERS = {
    0: (0.5, 0.25),  # L1/L2
    1: (0.5, 0.35),  # L2/L3
    2: (0.5, 0.45),  # L3/L4
    3: (0.5, 0.55),  # L4/L5
    4: (0.5, 0.65),  # L5/S1
}


class ClassificationDatasetConfig(BaseConfig):
    """Configuration for classification dataset creation."""

    base_path: Path = Path.cwd() / "data"
    """Base data directory."""

    output_name: str = "classification"
    """Output dataset folder name."""

    localization_model_path: Path | None = None
    """Path to trained localization model checkpoint. If None, uses center crop."""

    model_variant: Literal[
        "tiny", "small", "base", "large", "xlarge",
        "v2_tiny", "v2_small", "v2_base", "v2_large", "v2_huge"
    ] = "base"
    """ConvNext variant for localization model."""

    crop_size: tuple[int, int] = (256, 256)
    """Output size of cropped IVD regions in pixels (H, W)."""

    crop_delta_mm: tuple[float, float, float, float] = (55, 15, 17.5, 20)
    """Crop region deltas (left, right, top, bottom) in millimeters."""

    crop_mode: CropMode = "horizontal"
    """Crop mode: 'horizontal' for axis-aligned crops, 'rotated' for spinal canal-based rotated crops."""

    last_disc_angle_boost: float = 1.0
    """Multiplier for rotation angle at L5/S1 to account for steep lordosis curvature."""

    image_size: tuple[int, int] = (512, 512)
    """Input image size for localization model (H, W)."""

    include_phenikaa: bool = True
    """Include Phenikaa dataset."""

    include_spider: bool = True
    """Include SPIDER dataset."""

    append_to_existing: bool = True
    """If output directory exists, append new data to existing annotations."""

    device: str = "cuda:0"
    """Device for model inference."""

    @computed_field
    @property
    def phenikaa_path(self) -> Path:
        """Path to Phenikaa interim dataset."""
        return self.base_path / "interim" / "Phenikaa"

    @computed_field
    @property
    def spider_path(self) -> Path:
        """Path to raw SPIDER dataset."""
        return self.base_path / "raw" / "SPIDER"

    @computed_field
    @property
    def output_path(self) -> Path:
        """Output dataset path."""
        path = self.base_path / "processed" / self.output_name
        path.mkdir(parents=True, exist_ok=True)
        return path


class ClassificationRecord(BaseModel):
    """A single classification record."""

    image_path: str
    patient_id: str
    ivd_level: int
    series_type: str
    source: str
    pfirrmann_grade: int
    disc_herniation: int
    disc_narrowing: int
    disc_bulging: int
    spondylolisthesis: int
    modic: int
    up_endplate: int
    low_endplate: int


# IVD level mapping (index 0-4 corresponds to L1/L2 to L5/S1)
IVD_LEVEL_NAMES = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]


def convert_spider_to_phenikaa_level(spider_level: int) -> int:
    """Convert SPIDER IVD level to Phenikaa convention.

    SPIDER labels discs from L5/S1 to L1/L2 as 1 to 5 (bottom to top).
    Phenikaa labels discs from L1/L2 to L5/S1 as 1 to 5 (top to bottom).

    Args:
        spider_level: IVD level in SPIDER format (1=L5/S1, 5=L1/L2).

    Returns:
        IVD level in Phenikaa format (1=L1/L2, 5=L5/S1).
    """
    return 6 - spider_level

def resample_to_isotropic(
    image: sitk.Image,
    new_spacing: tuple[float, float, float] = ISOTROPIC_SPACING,
) -> sitk.Image:
    """
    Resamples a SimpleITK image to a uniform spacing (square pixels).
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    new_size = [
        int(round(osz * osp / nsp))
        for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    
    resampler.SetInterpolator(sitk.sitkLinear)
    
    return resampler.Execute(image)

def extract_middle_slice(image: sitk.Image) -> np.ndarray:
    """Extract middle sagittal slice from 3D image.

    Args:
        image: SimpleITK 3D image.

    Returns:
        2D numpy array of the middle sagittal slice.
    """
    image = sitk.DICOMOrient(image, 'LPI')
    arr = sitk.GetArrayFromImage(image)

    if arr.ndim == 2:
        return arr

    mid_idx = arr.shape[2] // 2
    return arr[:, :, mid_idx]


def get_slice_spacing(image: sitk.Image) -> tuple[float, float]:
    """Get 2D spacing for middle sagittal slice.

    Args:
        image: SimpleITK 3D image (should be oriented to LPI).

    Returns:
        2D spacing (row_spacing, col_spacing) in mm for the sagittal slice.
        After LPI orientation, sagittal slice is (I, P) plane, so returns (spacing_I, spacing_P).
    """
    oriented = sitk.DICOMOrient(image, 'LPI')
    spacing = oriented.GetSpacing()  # (L, P, I) after LPI orientation

    arr = sitk.GetArrayFromImage(oriented)
    if arr.ndim == 2:
        # 2D image: spacing is (x, y), array is (y, x)
        return (spacing[1], spacing[0])

    # 3D: array is (I, P, L), sagittal slice is (I, P)
    # Return (spacing_I, spacing_P) = (spacing[2], spacing[1])
    return (spacing[2], spacing[1])


def resize_with_padding(
    image: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Resize image to target size with letterboxing (no distortion).

    Scales the image so the longest dimension equals the target size,
    maintains the original aspect ratio, and centers the result on a
    black (zero-filled) square canvas.

    Args:
        image: Input 2D grayscale image (H, W), any dtype.
        target_size: Output size as (height, width).

    Returns:
        Resized image with padding, shape (target_height, target_width), uint8.
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Compute scale to fit longest dimension
    scale = min(target_h / h, target_w / w)

    # New dimensions preserving aspect ratio
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    # Resize using cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Ensure uint8 output
    if resized.dtype != np.uint8:
        resized = normalize_to_uint8(resized)

    # Create black canvas and center the resized image
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)

    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas


def mm_to_pixels(
    delta_mm: tuple[float, float, float, float],
    spacing: tuple[float, float],
) -> tuple[int, int, int, int]:
    """Convert crop deltas from millimeters to pixels.

    Args:
        delta_mm: Crop deltas (left, right, top, bottom) in mm.
        spacing: Image spacing (row_spacing, col_spacing) in mm/pixel.

    Returns:
        Crop deltas (left, right, top, bottom) in pixels.
    """
    row_spacing, col_spacing = spacing
    left_mm, right_mm, top_mm, bottom_mm = delta_mm
    return (
        int(round(left_mm / col_spacing)),
        int(round(right_mm / col_spacing)),
        int(round(top_mm / row_spacing)),
        int(round(bottom_mm / row_spacing)),
    )


def get_rotation_angles(
    ivd_locations: dict[int, tuple[float, float]],
    image_shape: tuple[int, int],
    last_disc_angle_boost: float = 1.0,
) -> dict[int, float]:
    """Compute rotation angles using finite difference with quadratic extrapolation.

    For each IVD point, calculates the local tangent direction:
    - Top point: forward difference to next point
    - Middle points: central difference (prev to next)
    - Bottom point: local quadratic extrapolation using last 3 points to capture
      the steep lordosis curvature at L5-S1

    The quadratic extrapolation fits x = ay² + by + c to the last 3 points and
    evaluates dx/dy = 2ay + b at the bottom point, providing a more accurate
    slope estimate than simple backward difference.

    Args:
        ivd_locations: Dictionary mapping level index to (x, y) normalized coordinates.
        image_shape: Image shape (H, W) for denormalization.

    Returns:
        Dictionary mapping level index to rotation angle in degrees.
        Negative angle means the crop should be rotated to flatten the tilt.
    """
    if len(ivd_locations) < 2:
        return {level: 0.0 for level in ivd_locations}

    h, w = image_shape

    # Convert to pixel coordinates and sort by Y (head-to-feet)
    points = []
    for level_idx, (norm_x, norm_y) in ivd_locations.items():
        px = norm_x * w
        py = norm_y * h
        points.append((level_idx, px, py))

    # Sort by Y coordinate (ascending, head to feet)
    points.sort(key=lambda p: p[2])

    n = len(points)
    rotation_angles: dict[int, float] = {}

    for i, (level_idx, px, py) in enumerate(points):
        if i == 0:
            # First point (top): forward difference to next point
            _, next_x, next_y = points[i + 1]
            dx = next_x - px
            dy = next_y - py
            dxdy = dx / dy if dy != 0 else 0.0
        elif i == n - 1:
            # Last point (bottom): use quadratic extrapolation if we have >= 3 points
            if n >= 3:
                # Fit quadratic x = a*y² + b*y + c using last 3 points
                last_3 = points[-3:]
                y_vals = np.array([p[2] for p in last_3])
                x_vals = np.array([p[1] for p in last_3])

                # polyfit returns coefficients [a, b, c] for a*y² + b*y + c
                coeffs = np.polyfit(y_vals, x_vals, deg=2)
                a, b, _ = coeffs

                # Derivative: dx/dy = 2*a*y + b, evaluated at bottom point's y
                dxdy = 2 * a * py + b
            else:
                # Fallback: backward difference from previous point
                _, prev_x, prev_y = points[i - 1]
                dx = px - prev_x
                dy = py - prev_y
                dxdy = dx / dy if dy != 0 else 0.0
        else:
            # Middle point: central difference (prev to next)
            _, prev_x, prev_y = points[i - 1]
            _, next_x, next_y = points[i + 1]
            dx = next_x - prev_x
            dy = next_y - prev_y
            dxdy = dx / dy if dy != 0 else 0.0

        # Compute angle: θ = arctan(dx/dy) gives angle relative to vertical (y-axis)
        angle_rad = np.arctan(dxdy)
        angle_deg = float(np.degrees(angle_rad))

        # Negate to flatten the tilt (rotate opposite to spine curve)
        # Apply boost to last disc (L5/S1) to account for steep lordosis
        if i == n - 1:
            angle_deg *= last_disc_angle_boost
        rotation_angles[level_idx] = -angle_deg

    return rotation_angles


def crop_region_rotated(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    crop_size: tuple[int, int],
    crop_delta: tuple[int, int, int, int],
    rotation_angle: float,
) -> np.ndarray:
    """Crop IVD region with rotation alignment.

    Rotates the image around the disc center by the specified angle,
    then extracts an axis-aligned crop from the rotated image.
    Uses letterboxing to resize rectangular crops without distortion.

    Args:
        image: 2D grayscale image.
        center_x: Relative x coordinate (0-1).
        center_y: Relative y coordinate (0-1).
        crop_size: Output crop size in pixels (H, W).
        crop_delta: Crop deltas (left, right, top, bottom) from center in pixels.
        rotation_angle: Rotation angle in degrees (negative to flatten spine tilt).

    Returns:
        Cropped image region resized to crop_size with padding.
    """
    h, w = image.shape[:2]

    # Convert normalized coordinates to pixel coordinates
    cx = int(center_x * w)
    cy = int(center_y * h)

    left, right, top, bottom = crop_delta

    # Create rotation matrix around the disc center
    rotation_matrix = cv2.getRotationMatrix2D((cx, cy), rotation_angle, 1.0)

    # Rotate the image with border replication to avoid black corners
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Extract axis-aligned crop from rotated image
    x1 = max(0, cx - left)
    x2 = min(w, cx + right)
    y1 = max(0, cy - top)
    y2 = min(h, cy + bottom)

    crop = rotated[y1:y2, x1:x2]
    crop_uint8 = normalize_to_uint8(crop)

    # Use letterboxing to avoid distortion
    return resize_with_padding(crop_uint8, crop_size)


def crop_region_horizontal(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    crop_size: tuple[int, int],
    crop_delta: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop IVD region centered at predicted coordinates using pixel deltas.

    Uses letterboxing to resize rectangular crops to the target size without
    distortion. The crop is scaled to fit the longest dimension and centered
    on a black canvas.

    Args:
        image: 2D grayscale image.
        center_x: Relative x coordinate (0-1).
        center_y: Relative y coordinate (0-1).
        crop_size: Output crop size in pixels (H, W).
        crop_delta: Crop deltas (left, right, top, bottom) from center in pixels.

    Returns:
        Cropped image region resized to crop_size with padding.
    """
    h, w = image.shape[:2]

    cx = int(center_x * w)
    cy = int(center_y * h)

    left, right, top, bottom = crop_delta

    x1 = max(0, cx - left)
    x2 = min(w, cx + right)
    y1 = max(0, cy - top)
    y2 = min(h, cy + bottom)

    crop = image[y1:y2, x1:x2]
    crop_uint8 = normalize_to_uint8(crop)

    # Use letterboxing to avoid distortion
    return resize_with_padding(crop_uint8, crop_size)


@dataclass
class CropContext:
    """Context for IVD cropping operations."""

    image: np.ndarray
    ivd_locations: dict[int, tuple[float, float]]
    crop_size: tuple[int, int]
    crop_delta_px: tuple[int, int, int, int]
    mode: CropMode
    last_disc_angle_boost: float = 1.0
    rotation_angles: dict[int, float] | None = None

    def __post_init__(self) -> None:
        """Compute rotation angles if using rotated mode."""
        if self.mode == "rotated" and self.rotation_angles is None:
            h, w = self.image.shape[:2]
            self.rotation_angles = get_rotation_angles(
                self.ivd_locations, (h, w), self.last_disc_angle_boost
            )

    def crop(self, level_idx: int) -> np.ndarray | None:
        """Crop IVD region for a given level.

        Args:
            level_idx: IVD level index (0-4).

        Returns:
            Cropped image or None if level not found.
        """
        if level_idx not in self.ivd_locations:
            return None

        center_x, center_y = self.ivd_locations[level_idx]

        if self.mode == "rotated" and self.rotation_angles:
            rotation_angle = self.rotation_angles.get(level_idx, 0.0)
            return crop_region_rotated(
                self.image,
                center_x,
                center_y,
                self.crop_size,
                self.crop_delta_px,
                rotation_angle,
            )

        return crop_region_horizontal(
            self.image, center_x, center_y, self.crop_size, self.crop_delta_px
        )

def load_localization_model(
    model_path: Path,
    variant: str,
    device: str,
) -> torch.nn.Module:
    """Load trained localization model.

    Args:
        model_path: Path to model checkpoint.
        variant: ConvNext variant.
        device: Target device.

    Returns:
        Loaded model in eval mode.
    """
    from spine_vision.training.models import CoordinateRegressor

    model = CoordinateRegressor(
        backbone=f"convnext_{variant}" if not variant.startswith("v2_") else f"convnextv2_{variant[3:]}",
        pretrained=False,
        num_levels=5,
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def predict_ivd_locations(
    model: torch.nn.Module,
    image: np.ndarray,
    device: str,
    image_size: tuple[int, int],
) -> dict[int, tuple[float, float]]:
    """Predict IVD locations for all 5 levels.

    Args:
        model: Localization model.
        image: 2D grayscale image.
        device: Target device.
        image_size: Model input size.

    Returns:
        Dictionary mapping level index to (x, y) coordinates.
    """
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    pil_img = Image.fromarray(normalize_to_uint8(image)).convert("RGB")
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)  # [1, num_levels, 2]
        output_np = output.cpu().numpy()[0]  # [num_levels, 2]

    predictions = {
        level_idx: (float(output_np[level_idx, 0]), float(output_np[level_idx, 1]))
        for level_idx in range(output_np.shape[0])
    }

    return predictions


def get_center_fallback_locations() -> dict[int, tuple[float, float]]:
    """Get approximate center locations for IVD levels as fallback.

    Returns:
        Dictionary mapping level index to (x, y) coordinates.
    """
    return DEFAULT_IVD_CENTERS.copy()


def _load_phenikaa_labels(labels_path: Path) -> dict[str, dict[int, dict]]:
    """Load Phenikaa labels from CSV into structured dict.

    Args:
        labels_path: Path to radiological_labels.csv

    Returns:
        Dict mapping patient_id -> ivd_level -> label_row
    """
    patient_labels: dict[str, dict[int, dict]] = {}
    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = row["Patient ID"]
            ivd_level = int(row["IVD label"])
            if patient_id not in patient_labels:
                patient_labels[patient_id] = {}
            patient_labels[patient_id][ivd_level] = row
    return patient_labels


def _find_series_directory(patient_dir: Path, series_pattern: str) -> Path | None:
    """Find series directory with case-insensitive matching.

    Args:
        patient_dir: Patient's image directory
        series_pattern: Pattern to match (e.g., "sag t1")

    Returns:
        Path to series directory if found, None otherwise
    """
    normalized_pattern = series_pattern.lower().replace(" ", "")
    for subdir in patient_dir.iterdir():
        if subdir.is_dir():
            normalized_name = subdir.name.lower().replace(" ", "")
            if normalized_name == normalized_pattern:
                return subdir
    return None


def _create_classification_record(
    output_filename: str,
    patient_id: str,
    ivd_level: int,
    series_type: str,
    label_row: dict,
) -> ClassificationRecord:
    """Create a classification record from label row.

    Args:
        output_filename: Output image filename
        patient_id: Patient identifier
        ivd_level: IVD level (1-5)
        series_type: Series type (sag_t1 or sag_t2)
        label_row: CSV row with labels

    Returns:
        ClassificationRecord instance
    """
    # Extract Modic value (only one can be 1)
    modic_value = 0
    for i in range(4):
        if label_row.get(f"Modic_{i}", "0") == "1":
            modic_value = i
            break

    return ClassificationRecord(
        image_path=f"images/{output_filename}",
        patient_id=patient_id,
        ivd_level=ivd_level,
        series_type=series_type,
        source="phenikaa",
        pfirrmann_grade=int(label_row.get("Pfirrman grade", 0)),
        disc_herniation=int(label_row.get("Disc herniation", 0)),
        disc_narrowing=int(label_row.get("Disc narrowing", 0)),
        disc_bulging=int(label_row.get("Disc bulging", 0)),
        spondylolisthesis=int(label_row.get("Spondylolisthesis", 0)),
        modic=modic_value,
        up_endplate=int(label_row.get("UP endplate", 0)),
        low_endplate=int(label_row.get("LOW endplate", 0)),
    )


def process_phenikaa(
    config: ClassificationDatasetConfig,
    output_images_path: Path,
    model: torch.nn.Module | None,
    existing_image_paths: set[str] | None = None,
) -> list[ClassificationRecord]:
    """Process Phenikaa dataset.

    Args:
        config: Dataset configuration.
        output_images_path: Output directory for images.
        model: Optional localization model.
        existing_image_paths: Set of existing image paths to skip (for appending).

    Returns:
        List of classification records.
    """
    records: list[ClassificationRecord] = []
    labels_path = config.phenikaa_path / "radiological_labels.csv"
    images_path = config.phenikaa_path / "images"

    if existing_image_paths is None:
        existing_image_paths = set()

    if not labels_path.exists():
        logger.warning(f"Phenikaa labels not found: {labels_path}")
        return records

    patient_labels = _load_phenikaa_labels(labels_path)

    for patient_id, levels in tqdm(
        patient_labels.items(), desc="Processing Phenikaa", unit="patient"
    ):
        try:
            patient_dir = images_path / patient_id

            if not patient_dir.exists():
                logger.debug(f"Patient directory not found: {patient_dir}")
                continue

            for series_pattern, series_type in [("sag t1", "sag_t1"), ("sag t2", "sag_t2")]:
                series_dir = _find_series_directory(patient_dir, series_pattern)
                if series_dir is None:
                    continue

                # Check if all IVD levels for this series already exist before expensive ops
                levels_to_process: dict[int, dict] = {}
                for ivd_level, label_row in levels.items():
                    if ivd_level < 1 or ivd_level > 5:
                        continue
                    output_filename = f"phenikaa_{patient_id}_{series_type}_L{ivd_level}.png"
                    if f"images/{output_filename}" not in existing_image_paths:
                        levels_to_process[ivd_level] = label_row

                if not levels_to_process:
                    logger.debug(f"Skipping {patient_id}/{series_type}: all levels exist")
                    continue

                try:
                    image = read_medical_image(series_dir)
                    image = resample_to_isotropic(image)
                    middle_slice = extract_middle_slice(image)
                    spacing_2d = get_slice_spacing(image)
                except Exception as e:
                    logger.debug(f"Error reading {series_dir}: {e}")
                    continue

                if model is not None:
                    ivd_locations = predict_ivd_locations(
                        model, middle_slice, config.device, config.image_size
                    )
                else:
                    ivd_locations = get_center_fallback_locations()

                # Compute crop delta in pixels from mm values
                crop_delta_px = mm_to_pixels(config.crop_delta_mm, spacing_2d)

                # Create crop context for this image
                crop_ctx = CropContext(
                    image=middle_slice,
                    ivd_locations=ivd_locations,
                    crop_size=config.crop_size,
                    crop_delta_px=crop_delta_px,
                    mode=config.crop_mode,
                    last_disc_angle_boost=config.last_disc_angle_boost,
                )

                for ivd_level, label_row in levels_to_process.items():
                    level_idx = ivd_level - 1
                    crop = crop_ctx.crop(level_idx)
                    if crop is None:
                        continue

                    output_filename = f"phenikaa_{patient_id}_{series_type}_L{ivd_level}.png"
                    output_path = output_images_path / output_filename
                    Image.fromarray(crop).save(output_path)

                    record = _create_classification_record(
                        output_filename, patient_id, ivd_level, series_type, label_row
                    )
                    records.append(record)
        except Exception as e:
            logger.debug(f"Failed processing for patient {patient_id}. Error: {e}")
            continue

    return records


def process_spider(
    config: ClassificationDatasetConfig,
    output_images_path: Path,
    model: torch.nn.Module | None,
    existing_image_paths: set[str] | None = None,
) -> list[ClassificationRecord]:
    """Process SPIDER dataset.

    Args:
        config: Dataset configuration.
        output_images_path: Output directory for images.
        model: Optional localization model.
        existing_image_paths: Set of existing image paths to skip (for appending).

    Returns:
        List of classification records.
    """
    records: list[ClassificationRecord] = []
    labels_path = config.spider_path / "radiological_gradings.csv"
    images_path = config.spider_path / "images"

    if existing_image_paths is None:
        existing_image_paths = set()

    if not labels_path.exists():
        logger.warning(f"SPIDER labels not found: {labels_path}")
        return records

    patient_labels: dict[int, dict[int, dict]] = {}
    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = int(row["Patient"])
            # Convert SPIDER level (1=L5/S1) to Phenikaa level (1=L1/L2)
            ivd_level = convert_spider_to_phenikaa_level(int(row["IVD label"]))
            if patient_id not in patient_labels:
                patient_labels[patient_id] = {}
            patient_labels[patient_id][ivd_level] = row

    # Cache: key -> (middle_slice, ivd_locations, spacing_2d)
    processed_images: dict[tuple[int, str], tuple[np.ndarray, dict[int, tuple[float, float]], tuple[float, float]]] = {}

    for patient_id, levels in tqdm(
        patient_labels.items(), desc="Processing SPIDER", unit="patient"
    ):
        for series_suffix, series_type in [("t1", "sag_t1"), ("t2", "sag_t2")]:
            image_file = images_path / f"{patient_id}_{series_suffix}.mha"
            if not image_file.exists():
                continue

            # Check if all IVD levels for this series already exist before expensive ops
            levels_to_process: dict[int, dict] = {}
            for ivd_level, label_row in levels.items():
                if ivd_level < 1 or ivd_level > 5:
                    continue
                output_filename = f"spider_{patient_id}_{series_type}_L{ivd_level}.png"
                if f"images/{output_filename}" not in existing_image_paths:
                    levels_to_process[ivd_level] = label_row

            if not levels_to_process:
                logger.debug(f"Skipping {patient_id}/{series_type}: all levels exist")
                continue

            cache_key = (patient_id, series_type)
            if cache_key not in processed_images:
                try:
                    image = read_medical_image(image_file)
                    image = resample_to_isotropic(image)
                    middle_slice = extract_middle_slice(image)

                    if model is not None:
                        ivd_locations = predict_ivd_locations(
                            model, middle_slice, config.device, config.image_size
                        )
                    else:
                        ivd_locations = get_center_fallback_locations()

                    spacing_2d = get_slice_spacing(image)
                    processed_images[cache_key] = (middle_slice, ivd_locations, spacing_2d)
                except Exception as e:
                    logger.debug(f"Error processing {image_file}: {e}")
                    continue
            else:
                middle_slice, ivd_locations, spacing_2d = processed_images[cache_key]

            # Compute crop delta in pixels from mm values
            crop_delta_px = mm_to_pixels(config.crop_delta_mm, spacing_2d)

            # Create crop context for this image
            crop_ctx = CropContext(
                image=middle_slice,
                ivd_locations=ivd_locations,
                crop_size=config.crop_size,
                crop_delta_px=crop_delta_px,
                mode=config.crop_mode,
                last_disc_angle_boost=config.last_disc_angle_boost,
            )

            for ivd_level, label_row in levels_to_process.items():
                level_idx = ivd_level - 1
                crop = crop_ctx.crop(level_idx)
                if crop is None:
                    continue

                output_filename = f"spider_{patient_id}_{series_type}_L{ivd_level}.png"
                output_path = output_images_path / output_filename
                Image.fromarray(crop).save(output_path)

                records.append(
                    ClassificationRecord(
                        image_path=f"images/{output_filename}",
                        patient_id=str(patient_id),
                        ivd_level=ivd_level,
                        series_type=series_type,
                        source="spider",
                        pfirrmann_grade=int(label_row.get("Pfirrman grade", 0)),
                        disc_herniation=int(label_row.get("Disc herniation", 0)),
                        disc_narrowing=int(label_row.get("Disc narrowing", 0)),
                        disc_bulging=int(label_row.get("Disc bulging", 0)),
                        spondylolisthesis=int(label_row.get("Spondylolisthesis", 0)),
                        modic=int(label_row.get("Modic", 0)),
                        up_endplate=int(label_row.get("UP endplate", 0)),
                        low_endplate=int(label_row.get("LOW endplate", 0)),
                    )
                )

    return records


def log_dataset_summary(records: list[ClassificationRecord]) -> None:
    """Log summary statistics for the dataset."""
    logger.info("=" * 50)
    logger.info("Classification Dataset Summary")
    logger.info("=" * 50)
    logger.info(f"Total records: {len(records)}")

    source_counts: dict[str, int] = {}
    series_counts: dict[str, int] = {}
    level_counts: dict[int, int] = {}
    grade_counts: dict[int, int] = {}

    for rec in records:
        source_counts[rec.source] = source_counts.get(rec.source, 0) + 1
        series_counts[rec.series_type] = series_counts.get(rec.series_type, 0) + 1
        level_counts[rec.ivd_level] = level_counts.get(rec.ivd_level, 0) + 1
        grade_counts[rec.pfirrmann_grade] = grade_counts.get(rec.pfirrmann_grade, 0) + 1

    logger.info("By source:")
    for source, count in sorted(source_counts.items()):
        logger.info(f"  {source}: {count}")

    logger.info("By series type:")
    for series, count in sorted(series_counts.items()):
        logger.info(f"  {series}: {count}")

    logger.info("By IVD level:")
    for level, count in sorted(level_counts.items()):
        logger.info(f"  L{level}: {count}")

    logger.info("By Pfirrmann grade:")
    for grade, count in sorted(grade_counts.items()):
        logger.info(f"  Grade {grade}: {count}")

    unique_patients = len(set((rec.source, rec.patient_id) for rec in records))
    logger.info(f"Unique patients: {unique_patients}")
    logger.info("=" * 50)


def load_existing_annotations(csv_path: Path) -> list[ClassificationRecord]:
    """Load existing annotations from CSV file.

    Args:
        csv_path: Path to existing annotations CSV.

    Returns:
        List of existing classification records.
    """
    records: list[ClassificationRecord] = []
    if not csv_path.exists():
        return records

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                ClassificationRecord(
                    image_path=row["image_path"],
                    patient_id=row["patient_id"],
                    ivd_level=int(row["ivd_level"]),
                    series_type=row["series_type"],
                    source=row["source"],
                    pfirrmann_grade=int(row["pfirrmann_grade"]),
                    disc_herniation=int(row["disc_herniation"]),
                    disc_narrowing=int(row["disc_narrowing"]),
                    disc_bulging=int(row["disc_bulging"]),
                    spondylolisthesis=int(row["spondylolisthesis"]),
                    modic=int(row["modic"]),
                    up_endplate=int(row["up_endplate"]),
                    low_endplate=int(row["low_endplate"]),
                )
            )

    return records


@dataclass
class ParsedImageInfo:
    """Parsed information from image filename."""

    source: str
    patient_id: str
    series_type: str
    ivd_level: int
    filename: str


def parse_image_filename(filename: str) -> ParsedImageInfo | None:
    """Parse image filename to extract metadata.

    Expected formats:
        - phenikaa_{patient_id}_{series_type}_L{level}.png
        - spider_{patient_id}_{series_type}_L{level}.png

    Args:
        filename: Image filename (without path).

    Returns:
        ParsedImageInfo if parsing successful, None otherwise.
    """
    # Pattern: {source}_{patient_id}_{series_type}_L{level}.png
    pattern = r"^(phenikaa|spider)_(.+)_(sag_t[12])_L(\d)\.png$"
    match = re.match(pattern, filename)

    if not match:
        return None

    return ParsedImageInfo(
        source=match.group(1),
        patient_id=match.group(2),
        series_type=match.group(3),
        ivd_level=int(match.group(4)),
        filename=filename,
    )


def scan_existing_images(images_path: Path) -> list[ParsedImageInfo]:
    """Scan images directory for existing processed images.

    Args:
        images_path: Path to images directory.

    Returns:
        List of parsed image info for all valid images found.
    """
    if not images_path.exists():
        return []

    existing: list[ParsedImageInfo] = []
    for img_file in images_path.glob("*.png"):
        parsed = parse_image_filename(img_file.name)
        if parsed is not None:
            existing.append(parsed)

    return existing


def recover_phenikaa_annotations(
    existing_images: list[ParsedImageInfo],
    labels_path: Path,
) -> list[ClassificationRecord]:
    """Recover annotations for existing Phenikaa images from source labels.

    Args:
        existing_images: List of existing Phenikaa image info.
        labels_path: Path to radiological_labels.csv.

    Returns:
        List of recovered classification records.
    """
    records: list[ClassificationRecord] = []

    if not labels_path.exists():
        logger.warning(f"Cannot recover Phenikaa annotations: {labels_path} not found")
        return records

    patient_labels = _load_phenikaa_labels(labels_path)

    for img_info in existing_images:
        if img_info.source != "phenikaa":
            continue

        patient_id = img_info.patient_id
        ivd_level = img_info.ivd_level

        if patient_id not in patient_labels:
            logger.debug(f"No labels found for patient {patient_id}")
            continue

        if ivd_level not in patient_labels[patient_id]:
            logger.debug(f"No labels found for {patient_id} level {ivd_level}")
            continue

        label_row = patient_labels[patient_id][ivd_level]
        record = _create_classification_record(
            img_info.filename,
            patient_id,
            ivd_level,
            img_info.series_type,
            label_row,
        )
        records.append(record)

    return records


def recover_spider_annotations(
    existing_images: list[ParsedImageInfo],
    labels_path: Path,
) -> list[ClassificationRecord]:
    """Recover annotations for existing SPIDER images from source labels.

    Args:
        existing_images: List of existing SPIDER image info.
        labels_path: Path to radiological_gradings.csv.

    Returns:
        List of recovered classification records.
    """
    records: list[ClassificationRecord] = []

    if not labels_path.exists():
        logger.warning(f"Cannot recover SPIDER annotations: {labels_path} not found")
        return records

    # Load SPIDER labels with level conversion to Phenikaa format
    patient_labels: dict[int, dict[int, dict]] = {}
    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = int(row["Patient"])
            # Convert SPIDER level (1=L5/S1) to Phenikaa level (1=L1/L2)
            ivd_level = convert_spider_to_phenikaa_level(int(row["IVD label"]))
            if patient_id not in patient_labels:
                patient_labels[patient_id] = {}
            patient_labels[patient_id][ivd_level] = row

    for img_info in existing_images:
        if img_info.source != "spider":
            continue

        try:
            patient_id = int(img_info.patient_id)
        except ValueError:
            logger.debug(f"Invalid SPIDER patient ID: {img_info.patient_id}")
            continue

        ivd_level = img_info.ivd_level

        if patient_id not in patient_labels:
            logger.debug(f"No labels found for SPIDER patient {patient_id}")
            continue

        if ivd_level not in patient_labels[patient_id]:
            logger.debug(f"No labels for SPIDER {patient_id} level {ivd_level}")
            continue

        label_row = patient_labels[patient_id][ivd_level]
        records.append(
            ClassificationRecord(
                image_path=f"images/{img_info.filename}",
                patient_id=str(patient_id),
                ivd_level=ivd_level,
                series_type=img_info.series_type,
                source="spider",
                pfirrmann_grade=int(label_row.get("Pfirrman grade", 0)),
                disc_herniation=int(label_row.get("Disc herniation", 0)),
                disc_narrowing=int(label_row.get("Disc narrowing", 0)),
                disc_bulging=int(label_row.get("Disc bulging", 0)),
                spondylolisthesis=int(label_row.get("Spondylolisthesis", 0)),
                modic=int(label_row.get("Modic", 0)),
                up_endplate=int(label_row.get("UP endplate", 0)),
                low_endplate=int(label_row.get("LOW endplate", 0)),
            )
        )

    return records


class ClassificationDatasetProcessor(BaseProcessor[ClassificationDatasetConfig]):
    """Processor for creating classification dataset from Phenikaa and SPIDER.

    Combines data from multiple sources with IVD-level cropping based on
    localization model predictions.
    """

    def __init__(self, config: ClassificationDatasetConfig) -> None:
        """Initialize processor with configuration.

        Args:
            config: Classification dataset configuration.
        """
        super().__init__(config)
        # Initialize logging
        setup_logger(verbose=config.verbose)
        if config.enable_file_log:
            add_file_log(config.log_path)

    def process(self) -> ProcessingResult:
        """Execute classification dataset creation pipeline.

        Uses filesystem-based detection for continuous integration:
        - Scans existing images from disk (not from CSV)
        - Recovers annotations for existing images from source labels
        - Only processes new images that don't exist on disk
        - Combines recovered + new annotations

        Returns:
            ProcessingResult with dataset statistics.
        """
        warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
        self.on_process_begin()

        csv_path = self.config.output_path / "annotations.csv"
        output_images_path = self.config.output_path / "images"
        output_images_path.mkdir(parents=True, exist_ok=True)

        # Scan filesystem for existing images (more reliable than CSV)
        existing_images = scan_existing_images(output_images_path)
        existing_image_paths: set[str] = set()
        recovered_records: list[ClassificationRecord] = []

        if existing_images and self.config.append_to_existing:
            logger.info(f"Found {len(existing_images)} existing images on disk")
            existing_image_paths = {f"images/{img.filename}" for img in existing_images}

            # Recover annotations from source labels for existing images
            phenikaa_labels_path = self.config.phenikaa_path / "radiological_labels.csv"
            spider_labels_path = self.config.spider_path / "radiological_gradings.csv"

            phenikaa_recovered = recover_phenikaa_annotations(
                existing_images, phenikaa_labels_path
            )
            spider_recovered = recover_spider_annotations(
                existing_images, spider_labels_path
            )

            recovered_records = phenikaa_recovered + spider_recovered
            logger.info(
                f"Recovered annotations for {len(recovered_records)} existing images "
                f"({len(phenikaa_recovered)} Phenikaa, {len(spider_recovered)} SPIDER)"
            )

            # Warn about orphan images (images without matching labels)
            orphan_count = len(existing_images) - len(recovered_records)
            if orphan_count > 0:
                logger.warning(
                    f"{orphan_count} existing images have no matching labels "
                    "(labels may have been removed from source)"
                )

        model: torch.nn.Module | None = None
        if self.config.localization_model_path is not None:
            logger.info(
                f"Loading localization model from: {self.config.localization_model_path}"
            )
            model = load_localization_model(
                self.config.localization_model_path,
                self.config.model_variant,
                self.config.device,
            )
        else:
            logger.warning(
                "No localization model provided, using center fallback locations"
            )

        new_records: list[ClassificationRecord] = []

        if self.config.include_phenikaa:
            logger.info("Processing Phenikaa dataset...")
            phenikaa_records = process_phenikaa(
                self.config, output_images_path, model, existing_image_paths
            )
            new_records.extend(phenikaa_records)
            logger.info(f"Processed {len(phenikaa_records)} new Phenikaa records")

        if self.config.include_spider:
            logger.info("Processing SPIDER dataset...")
            spider_records = process_spider(
                self.config, output_images_path, model, existing_image_paths
            )
            new_records.extend(spider_records)
            logger.info(f"Processed {len(spider_records)} new SPIDER records")

        # Combine recovered and new records
        all_records = recovered_records + new_records

        fieldnames = list(ClassificationRecord.model_fields.keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in all_records:
                writer.writerow(rec.model_dump())

        log_dataset_summary(all_records)
        logger.info(f"Dataset saved to: {self.config.output_path}")
        logger.info(f"Annotations CSV: {csv_path}")
        logger.info(f"Images directory: {output_images_path}")
        if recovered_records:
            logger.info(
                f"Total: {len(all_records)} records "
                f"({len(recovered_records)} recovered, {len(new_records)} new)"
            )

        result = ProcessingResult(
            num_samples=len(all_records),
            output_path=self.config.output_path,
            summary=f"Created {len(all_records)} classification samples ({len(new_records)} new, {len(recovered_records)} recovered)",
        )

        self.on_process_end(result)
        return result


def main(config: ClassificationDatasetConfig) -> None:
    """Create classification dataset.

    Convenience wrapper around ClassificationDatasetProcessor for backward compatibility.

    Args:
        config: Dataset configuration.
    """
    processor = ClassificationDatasetProcessor(config)
    result = processor.process()
    logger.info(result.summary)

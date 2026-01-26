"""Image processing utilities for classification dataset creation.

Functions for medical image resampling, slice extraction, and IVD region cropping.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image

from spine_vision.io import normalize_to_uint8

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


def resample_to_isotropic(
    image: sitk.Image,
    new_spacing: tuple[float, float, float] = ISOTROPIC_SPACING,
) -> sitk.Image:
    """Resample a SimpleITK image to uniform spacing (square pixels)."""
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
    image = sitk.DICOMOrient(image, "LPI")
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
    """
    oriented = sitk.DICOMOrient(image, "LPI")
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
    - Bottom point: local quadratic extrapolation using last 3 points

    Args:
        ivd_locations: Dictionary mapping level index to (x, y) normalized coordinates.
        image_shape: Image shape (H, W) for denormalization.
        last_disc_angle_boost: Multiplier for rotation angle at L5/S1.

    Returns:
        Dictionary mapping level index to rotation angle in degrees.
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
    distortion.

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

    backbone_name = (
        f"convnext_{variant}"
        if not variant.startswith("v2_")
        else f"convnextv2_{variant[3:]}"
    )

    model = CoordinateRegressor(
        backbone=backbone_name,
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

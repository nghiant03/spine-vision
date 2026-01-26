"""Configuration and record types for classification dataset creation."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, computed_field

from spine_vision.core import BaseConfig
from spine_vision.datasets.classification.cropping import CropMode


class ClassificationDatasetConfig(BaseConfig):
    """Configuration for classification dataset creation."""

    base_path: Path = Path.cwd() / "data"
    """Base data directory."""

    output_name: str = "classification"
    """Output dataset folder name."""

    localization_model_path: Path | None = None
    """Path to trained localization model checkpoint. If None, uses center crop."""

    model_variant: Literal[
        "tiny",
        "small",
        "base",
        "large",
        "xlarge",
        "v2_tiny",
        "v2_small",
        "v2_base",
        "v2_large",
        "v2_huge",
    ] = "base"
    """ConvNext variant for localization model."""

    crop_size: tuple[int, int] = (256, 256)
    """Output size of cropped IVD regions in pixels (H, W)."""

    crop_delta_mm: tuple[float, float, float, float] = (55, 15, 17.5, 20)
    """Crop region deltas (left, right, top, bottom) in millimeters."""

    crop_mode: CropMode = "horizontal"
    """Crop mode: 'horizontal' for axis-aligned, 'rotated' for spinal canal-based."""

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

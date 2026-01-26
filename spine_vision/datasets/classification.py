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
import warnings
from pathlib import Path
from typing import Literal

import torch
from loguru import logger
from pydantic import BaseModel, computed_field
from tqdm.std import TqdmExperimentalWarning

from spine_vision.core import BaseConfig, add_file_log, setup_logger
from spine_vision.datasets.base import BaseProcessor, ProcessingResult
from spine_vision.datasets.image_processing import CropMode, load_localization_model
from spine_vision.datasets.source_processors import (
    IVD_LEVEL_NAMES,
    process_phenikaa,
    process_spider,
    recover_phenikaa_annotations,
    recover_spider_annotations,
    scan_existing_images,
)


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
            summary=(
                f"Created {len(all_records)} classification samples "
                f"({len(new_records)} new, {len(recovered_records)} recovered)"
            ),
        )

        self.on_process_end(result)
        return result


def main(config: ClassificationDatasetConfig) -> None:
    """Create classification dataset.

    Convenience wrapper around ClassificationDatasetProcessor.

    Args:
        config: Dataset configuration.
    """
    processor = ClassificationDatasetProcessor(config)
    result = processor.process()
    logger.info(result.summary)


# Re-export for backwards compatibility
__all__ = [
    "ClassificationDatasetConfig",
    "ClassificationDatasetProcessor",
    "ClassificationRecord",
    "IVD_LEVEL_NAMES",
    "log_dataset_summary",
    "load_existing_annotations",
    "main",
]

"""Classification dataset creation subpackage.

Creates classification dataset for radiological grading prediction.
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

import torch
from loguru import logger
from tqdm.std import TqdmExperimentalWarning

from spine_vision.core import add_file_log, setup_logger
from spine_vision.datasets.base import ProcessingResult
from spine_vision.datasets.classification.config import (
    ClassificationDatasetConfig,
    ClassificationRecord,
)
from spine_vision.datasets.classification.cropping import (
    CropMode,
    load_localization_model,
)
from spine_vision.datasets.classification.phenikaa import process_phenikaa
from spine_vision.datasets.classification.recovery import (
    recover_phenikaa_annotations,
    recover_spider_annotations,
)
from spine_vision.datasets.classification.spider import (
    IVD_LEVEL_NAMES,
    ParsedImageInfo,
    process_spider,
    scan_existing_images,
)


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


def create_classification_dataset(
    config: ClassificationDatasetConfig,
) -> ProcessingResult:
    """Create classification dataset from Phenikaa and SPIDER.

    Uses filesystem-based detection for continuous integration:
    - Scans existing images from disk (not from CSV)
    - Recovers annotations for existing images from source labels
    - Only processes new images that don't exist on disk
    - Combines recovered + new annotations

    Args:
        config: Classification dataset configuration.

    Returns:
        ProcessingResult with dataset statistics.
    """
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

    # Initialize logging
    setup_logger(verbose=config.verbose)
    if config.enable_file_log:
        add_file_log(config.log_path)

    csv_path = config.output_path / "annotations.csv"
    output_images_path = config.output_path / "images"
    output_images_path.mkdir(parents=True, exist_ok=True)

    # Scan filesystem for existing images (more reliable than CSV)
    existing_images = scan_existing_images(output_images_path)
    existing_image_paths: set[str] = set()
    recovered_records: list[ClassificationRecord] = []

    if existing_images and config.append_to_existing:
        logger.info(f"Found {len(existing_images)} existing images on disk")
        existing_image_paths = {f"images/{img.filename}" for img in existing_images}

        # Recover annotations from source labels for existing images
        phenikaa_labels_path = config.phenikaa_path / "radiological_labels.csv"
        spider_labels_path = config.spider_path / "radiological_gradings.csv"

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
    if config.localization_model_path is not None:
        logger.info(
            f"Loading localization model from: {config.localization_model_path}"
        )
        model = load_localization_model(
            config.localization_model_path,
            config.model_variant,
            config.device,
        )
    else:
        logger.warning(
            "No localization model provided, using center fallback locations"
        )

    new_records: list[ClassificationRecord] = []

    if config.include_phenikaa:
        logger.info("Processing Phenikaa dataset...")
        phenikaa_records = process_phenikaa(
            config, output_images_path, model, existing_image_paths
        )
        new_records.extend(phenikaa_records)
        logger.info(f"Processed {len(phenikaa_records)} new Phenikaa records")

    if config.include_spider:
        logger.info("Processing SPIDER dataset...")
        spider_records = process_spider(
            config, output_images_path, model, existing_image_paths
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
    logger.info(f"Dataset saved to: {config.output_path}")
    logger.info(f"Annotations CSV: {csv_path}")
    logger.info(f"Images directory: {output_images_path}")
    if recovered_records:
        logger.info(
            f"Total: {len(all_records)} records "
            f"({len(recovered_records)} recovered, {len(new_records)} new)"
        )

    return ProcessingResult(
        num_samples=len(all_records),
        output_path=config.output_path,
        summary=(
            f"Created {len(all_records)} classification samples "
            f"({len(new_records)} new, {len(recovered_records)} recovered)"
        ),
    )


__all__ = [
    # Config and types
    "ClassificationDatasetConfig",
    "ClassificationRecord",
    "CropMode",
    "ParsedImageInfo",
    # Main function
    "create_classification_dataset",
    # Utilities
    "IVD_LEVEL_NAMES",
    "load_existing_annotations",
    "log_dataset_summary",
    "scan_existing_images",
]

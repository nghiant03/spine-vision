"""Source dataset processors for Phenikaa and SPIDER.

Functions for processing source datasets and creating classification records.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger
from PIL import Image
from tqdm.rich import tqdm

from spine_vision.datasets.image_processing import (
    CropContext,
    extract_middle_slice,
    get_center_fallback_locations,
    get_slice_spacing,
    mm_to_pixels,
    predict_ivd_locations,
    resample_to_isotropic,
)
from spine_vision.io import read_medical_image

if TYPE_CHECKING:
    from spine_vision.datasets.classification import (
        ClassificationDatasetConfig,
        ClassificationRecord,
    )


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
    source: str = "phenikaa",
) -> "ClassificationRecord":
    """Create a classification record from label row.

    Args:
        output_filename: Output image filename
        patient_id: Patient identifier
        ivd_level: IVD level (1-5)
        series_type: Series type (sag_t1 or sag_t2)
        label_row: CSV row with labels
        source: Data source name

    Returns:
        ClassificationRecord instance
    """
    from spine_vision.datasets.classification import ClassificationRecord

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
        source=source,
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
    config: "ClassificationDatasetConfig",
    output_images_path: Path,
    model: torch.nn.Module | None,
    existing_image_paths: set[str] | None = None,
) -> list["ClassificationRecord"]:
    """Process Phenikaa dataset.

    Args:
        config: Dataset configuration.
        output_images_path: Output directory for images.
        model: Optional localization model.
        existing_image_paths: Set of existing image paths to skip (for appending).

    Returns:
        List of classification records.
    """
    from spine_vision.datasets.classification import ClassificationRecord

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

            for series_pattern, series_type in [
                ("sag t1", "sag_t1"),
                ("sag t2", "sag_t2"),
            ]:
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
    config: "ClassificationDatasetConfig",
    output_images_path: Path,
    model: torch.nn.Module | None,
    existing_image_paths: set[str] | None = None,
) -> list["ClassificationRecord"]:
    """Process SPIDER dataset.

    Args:
        config: Dataset configuration.
        output_images_path: Output directory for images.
        model: Optional localization model.
        existing_image_paths: Set of existing image paths to skip (for appending).

    Returns:
        List of classification records.
    """
    from spine_vision.datasets.classification import ClassificationRecord

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
    processed_images: dict[
        tuple[int, str], tuple[np.ndarray, dict[int, tuple[float, float]], tuple[float, float]]
    ] = {}

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
) -> list["ClassificationRecord"]:
    """Recover annotations for existing Phenikaa images from source labels.

    Args:
        existing_images: List of existing Phenikaa image info.
        labels_path: Path to radiological_labels.csv.

    Returns:
        List of recovered classification records.
    """
    from spine_vision.datasets.classification import ClassificationRecord

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
) -> list["ClassificationRecord"]:
    """Recover annotations for existing SPIDER images from source labels.

    Args:
        existing_images: List of existing SPIDER image info.
        labels_path: Path to radiological_gradings.csv.

    Returns:
        List of recovered classification records.
    """
    from spine_vision.datasets.classification import ClassificationRecord

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

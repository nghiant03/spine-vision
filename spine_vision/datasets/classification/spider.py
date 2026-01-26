"""SPIDER source dataset processing for classification dataset creation."""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image
from tqdm.rich import tqdm

from spine_vision.datasets.classification.config import (
    ClassificationDatasetConfig,
    ClassificationRecord,
)
from spine_vision.datasets.classification.cropping import (
    CropContext,
    extract_middle_slice,
    get_center_fallback_locations,
    get_slice_spacing,
    mm_to_pixels,
    predict_ivd_locations,
    resample_to_isotropic,
)
from spine_vision.io import read_medical_image


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
    processed_images: dict[
        tuple[int, str],
        tuple[np.ndarray, dict[int, tuple[float, float]], tuple[float, float]],
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
                    processed_images[cache_key] = (
                        middle_slice,
                        ivd_locations,
                        spacing_2d,
                    )
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


# IVD level mapping (index 0-4 corresponds to L1/L2 to L5/S1)
IVD_LEVEL_NAMES = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]


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

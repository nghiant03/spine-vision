"""Phenikaa source dataset processing for classification dataset creation."""

import csv
from pathlib import Path

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
) -> ClassificationRecord:
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
                    output_filename = (
                        f"phenikaa_{patient_id}_{series_type}_L{ivd_level}.png"
                    )
                    if f"images/{output_filename}" not in existing_image_paths:
                        levels_to_process[ivd_level] = label_row

                if not levels_to_process:
                    logger.debug(
                        f"Skipping {patient_id}/{series_type}: all levels exist"
                    )
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

                    output_filename = (
                        f"phenikaa_{patient_id}_{series_type}_L{ivd_level}.png"
                    )
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

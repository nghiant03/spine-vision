"""Annotation recovery utilities for classification dataset creation.

Functions for recovering annotations from existing images using source label files.
"""

import csv
from pathlib import Path

from loguru import logger

from spine_vision.datasets.classification.config import ClassificationRecord
from spine_vision.datasets.classification.phenikaa import _create_classification_record
from spine_vision.datasets.classification.spider import (
    ParsedImageInfo,
    convert_spider_to_phenikaa_level,
)


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

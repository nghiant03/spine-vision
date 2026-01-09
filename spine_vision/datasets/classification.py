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
from typing import Annotated, Literal

import numpy as np
import SimpleITK as sitk
import torch
import tyro
from loguru import logger
from PIL import Image
from pydantic import BaseModel, computed_field
from tqdm.rich import tqdm
from tqdm.std import TqdmExperimentalWarning

from spine_vision.core.logging import setup_logger
from spine_vision.io import normalize_to_uint8,  read_medical_image


class ClassificationDatasetConfig(BaseModel):
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

    crop_size: tuple[int, int] = (128, 128)
    """Output size of cropped IVD regions in pixels (H, W)."""

    crop_delta: tuple[int, int, int, int] = (96, 32, 64, 64)
    """Crop region deltas (left, right, top, bottom) in pixels. Used when crop_delta_mm is None."""

    crop_delta_mm: tuple[float, float, float, float] | None = None
    """Crop region deltas (left, right, top, bottom) in millimeters. Takes precedence over crop_delta."""

    image_size: tuple[int, int] = (224, 224)
    """Input image size for localization model (H, W)."""

    include_phenikaa: bool = True
    """Include Phenikaa dataset."""

    include_spider: bool = True
    """Include SPIDER dataset."""

    append_to_existing: bool = True
    """If output directory exists, append new data to existing annotations."""

    device: str = "cuda:0"
    """Device for model inference."""

    verbose: Annotated[bool, tyro.conf.arg(aliases=["-v"])] = False
    """Enable verbose logging."""

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
        return self.base_path / "processed" / self.output_name


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

def resample_to_isotropic(image: sitk.Image, new_spacing: tuple[float, float, float]=(1.0, 1.0, 1.0)) -> sitk.Image:
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
    
    resampler.SetInterpolator(sitk.sitkBSpline)
    
    return resampler.Execute(image)

def extract_middle_slice(image: sitk.Image) -> np.ndarray:
    """Extract middle sagittal slice from 3D image.

    Args:
        image: SimpleITK 3D image.

    Returns:
        Tuple of (2D numpy array of the middle slice, (row_spacing, col_spacing) in mm).
    """
    image = sitk.DICOMOrient(image, 'LPI')
    arr = sitk.GetArrayFromImage(image)

    if arr.ndim == 2:
        # 2D image: spacing is (x, y)
        return arr

    mid_idx = arr.shape[2] // 2
    return arr[:, :, mid_idx]


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


def crop_region(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    crop_size: tuple[int, int],
    crop_delta: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop IVD region centered at predicted coordinates using pixel deltas.

    Args:
        image: 2D grayscale image.
        center_x: Relative x coordinate (0-1).
        center_y: Relative y coordinate (0-1).
        crop_size: Output crop size in pixels (H, W).
        crop_delta: Crop deltas (left, right, top, bottom) from center in pixels.

    Returns:
        Cropped image region resized to crop_size.
    """
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size

    cx = int(center_x * w)
    cy = int(center_y * h)

    left, right, top, bottom = crop_delta

    x1 = max(0, cx - left)
    x2 = min(w, cx + right)
    y1 = max(0, cy - top)
    y2 = min(h, cy + bottom)

    crop = image[y1:y2, x1:x2]
    crop_uint8 = normalize_to_uint8(crop)

    if crop_uint8.shape[0] != crop_h or crop_uint8.shape[1] != crop_w:
        pil_crop = Image.fromarray(crop_uint8)
        pil_crop = pil_crop.resize((crop_w, crop_h), Image.Resampling.BILINEAR)
        crop_uint8 = np.array(pil_crop)

    return crop_uint8

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
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    pil_img = Image.fromarray(normalize_to_uint8(image)).convert("RGB")
    tensor = transform(pil_img).unsqueeze(0).to(device)

    predictions = {}
    with torch.no_grad():
        for level_idx in range(5):
            level_tensor = torch.tensor([level_idx], device=device)
            pred = model(tensor, level_tensor)
            pred_np = pred.cpu().numpy()[0]
            predictions[level_idx] = (float(pred_np[0]), float(pred_np[1]))

    return predictions


def get_center_fallback_locations() -> dict[int, tuple[float, float]]:
    """Get approximate center locations for IVD levels as fallback.

    These are rough estimates for L1/L2 to L5/S1 in a typical sagittal view.

    Returns:
        Dictionary mapping level index to (x, y) coordinates.
    """
    return {
        0: (0.5, 0.25),
        1: (0.5, 0.35),
        2: (0.5, 0.45),
        3: (0.5, 0.55),
        4: (0.5, 0.65),
    }


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

    patient_labels: dict[str, dict[int, dict]] = {}
    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = row["Patient ID"]
            ivd_level = int(row["IVD label"])
            if patient_id not in patient_labels:
                patient_labels[patient_id] = {}
            patient_labels[patient_id][ivd_level] = row

    for patient_id, levels in tqdm(
        patient_labels.items(), desc="Processing Phenikaa", unit="patient"
    ):
        try:
            patient_dir = images_path / patient_id

            if not patient_dir.exists():
                logger.debug(f"Patient directory not found: {patient_dir}")
                continue

            for series_pattern, series_type in [("sag t1", "sag_t1"), ("sag t2", "sag_t2")]:
                # Case-insensitive folder matching (handles "SAG T1", "Sag T1", "SagT1", etc.)
                series_dir = None
                for subdir in patient_dir.iterdir():
                    if subdir.is_dir() and subdir.name.lower().replace(" ", "") == series_pattern.replace(" ", ""):
                        series_dir = subdir
                        break
                if series_dir is None:
                    continue

                try:
                    image = read_medical_image(series_dir)
                    image = resample_to_isotropic(image, new_spacing=(0.5, 0.5, 0.5))
                    middle_slice = extract_middle_slice(image)
                except Exception as e:
                    logger.debug(f"Error reading {series_dir}: {e}")
                    continue

                if model is not None:
                    ivd_locations = predict_ivd_locations(
                        model, middle_slice, config.device, config.image_size
                    )
                else:
                    ivd_locations = get_center_fallback_locations()

                # Compute crop delta in pixels (from mm or direct pixel values)
                spacing = image.GetSpacing()
                if config.crop_delta_mm is not None:
                    crop_delta_px = mm_to_pixels(config.crop_delta_mm, spacing)
                else:
                    crop_delta_px = config.crop_delta

                for ivd_level, label_row in levels.items():
                    if ivd_level < 1 or ivd_level > 5:
                        continue

                    level_idx = ivd_level - 1
                    if level_idx not in ivd_locations:
                        continue

                    # Skip if already exists
                    output_filename = f"phenikaa_{patient_id}_{series_type}_L{ivd_level}.png"
                    if f"images/{output_filename}" in existing_image_paths:
                        logger.debug(f"Skipping existing: {output_filename}")
                        continue

                    center_x, center_y = ivd_locations[level_idx]

                    crop = crop_region(
                        middle_slice, center_x, center_y, config.crop_size, crop_delta_px
                    )

                    output_path = output_images_path / output_filename

                    Image.fromarray(crop).save(output_path)

                    modic_value = 0
                    for i in range(4):
                        if label_row.get(f"Modic_{i}", "0") == "1":
                            modic_value = i
                            break

                    records.append(
                        ClassificationRecord(
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
                    )
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
            ivd_level = int(row["IVD label"])
            if patient_id not in patient_labels:
                patient_labels[patient_id] = {}
            patient_labels[patient_id][ivd_level] = row

    # Cache: key -> (middle_slice, ivd_locations, spacing)
    processed_images: dict[tuple[int, str], tuple[np.ndarray, dict[int, tuple[float, float]], tuple[float, float]]] = {}

    for patient_id, levels in tqdm(
        patient_labels.items(), desc="Processing SPIDER", unit="patient"
    ):
        for series_suffix, series_type in [("t1", "sag_t1"), ("t2", "sag_t2")]:
            image_file = images_path / f"{patient_id}_{series_suffix}.mha"
            if not image_file.exists():
                continue

            cache_key = (patient_id, series_type)
            if cache_key not in processed_images:
                try:
                    image = read_medical_image(image_file)
                    image = resample_to_isotropic(image, new_spacing=(0.5, 0.5, 0.5))
                    middle_slice = extract_middle_slice(image)

                    if model is not None:
                        ivd_locations = predict_ivd_locations(
                            model, middle_slice, config.device, config.image_size
                        )
                    else:
                        ivd_locations = get_center_fallback_locations()

                    spacing = image.GetSpacing()
                    processed_images[cache_key] = (middle_slice, ivd_locations, spacing)
                except Exception as e:
                    logger.debug(f"Error processing {image_file}: {e}")
                    continue
            else:
                middle_slice, ivd_locations, spacing = processed_images[cache_key]

            # Compute crop delta in pixels (from mm or direct pixel values)
            if config.crop_delta_mm is not None:
                crop_delta_px = mm_to_pixels(config.crop_delta_mm, spacing)
            else:
                crop_delta_px = config.crop_delta

            for ivd_level, label_row in levels.items():
                if ivd_level < 1 or ivd_level > 5:
                    continue

                level_idx = ivd_level - 1
                if level_idx not in ivd_locations:
                    continue

                # Skip if already exists
                output_filename = f"spider_{patient_id}_{series_type}_L{ivd_level}.png"
                if f"images/{output_filename}" in existing_image_paths:
                    logger.debug(f"Skipping existing: {output_filename}")
                    continue

                center_x, center_y = ivd_locations[level_idx]

                crop = crop_region(
                    middle_slice, center_x, center_y, config.crop_size, crop_delta_px
                )
                crop_uint8 = normalize_to_uint8(crop)

                output_path = output_images_path / output_filename

                Image.fromarray(crop_uint8).save(output_path)

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


def main(config: ClassificationDatasetConfig) -> None:
    """Create classification dataset.

    Args:
        config: Dataset configuration.
    """
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    setup_logger(verbose=config.verbose)

    csv_path = config.output_path / "annotations.csv"
    output_images_path = config.output_path / "images"

    # Check for existing dataset and load if appending
    existing_records: list[ClassificationRecord] = []
    existing_image_paths: set[str] = set()

    if config.append_to_existing and csv_path.exists():
        logger.info(f"Found existing dataset at: {config.output_path}")
        existing_records = load_existing_annotations(csv_path)
        existing_image_paths = {rec.image_path for rec in existing_records}
        logger.info(f"Loaded {len(existing_records)} existing records")

    output_images_path.mkdir(parents=True, exist_ok=True)

    model: torch.nn.Module | None = None
    if config.localization_model_path is not None:
        logger.info(f"Loading localization model from: {config.localization_model_path}")
        model = load_localization_model(
            config.localization_model_path,
            config.model_variant,
            config.device,
        )
    else:
        logger.warning("No localization model provided, using center fallback locations")

    new_records: list[ClassificationRecord] = []

    if config.include_phenikaa:
        logger.info("Processing Phenikaa dataset...")
        phenikaa_records = process_phenikaa(config, output_images_path, model, existing_image_paths)
        new_records.extend(phenikaa_records)
        logger.info(f"Processed {len(phenikaa_records)} new Phenikaa records")

    if config.include_spider:
        logger.info("Processing SPIDER dataset...")
        spider_records = process_spider(config, output_images_path, model, existing_image_paths)
        new_records.extend(spider_records)
        logger.info(f"Processed {len(spider_records)} new SPIDER records")

    # Combine existing and new records
    all_records = existing_records + new_records

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
    if existing_records:
        logger.info(f"Existing records: {len(existing_records)}, New records: {len(new_records)}")

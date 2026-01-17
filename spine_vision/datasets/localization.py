"""Create combined IVD coordinates dataset for training.

Generates images and CSV for training ConvNext models to predict IVD-spinal canal
intersection points from RSNA and Lumbar Coords data.

Dataset structure:
    output_path/
    ├── images/           # All images (PNG/JPG)
    └── annotations.csv   # CSV with columns: image_path, level, relative_x, relative_y, series_type, source
"""

import shutil
import warnings
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from loguru import logger
from PIL import Image
from pydantic import BaseModel, computed_field
from tqdm.rich import tqdm
from tqdm.std import TqdmExperimentalWarning

from spine_vision.core import BaseConfig, add_file_log, setup_logger
from spine_vision.datasets.base import BaseProcessor, ProcessingResult
from spine_vision.datasets.rsna import get_series_type, load_series_mapping
from spine_vision.io import normalize_to_uint8, write_records_csv


class LocalizationDatasetConfig(BaseConfig):
    """Configuration for localization dataset creation."""

    base_path: Path = Path.cwd() / "data"
    """Base data directory."""

    output_name: str = "localization"
    """Output dataset folder name."""

    include_neural_foraminal: bool = True
    """Include Neural Foraminal Narrowing annotations (Sag T1/T2)."""

    include_spinal_canal: bool = True
    """Include Spinal Canal Stenosis annotations (Sag T2)."""

    skip_invalid_instances: bool = True
    """Skip records with invalid instance numbers (instance_number < 0)."""

    @computed_field
    @property
    def lumbar_coords_path(self) -> Path:
        """Path to Lumbar Coords dataset."""
        return self.base_path / "raw" / "Lumbar Coords"

    @computed_field
    @property
    def rsna_path(self) -> Path:
        """Path to RSNA dataset."""
        return self.base_path / "raw" / "RSNA"

    @computed_field
    @property
    def output_path(self) -> Path:
        """Output dataset path."""
        path = self.base_path / "processed" / self.output_name
        path.mkdir(parents=True, exist_ok=True)
        return path


class AnnotationRecord(BaseModel):
    """A single annotation record."""

    image_path: str
    level: str
    relative_x: float
    relative_y: float
    series_type: str
    source: str


def process_lumbar_coords_pretrain(
    coords_csv_path: Path,
    data_path: Path,
    output_images_path: Path,
) -> list[AnnotationRecord]:
    """Process Lumbar Coords pretrain data (spider, lsd, osf, tseg).

    Sources contain different modalities: Spider/LSD (T2), OSF (T1), Tseg (CT).

    Args:
        coords_csv_path: Path to coords_pretrain.csv.
        data_path: Path to data folder containing processed images.
        output_images_path: Output directory for images.

    Returns:
        List of annotation records.
    """
    import csv

    records: list[AnnotationRecord] = []
    source_to_folder = {
        "spider": "processed_spider_jpgs",
        "lsd": "processed_lsd_jpgs",
        "osf": "processed_osf_jpgs",
        "tseg": "processed_tseg_jpgs",
    }
    source_to_npy_folder = {
        "spider": None,
        "lsd": "processed_lsd",
        "osf": "processed_osf",
        "tseg": "processed_tseg",
    }
    source_to_series_type = {
        "spider": "sag_t2",
        "lsd": "sag_t2",
        "osf": "sag_t1",
        "tseg": "ct",
    }

    processed_files: set[str] = set()

    with open(coords_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            source = row["source"]
            level = row["level"]
            rel_x = float(row["relative_x"])
            rel_y = float(row["relative_y"])

            folder = source_to_folder.get(source)
            if folder is None:
                logger.warning(f"Unknown source: {source}")
                continue

            series_type = source_to_series_type[source]
            output_filename = f"pretrain_{source}_{filename}"
            if not output_filename.endswith((".jpg", ".png")):
                output_filename = output_filename.replace(".npy", ".png")

            src_img_path = data_path / folder / filename
            output_path = output_images_path / output_filename

            if output_filename not in processed_files:
                if src_img_path.exists():
                    shutil.copy(src_img_path, output_path)
                    processed_files.add(output_filename)
                else:
                    npy_folder = source_to_npy_folder.get(source)
                    if npy_folder:
                        npy_filename = filename.replace(".jpg", ".npy")
                        npy_path = data_path / npy_folder / npy_filename
                        if npy_path.exists():
                            arr = np.load(npy_path)
                            arr = normalize_to_uint8(arr)
                            img = Image.fromarray(arr)
                            img.save(output_path)
                            processed_files.add(output_filename)
                        else:
                            logger.warning(
                                f"File not found: {src_img_path} or {npy_path}"
                            )
                            continue
                    else:
                        logger.warning(f"File not found: {src_img_path}")
                        continue

            records.append(
                AnnotationRecord(
                    image_path=f"images/{output_filename}",
                    level=level,
                    relative_x=rel_x,
                    relative_y=rel_y,
                    series_type=series_type,
                    source=f"pretrain_{source}",
                )
            )

    return records


def process_rsna_improved(
    coords_csv_path: Path,
    series_desc_path: Path,
    rsna_images_path: Path,
    output_images_path: Path,
    config: LocalizationDatasetConfig,
) -> list[AnnotationRecord]:
    """Process RSNA improved coordinates.

    Filters to Sagittal T1 and Sagittal T2 only (excludes Axial T2).

    Condition types and their series:
    - Spinal Canal Stenosis: Sagittal T2/STIR (IVD-canal intersection points)
    - Neural Foraminal Narrowing: Sagittal T1 (Left/Right side annotations)
    - Subarticular Stenosis: Axial T2 (excluded)

    Args:
        coords_csv_path: Path to coords_rsna_improved.csv.
        series_desc_path: Path to train_series_descriptions.csv.
        rsna_images_path: Path to train_images directory.
        output_images_path: Output directory for images.
        config: Dataset configuration.

    Returns:
        List of annotation records.
    """
    import csv

    records: list[AnnotationRecord] = []
    series_mapping = load_series_mapping(series_desc_path)
    processed_images: set[str] = set()

    with open(coords_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in tqdm(rows, desc="Processing RSNA", unit="record"):
        series_id = int(row["series_id"])
        study_id = int(row["study_id"])
        instance_number = int(row["instance_number"])
        rel_x = float(row["relative_x"])
        rel_y = float(row["relative_y"])
        level = row["level"]
        condition = row["condition"]

        if "Subarticular" in condition:
            continue

        is_spinal_canal = "Spinal Canal" in condition
        is_neural_foraminal = "Neural Foraminal" in condition

        if is_spinal_canal and not config.include_spinal_canal:
            continue
        if is_neural_foraminal and not config.include_neural_foraminal:
            continue

        if config.skip_invalid_instances and instance_number < 0:
            continue

        series_type_str = get_series_type(series_id, study_id, series_mapping)
        if series_type_str is None:
            logger.debug(f"Series {series_id} not found for study {study_id}")
            continue

        if "Sagittal T1" in series_type_str:
            series_type = "sag_t1"
        elif "Sagittal T2" in series_type_str:
            series_type = "sag_t2"
        else:
            continue

        dcm_path = (
            rsna_images_path / str(study_id) / str(series_id) / f"{instance_number}.dcm"
        )
        if not dcm_path.exists():
            logger.debug(f"DICOM not found: {dcm_path}")
            continue

        output_filename = f"rsna_{study_id}_{series_id}_{instance_number}.png"
        output_path = output_images_path / output_filename

        if output_filename not in processed_images:
            try:
                image = sitk.ReadImage(str(dcm_path))
                arr = sitk.GetArrayFromImage(image)
                if arr.ndim == 3:
                    arr = arr[0]
                arr = normalize_to_uint8(arr)
                img = Image.fromarray(arr)
                img.save(output_path)
                processed_images.add(output_filename)
            except Exception as e:
                logger.error(f"Error processing {dcm_path}: {e}")
                continue

        records.append(
            AnnotationRecord(
                image_path=f"images/{output_filename}",
                level=level,
                relative_x=rel_x,
                relative_y=rel_y,
                series_type=series_type,
                source="rsna",
            )
        )

    return records


def log_dataset_summary(records: list[AnnotationRecord]) -> None:
    """Log summary statistics for the dataset.

    Args:
        records: List of annotation records.
    """
    logger.info("=" * 50)
    logger.info("Dataset Creation Summary")
    logger.info("=" * 50)
    logger.info(f"Total annotation records: {len(records)}")

    source_counts: dict[str, int] = {}
    series_counts: dict[str, int] = {}
    level_counts: dict[str, int] = {}
    for rec in records:
        source_counts[rec.source] = source_counts.get(rec.source, 0) + 1
        series_counts[rec.series_type] = series_counts.get(rec.series_type, 0) + 1
        level_counts[rec.level] = level_counts.get(rec.level, 0) + 1

    logger.info("By source:")
    for source, count in sorted(source_counts.items()):
        logger.info(f"  {source}: {count}")

    logger.info("By series type:")
    for series, count in sorted(series_counts.items()):
        logger.info(f"  {series}: {count}")

    logger.info("By level:")
    for level, count in sorted(level_counts.items()):
        logger.info(f"  {level}: {count}")

    unique_images = len(set(rec.image_path for rec in records))
    logger.info(f"Unique images: {unique_images}")
    logger.info("=" * 50)


class LocalizationDatasetProcessor(BaseProcessor[LocalizationDatasetConfig]):
    """Processor for creating localization dataset.

    Combines Lumbar Coords pretrain data and RSNA improved coordinates
    for training localization models.
    """

    def __init__(self, config: LocalizationDatasetConfig) -> None:
        """Initialize processor with configuration.

        Args:
            config: Localization dataset configuration.
        """
        super().__init__(config)
        # Initialize logging
        setup_logger(verbose=config.verbose)
        if config.enable_file_log:
            add_file_log(config.log_path)

    def process(self) -> ProcessingResult:
        """Execute IVD coordinates dataset creation pipeline.

        Returns:
            ProcessingResult with dataset statistics.
        """
        warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
        self.on_process_begin()

        output_images_path = self.config.output_path / "images"
        output_images_path.mkdir(parents=True, exist_ok=True)

        all_records: list[AnnotationRecord] = []

        logger.info("Processing Lumbar Coords pretrain data...")
        pretrain_records = process_lumbar_coords_pretrain(
            coords_csv_path=self.config.lumbar_coords_path / "coords_pretrain.csv",
            data_path=self.config.lumbar_coords_path / "data",
            output_images_path=output_images_path,
        )
        all_records.extend(pretrain_records)
        logger.info(f"Processed {len(pretrain_records)} pretrain annotation records")

        logger.info("Processing RSNA improved coordinates...")
        rsna_records = process_rsna_improved(
            coords_csv_path=self.config.lumbar_coords_path / "coords_rsna_improved.csv",
            series_desc_path=self.config.rsna_path / "train_series_descriptions.csv",
            rsna_images_path=self.config.rsna_path / "train_images",
            output_images_path=output_images_path,
            config=self.config,
        )
        all_records.extend(rsna_records)
        logger.info(f"Processed {len(rsna_records)} RSNA annotation records")

        csv_path = self.config.output_path / "annotations.csv"
        write_records_csv(all_records, csv_path)

        log_dataset_summary(all_records)
        logger.info(f"Dataset saved to: {self.config.output_path}")
        logger.info(f"Annotations CSV: {csv_path}")
        logger.info(f"Images directory: {output_images_path}")

        result = ProcessingResult(
            num_samples=len(all_records),
            output_path=self.config.output_path,
            summary=f"Created {len(all_records)} IVD coordinate annotations",
        )

        self.on_process_end(result)
        return result


def main(config: LocalizationDatasetConfig) -> None:
    """Create combined localization dataset.

    Convenience wrapper around LocalizationDatasetProcessor for backward compatibility.

    Args:
        config: Dataset configuration.
    """
    processor = LocalizationDatasetProcessor(config)
    result = processor.process()
    logger.info(result.summary)

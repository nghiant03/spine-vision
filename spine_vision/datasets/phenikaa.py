"""Preprocess medical data: OCR extraction and patient matching."""

import shutil
import warnings
from pathlib import Path
from typing import Annotated

import tyro
from loguru import logger
from pydantic import BaseModel, computed_field
from tqdm.rich import tqdm
from tqdm.std import TqdmExperimentalWarning

from spine_vision.core.logging import setup_logger
from spine_vision.io.tabular import load_tabular_data
from spine_vision.matching.fuzzy import fuzzy_value_extract
from spine_vision.matching.patient import PatientMatcher
from spine_vision.ocr.extraction import DocumentExtractor

NAME_FIELD_PATTERN = "Ho ten nguoi benh"
BIRTHDAY_FIELD_PATTERN = "Ngay sinh"
ONE_HOT_COL = "Modic"


class PreprocessConfig(BaseModel):
    """Configuration for preprocessing pipeline."""

    data_path: Path = Path.cwd() / "data/silver/Phenikaa"
    exclude_files: list[str] = []
    id_col: str = "Patient ID"
    corrupted_ids: list[int] = [
        25001,
        250027783,
        250026093,
        250026925,
        250026665,
        250010269,
    ]
    output_path: Path = Path.cwd() / "data/gold/classification"
    output_table: str = "radiological_labels.csv"
    model_path: Path = Path.cwd() / "weights/ocr"
    detection_model: str = "PP-OCRv5_server_det"
    recognition_model: str = "vgg_transformer"
    report_fuzzy_threshold: float = 80
    image_fuzzy_threshold: float = 85
    use_gpu: Annotated[bool, tyro.conf.arg(aliases=["-g"])] = True
    verbose: Annotated[bool, tyro.conf.arg(aliases=["-v"])] = False
    enable_file_log: bool = False
    log_path: Path = Path.cwd() / "logs"

    @computed_field
    @property
    def image_path(self) -> Path:
        return self.data_path / "images"

    @computed_field
    @property
    def label_path(self) -> Path:
        return self.data_path / "labels"

    @computed_field
    @property
    def report_path(self) -> Path:
        return self.label_path / "reports"

    @computed_field
    @property
    def table_path(self) -> Path:
        return self.label_path / "tables"

    @computed_field
    @property
    def output_table_path(self) -> Path:
        return self.output_path / self.output_table

    @computed_field
    @property
    def output_image_path(self) -> Path:
        return self.output_path / "images"


def main(config: PreprocessConfig) -> None:
    """Run preprocessing pipeline."""
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

    setup_logger(
        verbose=config.verbose,
        enable_file_log=config.enable_file_log,
        log_path=config.log_path,
        log_filename="preprocess.log",
    )

    logger.debug("Started preprocessing.")

    label_data = load_tabular_data(
        table_path=config.table_path,
        exclude_files=config.exclude_files,
        id_col=config.id_col,
        corrupted_ids=config.corrupted_ids,
        one_hot_col=ONE_HOT_COL,
    )

    if label_data.empty:
        logger.info(f"No valid data found at {config.table_path}")
        return

    label_data = label_data.astype(int)
    label_data.to_csv(config.table_path / "full_radiological_gradings.csv", index=False)

    config.output_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Unique Patients: {label_data[config.id_col].nunique()}")

    device = "cuda:0" if config.use_gpu else "cpu"
    logger.info("Loading OCR models.")
    extractor = DocumentExtractor(
        detection_model=config.detection_model,
        recognition_model=config.recognition_model,
        device=device,
        use_gpu=config.use_gpu,
    )

    report_lookup = {path.stem: path for path in config.report_path.rglob("*.png")}

    patient_matcher = PatientMatcher(
        image_path=config.image_path,
        threshold=config.image_fuzzy_threshold,
    )

    unique_ids = label_data[config.id_col].unique()
    matched_ids = []

    for unique_id in tqdm(unique_ids, desc="Processing Patients", unit="id"):
        report_path = report_lookup.get(str(unique_id))
        if not report_path:
            continue

        text_lines = extractor.extract(report_path)
        if not text_lines:
            continue

        patient_name = fuzzy_value_extract(
            text_lines, NAME_FIELD_PATTERN, config.report_fuzzy_threshold, 3
        )
        if not patient_name:
            logger.warning(f"Could not extract name for ID {unique_id}")
            continue

        patient_birthday = fuzzy_value_extract(
            text_lines, BIRTHDAY_FIELD_PATTERN, config.report_fuzzy_threshold, 2
        )
        if not patient_birthday:
            logger.warning(f"Could not extract birthday for ID {unique_id}")
            continue

        best_folder = patient_matcher.match(patient_name, patient_birthday)
        if best_folder:
            dest = config.output_image_path / str(unique_id)
            shutil.copytree(best_folder, dest, dirs_exist_ok=True)
            logger.info(f"Copied {best_folder.name} -> {dest}")
            matched_ids.append(unique_id)
        else:
            logger.warning(f"No matching folder for '{patient_name}' (ID: {unique_id})")

    label_data = label_data[label_data[config.id_col].isin(matched_ids)]
    label_data.to_csv(config.output_table_path, index=False)
    logger.info(f"Saved table to {config.output_table_path}")

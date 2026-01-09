"""Phenikaa dataset creation: OCR extraction and patient matching.

Supports two report formats:
1. ID-named reports (e.g., 250010139.png): Extract patient name/birthday via OCR,
   then match to image folder by fuzzy name matching.
2. Patient-named reports (e.g., NGUYEN_VAN_SON_20250718.pdf): Parse patient name
   from filename, extract ID via OCR, then match folder directly by name.
"""

import re
import shutil
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro
from loguru import logger
from pydantic import BaseModel, computed_field
from tqdm.rich import tqdm
from tqdm.std import TqdmExperimentalWarning

from spine_vision.core import add_file_log, setup_logger
from spine_vision.datasets.phenikaa.matching import PatientMatcher, fuzzy_value_extract
from spine_vision.datasets.phenikaa.ocr import (
    SUPPORTED_EXTENSIONS,
    DocumentExtractor,
)
from spine_vision.io import load_tabular_data

# OCR field patterns
NAME_FIELD_PATTERN = "Ho ten nguoi benh"
BIRTHDAY_FIELD_PATTERN = "Ngay sinh"
ID_FIELD_PATTERN = "So phieu"
ONE_HOT_COL = "Modic"

# Regex for patient-named report files (e.g., "Bùi Thị Dung.pdf" or "NGUYEN_VAN_SON_20250718.pdf")
# Matches Vietnamese names with spaces/underscores, optional date suffix
# Unicode range À-ỹ covers Vietnamese diacritics
PATIENT_NAMED_REPORT_REGEX = re.compile(
    r"^[a-zA-ZÀ-ỹ]+(?:[\s_][a-zA-ZÀ-ỹ]+)*(?:[\s_]\d{8})?$"
)

# Regex for ID-only report files (e.g., 250010139.png)
ID_NAMED_REPORT_REGEX = re.compile(r"^\d+$")


@dataclass
class ReportInfo:
    """Information extracted from a medical report."""

    patient_id: int | None
    patient_name: str | None
    patient_birthday: str | None
    source_path: Path


class ReportProcessor(ABC):
    """Abstract base for report processing strategies."""

    @abstractmethod
    def can_process(self, report_path: Path) -> bool:
        """Check if this processor can handle the given report."""
        ...

    @abstractmethod
    def process(
        self,
        report_path: Path,
        extractor: DocumentExtractor,
        fuzzy_threshold: float,
    ) -> ReportInfo | None:
        """Extract information from the report.

        Returns:
            ReportInfo with extracted data, or None if extraction failed.
        """
        ...


class IdNamedReportProcessor(ReportProcessor):
    """Process reports named with patient ID (e.g., 250010139.png).

    Extracts patient name and birthday from report content via OCR.
    The ID comes from the filename.
    """

    def can_process(self, report_path: Path) -> bool:
        """Check if filename is a numeric ID."""
        return ID_NAMED_REPORT_REGEX.match(report_path.stem) is not None

    def process(
        self,
        report_path: Path,
        extractor: DocumentExtractor,
        fuzzy_threshold: float,
    ) -> ReportInfo | None:
        """Extract name and birthday from report, ID from filename."""
        try:
            patient_id = int(report_path.stem)
        except ValueError:
            logger.warning(f"Could not parse ID from filename: {report_path.name}")
            return None

        text_lines = extractor.extract(report_path)
        if not text_lines:
            logger.warning(f"No text extracted from report: {report_path}")
            return None

        patient_name = fuzzy_value_extract(
            text_lines, NAME_FIELD_PATTERN, fuzzy_threshold, window_length=3
        )
        if not patient_name:
            logger.warning(f"Could not extract name for ID {patient_id}")
            return None

        patient_birthday = fuzzy_value_extract(
            text_lines, BIRTHDAY_FIELD_PATTERN, fuzzy_threshold, window_length=2
        )
        if not patient_birthday:
            logger.warning(f"Could not extract birthday for ID {patient_id}")
            return None

        return ReportInfo(
            patient_id=patient_id,
            patient_name=patient_name,
            patient_birthday=patient_birthday,
            source_path=report_path,
        )


# Default crop region for ID extraction in PDF (x1, y1, x2, y2) in pixels at 200 DPI
DEFAULT_PDF_ID_CROP_REGION: tuple[int, int, int, int] = (1100, 200, 1500, 400)


class PatientNamedReportProcessor(ReportProcessor):
    """Process reports named with patient name (e.g., NGUYEN_VAN_SON_20250718.pdf).

    Parses patient name from filename, extracts ID from report content via OCR.
    For PDFs, first tries extracting ID from a predefined crop region before
    falling back to full-document fuzzy extraction.
    """

    def __init__(
        self,
        pdf_id_crop_region: tuple[int, int, int, int] = DEFAULT_PDF_ID_CROP_REGION,
    ) -> None:
        """Initialize processor with optional custom crop region.

        Args:
            pdf_id_crop_region: Crop box (x1, y1, x2, y2) in pixels at 200 DPI.
        """
        self.pdf_id_crop_region = pdf_id_crop_region

    def can_process(self, report_path: Path) -> bool:
        """Check if filename matches patient name pattern."""
        return PATIENT_NAMED_REPORT_REGEX.match(report_path.stem) is not None

    def _parse_filename(self, filename: str) -> tuple[str, str | None]:
        """Parse patient name and date from filename.

        Args:
            filename: Filename stem (e.g., NGUYEN_VAN_SON_20250718)

        Returns:
            Tuple of (patient_name without underscores, date string or None)
        """
        parts = filename.split("_")

        # Last part should be date (8 digits)
        if len(parts) >= 2 and re.match(r"^\d{8}$", parts[-1]):
            name_parts = parts[:-1]
            date_str = parts[-1]
        else:
            name_parts = parts
            date_str = None

        # Join name parts for matching (no underscores)
        patient_name = "".join(name_parts)
        return patient_name, date_str

    def _extract_id_from_pdf_crop(
        self,
        report_path: Path,
        extractor: DocumentExtractor,
    ) -> int | None:
        """Try to extract patient ID from a cropped region of the PDF.

        Uses a predefined crop region where the ID is typically located.

        Args:
            report_path: Path to PDF file.
            extractor: DocumentExtractor instance.

        Returns:
            Patient ID if successfully extracted, None otherwise.
        """
        try:
            text_lines = extractor.extract_from_pdf_crop(
                report_path, self.pdf_id_crop_region
            )
        except Exception as e:
            logger.debug(f"Failed to extract from PDF crop: {e}")
            return None

        if not text_lines:
            return None

        # Look for numeric ID in extracted text
        for line in text_lines:
            # Extract digits from the line
            digits = re.sub(r"\D", "", line)
            if len(digits) >= 6:  # Minimum ID length
                try:
                    return int(digits)
                except ValueError:
                    continue

        return None

    def process(
        self,
        report_path: Path,
        extractor: DocumentExtractor,
        fuzzy_threshold: float,
    ) -> ReportInfo | None:
        """Extract ID from report, name from filename.

        For PDFs, first tries extracting ID from a predefined crop region.
        Falls back to full-document fuzzy extraction if crop extraction fails.
        """
        patient_name, _ = self._parse_filename(report_path.stem)

        # For PDFs, try crop-based extraction first
        patient_id: int | None = None
        if report_path.suffix.lower() == ".pdf":
            patient_id = self._extract_id_from_pdf_crop(report_path, extractor)
            if patient_id:
                logger.debug(f"Extracted ID {patient_id} from PDF crop region")

        # Fall back to full-document fuzzy extraction
        if patient_id is None:
            text_lines = extractor.extract(report_path)
            if not text_lines:
                logger.warning(f"No text extracted from report: {report_path}")
                return None

            # Extract patient ID from report content
            id_str = fuzzy_value_extract(
                text_lines, ID_FIELD_PATTERN, fuzzy_threshold, window_length=2
            )
            if not id_str:
                logger.warning(f"Could not extract ID for patient: {patient_name}")
                return None

            # Parse ID - handle potential OCR noise
            id_cleaned = re.sub(r"\D", "", id_str)  # Keep only digits
            if not id_cleaned:
                logger.warning(
                    f"Invalid ID format '{id_str}' for patient: {patient_name}"
                )
                return None

            try:
                patient_id = int(id_cleaned)
            except ValueError:
                logger.warning(
                    f"Could not parse ID '{id_str}' for patient: {patient_name}"
                )
                return None

        # Extract birthday for folder matching (always from full document)
        text_lines = extractor.extract(report_path)
        patient_birthday = None
        if text_lines:
            patient_birthday = fuzzy_value_extract(
                text_lines, BIRTHDAY_FIELD_PATTERN, fuzzy_threshold, window_length=2
            )

        return ReportInfo(
            patient_id=patient_id,
            patient_name=patient_name,
            patient_birthday=patient_birthday,
            source_path=report_path,
        )


class ReportProcessorRegistry:
    """Registry of report processors, tried in order."""

    def __init__(self) -> None:
        self._processors: list[ReportProcessor] = []

    def register(self, processor: ReportProcessor) -> None:
        """Register a processor."""
        self._processors.append(processor)

    def process(
        self,
        report_path: Path,
        extractor: DocumentExtractor,
        fuzzy_threshold: float,
    ) -> ReportInfo | None:
        """Try each processor until one succeeds."""
        for processor in self._processors:
            if processor.can_process(report_path):
                return processor.process(report_path, extractor, fuzzy_threshold)

        logger.debug(f"No processor matched: {report_path.name}")
        return None


def build_report_processor_registry(
    pdf_id_crop_region: tuple[int, int, int, int] = DEFAULT_PDF_ID_CROP_REGION,
) -> ReportProcessorRegistry:
    """Create registry with default processors.

    Args:
        pdf_id_crop_region: Crop box for PDF ID extraction (x1, y1, x2, y2) in pixels.
    """
    registry = ReportProcessorRegistry()
    registry.register(IdNamedReportProcessor())
    registry.register(PatientNamedReportProcessor(pdf_id_crop_region))
    return registry


def collect_report_files(report_path: Path) -> list[Path]:
    """Collect all supported report files from a directory.

    Args:
        report_path: Root directory to search.

    Returns:
        List of paths to report files (images and PDFs).
    """
    report_files: list[Path] = []

    for ext in SUPPORTED_EXTENSIONS:
        report_files.extend(report_path.rglob(f"*{ext}"))

    logger.info(f"Found {len(report_files)} report files")
    return report_files


class PreprocessConfig(BaseModel):
    """Configuration for preprocessing pipeline."""

    data_path: Path = Path.cwd() / "data/raw/Phenikaa"
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
    output_path: Path = Path.cwd() / "data/interim/Phenikaa"
    output_table: str = "radiological_labels.csv"
    model_path: Path = Path.cwd() / "weights/ocr"
    detection_model: str = "PP-OCRv5_server_det"
    recognition_model: str = "vgg_transformer"
    report_fuzzy_threshold: float = 80
    image_fuzzy_threshold: float = 85
    pdf_dpi: int = 200
    pdf_id_crop_region: tuple[int, int, int, int] = DEFAULT_PDF_ID_CROP_REGION
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
    """Run preprocessing pipeline.

    Processes medical reports to match patient IDs to image folders:
    1. Loads radiological grading labels from Excel/CSV
    2. Scans for report files (images and PDFs)
    3. For each report, extracts patient info via OCR
    4. Matches extracted info to image folders
    5. Copies matched images and saves filtered labels
    """
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

    setup_logger(
        verbose=config.verbose,
    )

    if config.enable_file_log:
        add_file_log()

    logger.debug("Started preprocessing.")

    # Load label data
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

    logger.debug(f"Unique Patients: {label_data[config.id_col].nunique()}")

    # Initialize OCR
    device = "cuda:0" if config.use_gpu else "cpu"
    logger.info("Loading OCR models.")
    extractor = DocumentExtractor(
        detection_model=config.detection_model,
        recognition_model=config.recognition_model,
        device=device,
        use_gpu=config.use_gpu,
        pdf_dpi=config.pdf_dpi,
    )

    # Collect all report files
    report_files = collect_report_files(config.report_path)

    # Build processor registry
    processor_registry = build_report_processor_registry(config.pdf_id_crop_region)

    # Build patient matcher
    patient_matcher = PatientMatcher(
        image_path=config.image_path,
        threshold=config.image_fuzzy_threshold,
    )

    # Track which IDs we need
    valid_ids = set(label_data[config.id_col].unique())
    matched_ids: list[int] = []

    for report_path in tqdm(report_files, desc="Processing Reports", unit="report"):
        # Process report to extract info
        report_info = processor_registry.process(
            report_path, extractor, config.report_fuzzy_threshold
        )
        if not report_info:
            continue

        # Skip if ID is None or not in label data
        if report_info.patient_id is None:
            continue
        if report_info.patient_id not in valid_ids:
            logger.debug(f"ID {report_info.patient_id} not in label data, skipping")
            continue

        # Match to image folder
        if report_info.patient_name and report_info.patient_birthday:
            best_folder = patient_matcher.match(
                report_info.patient_name, report_info.patient_birthday
            )
        elif report_info.patient_name:
            # Try matching by name only (patient-named reports may lack birthday)
            best_folder = patient_matcher.match_by_name(report_info.patient_name)
        else:
            best_folder = None

        if best_folder:
            dest = config.output_image_path / str(report_info.patient_id)
            shutil.copytree(best_folder, dest, dirs_exist_ok=True)
            logger.info(f"Copied {best_folder.name} -> {dest}")
            matched_ids.append(report_info.patient_id)
        else:
            logger.warning(
                f"No matching folder for '{report_info.patient_name}' "
                f"(ID: {report_info.patient_id})"
            )

    # Filter and save labels
    label_data = label_data[label_data[config.id_col].isin(matched_ids)]
    label_data.to_csv(config.output_table_path, index=False)
    logger.info(f"Saved table to {config.output_table_path}")
    logger.info(f"Matched {len(matched_ids)} patients out of {len(valid_ids)}")

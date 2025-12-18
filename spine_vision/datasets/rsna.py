"""RSNA dataset utilities for loading and parsing RSNA-format data."""

import csv
from pathlib import Path


def load_series_mapping(series_desc_path: Path) -> dict[int, dict[str, int]]:
    """Load RSNA series descriptions to map study_id -> {series_description: series_id}.

    Parses the train_series_descriptions.csv file from RSNA dataset.

    Args:
        series_desc_path: Path to train_series_descriptions.csv.

    Returns:
        Nested dict mapping study_id -> series_description -> series_id.

    Example:
        >>> mapping = load_series_mapping(Path("train_series_descriptions.csv"))
        >>> mapping[12345]["Sagittal T1"]
        67890
    """
    mapping: dict[int, dict[str, int]] = {}
    with open(series_desc_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            study_id = int(row["study_id"])
            series_id = int(row["series_id"])
            series_desc = row["series_description"]
            if study_id not in mapping:
                mapping[study_id] = {}
            mapping[study_id][series_desc] = series_id
    return mapping


def get_series_type(
    series_id: int, study_id: int, series_mapping: dict[int, dict[str, int]]
) -> str | None:
    """Get series type (Sagittal T1, Sagittal T2/STIR, Axial T2) from series_id.

    Looks up the series description for a given series_id within a study.

    Args:
        series_id: Series ID to look up.
        study_id: Study ID containing the series.
        series_mapping: Mapping from load_series_mapping.

    Returns:
        Series description string or None if not found.

    Example:
        >>> mapping = load_series_mapping(Path("train_series_descriptions.csv"))
        >>> get_series_type(67890, 12345, mapping)
        "Sagittal T1"
    """
    if study_id not in series_mapping:
        return None
    for series_desc, sid in series_mapping[study_id].items():
        if sid == series_id:
            return series_desc
    return None

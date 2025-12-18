"""Tabular data loading utilities for Excel and CSV files."""

import csv
from pathlib import Path
from typing import TypeVar

import pandas as pd
from loguru import logger
from openpyxl.reader.excel import SUPPORTED_FORMATS
from pydantic import BaseModel


EXCEL_FORMATS = (*SUPPORTED_FORMATS, ".xls")


T = TypeVar("T", bound=BaseModel)


def write_records_csv(records: list[T], csv_path: Path) -> None:
    """Write Pydantic model records to CSV file.

    Args:
        records: List of Pydantic model instances.
        csv_path: Output CSV path.

    Raises:
        ValueError: If records list is empty.
    """
    if not records:
        raise ValueError("Cannot write empty records list")

    fieldnames = list(records[0].model_fields.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.model_dump())


def load_tabular_data(
    table_path: Path,
    exclude_files: list[str] | None = None,
    id_col: str = "Patient ID",
    corrupted_ids: list[int] | None = None,
    one_hot_col: str | None = None,
    one_hot_sep: str = "&",
) -> pd.DataFrame:
    """Load and preprocess tabular data from Excel/CSV files.

    Args:
        table_path: Directory containing Excel/CSV files.
        exclude_files: List of filenames to exclude.
        id_col: Column name for patient IDs.
        corrupted_ids: List of IDs to filter out.
        one_hot_col: Column name to one-hot encode (optional).
        one_hot_sep: Separator for one-hot encoding.

    Returns:
        Preprocessed DataFrame with duplicates and NAs removed.
    """
    exclude_files = exclude_files or []
    corrupted_ids = corrupted_ids or []

    files_data = []
    valid_files = (
        p
        for p in table_path.rglob("*")
        if p.is_file() and p.name not in exclude_files
    )

    for file_path in valid_files:
        match file_path.suffix.lower():
            case ".csv":
                files_data.append(pd.read_csv(file_path))
            case suffix if suffix in EXCEL_FORMATS:
                files_data.append(pd.read_excel(file_path))
            case _:
                logger.warning(f"Unsupported format: {file_path}")

    if not files_data:
        logger.warning(f"No valid data files found in {table_path}")
        return pd.DataFrame()

    df = pd.concat(files_data, ignore_index=True)

    initial_size = len(df)
    df = df.drop_duplicates()
    logger.debug(f"Dropped {initial_size - len(df)} duplicates.")

    na_count = df.isna().any(axis=1).sum()
    df = df.dropna()
    logger.debug(f"Dropped {na_count} rows with NA.")

    if corrupted_ids and id_col in df.columns:
        df = df[~df[id_col].isin(corrupted_ids)]

    if one_hot_col and one_hot_col in df.columns:
        dummies = (
            df[one_hot_col]
            .astype(str)
            .str.replace(r"\.0\b", "", regex=True)
            .str.get_dummies(sep=one_hot_sep)
            .add_prefix(f"{one_hot_col}_")
        )
        df = pd.concat([df, dummies], axis=1).drop(columns=one_hot_col)

    logger.info(f"Loaded {len(df)} rows from tabular data.")
    return df

"""Stratification utilities for patient-level data splitting.

Provides functions for stratified train/val/test splits that maintain
label distribution across splits, with support for both single-label
and multilabel stratification.
"""


import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit

from spine_vision.core.tasks import get_task


def get_patient_single_label(
    patients: list[str],
    records: list[dict],
    label: str,
) -> np.ndarray:
    """Get single stratification label for each patient.

    Uses the most frequent value of the specified label across all IVD levels.
    Used for single-task training where only one label is targeted.

    Args:
        patients: List of patient keys.
        records: List of record dictionaries.
        label: The label name to stratify on.

    Returns:
        Array of shape (n_patients,) with the most frequent label value per patient.
    """
    patient_set = set(patients)
    patient_to_labels: dict[str, list[int]] = {p: [] for p in patients}

    # Map label name to record key
    label_to_record_key = {
        "pfirrmann": "pfirrmann",
        "modic": "modic",
        "herniation": "herniation",
        "bulging": "bulging",
        "upper_endplate": "upper_endplate",
        "lower_endplate": "lower_endplate",
        "spondy": "spondylolisthesis",
        "narrowing": "narrowing",
    }
    record_key = label_to_record_key.get(label, label)

    for record in records:
        patient_key = record["patient_key"]
        if patient_key in patient_set:
            patient_to_labels[patient_key].append(record[record_key])

    # Use most frequent value as the stratification label
    labels = []
    for patient in patients:
        values = patient_to_labels.get(patient, [0])
        if not values:
            values = [0]

        strat_label = max(values)
        labels.append(strat_label)

    return np.array(labels)


def get_patient_multilabel_matrix(
    patients: list[str],
    records: list[dict],
    target_labels: list[str],
) -> np.ndarray:
    """Get multilabel binary matrix for each patient.

    For each patient, aggregates labels across all IVD levels:
    - Binary labels: 1 if any IVD level has the condition
    - Multiclass labels (pfirrmann, modic): converted to binary indicators
      for each class (one-hot style aggregation with max across levels)

    Args:
        patients: List of patient keys.
        records: List of record dictionaries.
        target_labels: List of target label names.

    Returns:
        Array of shape (n_patients, n_label_columns) with binary indicators.
    """
    patient_set = set(patients)
    patient_idx = {p: i for i, p in enumerate(patients)}

    # Build column structure based on target_labels
    columns: list[tuple[str, int | None]] = []  # (label_name, class_idx or None)
    for label in target_labels:
        task = get_task(label)
        if task.is_multiclass:
            for cls_idx in range(task.num_classes):
                columns.append((label, cls_idx))
        else:
            columns.append((label, None))

    n_patients = len(patients)
    n_columns = len(columns)
    matrix = np.zeros((n_patients, n_columns), dtype=np.float32)

    # Map label name to record key
    label_to_record_key = {
        "pfirrmann": "pfirrmann",
        "modic": "modic",
        "herniation": "herniation",
        "bulging": "bulging",
        "upper_endplate": "upper_endplate",
        "lower_endplate": "lower_endplate",
        "spondy": "spondylolisthesis",
        "narrowing": "narrowing",
    }

    # Aggregate labels per patient
    for record in records:
        patient_key = record["patient_key"]
        if patient_key not in patient_set:
            continue

        row_idx = patient_idx[patient_key]

        for col_idx, (label, cls_idx) in enumerate(columns):
            record_key = label_to_record_key.get(label, label)
            value = record[record_key]

            if cls_idx is not None:
                # Multiclass: check if this class is present
                if label == "pfirrmann":
                    if value == cls_idx + 1:  # pfirrmann is 1-indexed
                        matrix[row_idx, col_idx] = 1.0
                else:
                    if value == cls_idx:
                        matrix[row_idx, col_idx] = 1.0
            else:
                # Binary: use max (if any IVD has condition, patient has it)
                if value > 0:
                    matrix[row_idx, col_idx] = 1.0

    return matrix


def split_patients_single_label(
    patients: list[str],
    records: list[dict],
    target_label: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    """Split patients using single-label stratification.

    Uses sklearn's StratifiedShuffleSplit with the most frequent value
    of the single target label per patient.

    Args:
        patients: List of unique patient keys.
        records: List of record dictionaries.
        target_label: The label to stratify on.
        val_ratio: Fraction of data for validation set.
        test_ratio: Fraction of data for test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_patients, val_patients, test_patients) as sets.
    """
    patients_arr = np.array(patients)
    stratify_labels = get_patient_single_label(patients, records, target_label)

    # First split: separate test set
    if test_ratio > 0:
        splitter_test = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_ratio,
            random_state=seed,
        )
        train_val_idx, test_idx = next(
            splitter_test.split(patients_arr, stratify_labels)
        )
        test_patients = set(patients_arr[test_idx])
        remaining_patients = patients_arr[train_val_idx]
        remaining_labels = stratify_labels[train_val_idx]
    else:
        test_patients = set()
        remaining_patients = patients_arr
        remaining_labels = stratify_labels

    # Second split: separate val from train
    if val_ratio > 0:
        adjusted_val_ratio = val_ratio / (1 - test_ratio)
        splitter_val = StratifiedShuffleSplit(
            n_splits=1,
            test_size=adjusted_val_ratio,
            random_state=seed,
        )
        train_idx, val_idx = next(
            splitter_val.split(remaining_patients, remaining_labels)
        )
        train_patients = set(remaining_patients[train_idx])
        val_patients = set(remaining_patients[val_idx])
    else:
        train_patients = set(remaining_patients)
        val_patients = set()

    return train_patients, val_patients, test_patients


def split_patients_multilabel(
    patients: list[str],
    records: list[dict],
    target_labels: list[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    """Split patients using multilabel iterative stratification.

    Uses iterstrat's MultilabelStratifiedShuffleSplit to preserve
    the distribution of all target labels across splits.

    Args:
        patients: List of unique patient keys.
        records: List of record dictionaries.
        target_labels: List of target label names.
        val_ratio: Fraction of data for validation set.
        test_ratio: Fraction of data for test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_patients, val_patients, test_patients) as sets.
    """
    patients_arr = np.array(patients)
    label_matrix = get_patient_multilabel_matrix(patients, records, target_labels)

    # First split: separate test set
    if test_ratio > 0:
        # iterstrat type stubs incorrectly mark test_size as str
        splitter_test = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=test_ratio,  # pyright: ignore[reportArgumentType]
            random_state=seed,
        )
        train_val_idx, test_idx = next(
            splitter_test.split(patients_arr, label_matrix)
        )
        test_patients = set(patients_arr[test_idx])
        remaining_patients = patients_arr[train_val_idx]
        remaining_labels = label_matrix[train_val_idx]
    else:
        test_patients = set()
        remaining_patients = patients_arr
        remaining_labels = label_matrix

    # Second split: separate val from train
    if val_ratio > 0:
        adjusted_val_ratio = val_ratio / (1 - test_ratio)
        splitter_val = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=adjusted_val_ratio,  # pyright: ignore[reportArgumentType]
            random_state=seed,
        )
        train_idx, val_idx = next(
            splitter_val.split(remaining_patients, remaining_labels)
        )
        train_patients = set(remaining_patients[train_idx])
        val_patients = set(remaining_patients[val_idx])
    else:
        train_patients = set(remaining_patients)
        val_patients = set()

    return train_patients, val_patients, test_patients


def split_patients(
    patients: list[str],
    records: list[dict],
    target_labels: list[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    """Split patients into train/val/test sets with stratification.

    Automatically selects the appropriate stratification strategy:
    - Single-label mode (1 target label): Uses standard StratifiedShuffleSplit
    - Multilabel mode (2+ target labels): Uses MultilabelStratifiedShuffleSplit

    Args:
        patients: List of unique patient keys.
        records: List of record dictionaries.
        target_labels: List of target label names.
        val_ratio: Fraction of data for validation set.
        test_ratio: Fraction of data for test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_patients, val_patients, test_patients) as sets.
    """
    is_multilabel = len(target_labels) > 1

    if is_multilabel:
        return split_patients_multilabel(
            patients, records, target_labels, val_ratio, test_ratio, seed
        )
    else:
        return split_patients_single_label(
            patients, records, target_labels[0], val_ratio, test_ratio, seed
        )

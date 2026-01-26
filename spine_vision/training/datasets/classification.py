"""Classification Dataset for lumbar spine multi-task learning.

Loads pre-extracted IVD crops from the classification dataset CLI.
Supports dual-modality (T1+T2) input construction with 3-channel output.

Supports training on all labels (multi-task) or individual labels (single-task)
via the `target_labels` parameter.

Expected dataset structure (from `spine-vision dataset classification`):
    data_path/
        images/
            phenikaa_<patient_id>_<series_type>_L<level>.png
            spider_<patient_id>_<series_type>_L<level>.png
        annotations.csv

Annotations CSV columns:
    image_path, patient_id, ivd_level, series_type, source,
    pfirrmann_grade, disc_herniation, disc_narrowing, disc_bulging,
    spondylolisthesis, modic, up_endplate, low_endplate
"""

import csv
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from spine_vision.core.tasks import AVAILABLE_TASK_NAMES, get_task
from spine_vision.datasets.levels import IDX_TO_LEVEL


def construct_3channel(
    t2_crop: np.ndarray | None,
    t1_crop: np.ndarray | None,
) -> np.ndarray:
    """Construct 3-channel image from T1 and/or T2 crops.

    Channel layout:
        - Both T1 and T2: [T2, T1, T2] (RGB-like)
        - T2 only: [T2, T2, T2]
        - T1 only: [T1, T1, T1]

    Args:
        t2_crop: T2 crop, shape (H, W), uint8. Can be None if T1 provided.
        t1_crop: T1 crop, shape (H, W), uint8. Can be None if T2 provided.

    Returns:
        3-channel image, shape (H, W, 3), uint8.

    Raises:
        ValueError: If both crops are None.
    """
    if t2_crop is not None and t1_crop is not None:
        return np.stack([t2_crop, t1_crop, t2_crop], axis=-1)
    elif t2_crop is not None:
        return np.stack([t2_crop, t2_crop, t2_crop], axis=-1)
    elif t1_crop is not None:
        return np.stack([t1_crop, t1_crop, t1_crop], axis=-1)
    else:
        raise ValueError("At least one of t2_crop or t1_crop must be provided")


class ClassificationDataset(Dataset[dict[str, Any]]):
    """Dataset for lumbar spine classification using pre-extracted crops.

    Loads crops created by the `spine-vision dataset classification` CLI command.
    Automatically pairs T1 and T2 crops for the same patient/level to create
    3-channel inputs.

    Supports training on all labels (multi-task) or specific labels (single-task)
    via the `target_labels` parameter.

    Expected directory structure:
        data_path/
            images/
                <source>_<patient_id>_<series_type>_L<level>.png
            annotations.csv

    The dataset automatically pairs T1 and T2 images for the same patient and
    level, creating a 3-channel [T2, T1, T2] input for each sample.
    """

    def __init__(
        self,
        data_path: Path,
        split: Literal["train", "val", "test", "all"] = "all",
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        levels: list[str] | None = None,
        series_types: list[str] | None = None,
        target_labels: list[str] | None = None,
        output_size: tuple[int, int] = (256, 256),
        augment: bool = True,
        normalize: bool = True,
        seed: int = 42,
    ) -> None:
        """Initialize dataset.

        Args:
            data_path: Path to dataset directory (containing images/ and annotations.csv).
            split: Data split ('train', 'val', 'test', 'all').
            val_ratio: Fraction for validation.
            test_ratio: Fraction for testing.
            levels: Filter to specific IVD levels (e.g., ["L4/L5", "L5/S1"]).
            series_types: Filter to specific series types (e.g., ["sag_t2"] for T2 only,
                ["sag_t1", "sag_t2"] for both). If None, requires both T1 and T2.
            target_labels: Filter to specific labels (e.g., ["pfirrmann", "modic"]).
                If None, includes all labels.
            output_size: Final output size after resizing (H, W).
            augment: Apply data augmentation (training only).
            normalize: Apply ImageNet normalization.
            seed: Random seed for splitting.

        Raises:
            ValueError: If any target_label is not a valid label name.
            ValueError: If any series_type is not valid.
        """
        self.data_path = Path(data_path)
        self.split = split
        self.output_size = output_size
        self.augment = augment and split == "train"
        self.normalize = normalize

        # Validate and store series types
        valid_series = {"sag_t1", "sag_t2"}
        if series_types is not None:
            invalid_series = set(series_types) - valid_series
            if invalid_series:
                raise ValueError(
                    f"Invalid series types: {invalid_series}. "
                    f"Valid types: {valid_series}"
                )
            self.series_types = set(series_types)
        else:
            self.series_types = valid_series  # Require both by default

        # Validate and store target labels
        if target_labels is not None:
            invalid_labels = set(target_labels) - set(AVAILABLE_TASK_NAMES)
            if invalid_labels:
                raise ValueError(
                    f"Invalid target labels: {invalid_labels}. "
                    f"Available labels: {AVAILABLE_TASK_NAMES}"
                )
            self.target_labels = list(target_labels)
        else:
            self.target_labels = list(AVAILABLE_TASK_NAMES)

        # Load and process annotations
        self.records = self._load_and_pair_annotations()

        # Filter by level
        if levels:
            level_set = set(levels)
            self.records = [
                r for r in self.records if IDX_TO_LEVEL.get(r["level_idx"]) in level_set
            ]

        # Split by patient to avoid data leakage
        unique_patients = self._get_unique_patients()
        train_patients, val_patients, test_patients = self._split_patients(
            unique_patients, val_ratio, test_ratio, seed
        )

        if split == "train":
            self.records = [
                r for r in self.records if r["patient_key"] in train_patients
            ]
        elif split == "val":
            self.records = [
                r for r in self.records if r["patient_key"] in val_patients
            ]
        elif split == "test":
            self.records = [
                r for r in self.records if r["patient_key"] in test_patients
            ]

        # Build transforms
        self.transform = self._build_transforms()

    def _load_and_pair_annotations(self) -> list[dict[str, Any]]:
        """Load annotations and pair T1/T2 crops for the same patient/level."""
        csv_path = self.data_path / "annotations.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Annotations not found: {csv_path}")

        # First pass: group by (source, patient_id, ivd_level)
        groups: dict[tuple[str, str, int], dict[str, Any]] = {}

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                source = row["source"]
                patient_id = row["patient_id"]
                ivd_level = int(row["ivd_level"])
                series_type = row["series_type"]

                key = (source, patient_id, ivd_level)

                if key not in groups:
                    groups[key] = {
                        "source": source,
                        "patient_id": patient_id,
                        "patient_key": f"{source}_{patient_id}",
                        "ivd_level": ivd_level,
                        "level_idx": ivd_level - 1,  # 1-5 -> 0-4
                        # Labels (same for T1 and T2 of same level)
                        "pfirrmann": int(row["pfirrmann_grade"]),
                        "modic": int(row["modic"]),
                        "herniation": int(row["disc_herniation"]),
                        "bulging": int(row["disc_bulging"]),
                        "upper_endplate": int(row["up_endplate"]),
                        "lower_endplate": int(row["low_endplate"]),
                        "spondylolisthesis": int(row["spondylolisthesis"]),
                        "narrowing": int(row["disc_narrowing"]),
                        "t1_path": None,
                        "t2_path": None,
                    }

                # Store image path by series type
                image_path = self.data_path / row["image_path"]
                if series_type == "sag_t1":
                    groups[key]["t1_path"] = image_path
                elif series_type == "sag_t2":
                    groups[key]["t2_path"] = image_path

        # Second pass: filter based on series_types
        require_t1 = "sag_t1" in self.series_types
        require_t2 = "sag_t2" in self.series_types

        records = []
        for group in groups.values():
            has_t1 = group["t1_path"] is not None
            has_t2 = group["t2_path"] is not None

            # Check if sample meets requirements
            if require_t1 and require_t2:
                # Need both T1 and T2
                if has_t1 and has_t2:
                    records.append(group)
            elif require_t1:
                # T1 only mode
                if has_t1:
                    records.append(group)
            elif require_t2:
                # T2 only mode
                if has_t2:
                    records.append(group)

        return records

    def _get_unique_patients(self) -> list[str]:
        """Get list of unique patient keys (source_patientid)."""
        return list(set(r["patient_key"] for r in self.records))

    def _get_patient_single_label(self, patients: list[str], label: str) -> np.ndarray:
        """Get single stratification label for each patient.

        Uses the most frequent value of the specified label across all IVD levels.
        Used for single-task training where only one label is targeted.

        Args:
            patients: List of patient keys.
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

        for record in self.records:
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

            # most_common = Counter(values).most_common(1)[0][0]
            # labels.append(most_common)

        return np.array(labels)

    def _get_patient_multilabel_matrix(self, patients: list[str]) -> np.ndarray:
        """Get multilabel binary matrix for each patient.

        For each patient, aggregates labels across all IVD levels:
        - Binary labels: 1 if any IVD level has the condition
        - Multiclass labels (pfirrmann, modic): converted to binary indicators
          for each class (one-hot style aggregation with max across levels)

        Args:
            patients: List of patient keys.

        Returns:
            Array of shape (n_patients, n_label_columns) with binary indicators.
            Columns correspond to all target labels, with multiclass labels
            expanded into binary indicators.
        """
        patient_set = set(patients)
        patient_idx = {p: i for i, p in enumerate(patients)}

        # Build column structure based on target_labels
        # Binary labels: 1 column each
        # Multiclass labels: expanded to n_classes columns
        columns: list[tuple[str, int | None]] = []  # (label_name, class_idx or None)
        for label in self.target_labels:
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
        for record in self.records:
            patient_key = record["patient_key"]
            if patient_key not in patient_set:
                continue

            row_idx = patient_idx[patient_key]

            for col_idx, (label, cls_idx) in enumerate(columns):
                record_key = label_to_record_key.get(label, label)
                value = record[record_key]

                if cls_idx is not None:
                    # Multiclass: check if this class is present
                    # For pfirrmann (1-5), compare directly
                    # For modic (0-3), compare directly
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

    def _split_patients(
        self,
        patients: list[str],
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> tuple[set[str], set[str], set[str]]:
        """Split patients into train/val/test sets with stratification.

        Automatically selects the appropriate stratification strategy:
        - Single-label mode (1 target label): Uses standard StratifiedShuffleSplit
          with the most frequent label value per patient.
        - Multilabel mode (2+ target labels): Uses MultilabelStratifiedShuffleSplit
          (iterative stratification) to preserve distribution across all labels.

        Args:
            patients: List of unique patient keys.
            val_ratio: Fraction of data for validation set.
            test_ratio: Fraction of data for test set.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_patients, val_patients, test_patients) as sets.
        """
        is_multilabel = len(self.target_labels) > 1

        if is_multilabel:
            return self._split_patients_multilabel(
                patients, val_ratio, test_ratio, seed
            )
        else:
            return self._split_patients_single_label(
                patients, val_ratio, test_ratio, seed
            )

    def _split_patients_single_label(
        self,
        patients: list[str],
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> tuple[set[str], set[str], set[str]]:
        """Split patients using single-label stratification.

        Uses sklearn's StratifiedShuffleSplit with the most frequent value
        of the single target label per patient.
        """
        patients_arr = np.array(patients)
        # Use the single target label for stratification
        stratify_labels = self._get_patient_single_label(
            patients, self.target_labels[0]
        )

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

    def _split_patients_multilabel(
        self,
        patients: list[str],
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> tuple[set[str], set[str], set[str]]:
        """Split patients using multilabel iterative stratification.

        Uses iterstrat's MultilabelStratifiedShuffleSplit to preserve
        the distribution of all target labels across splits.
        """
        patients_arr = np.array(patients)
        label_matrix = self._get_patient_multilabel_matrix(patients)

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

    def _build_transforms(self) -> Callable[[Image.Image], torch.Tensor]:
        """Build image transforms."""
        transform_list: list[Any] = [
            transforms.Resize(self.output_size),
        ]

        if self.augment:
            transform_list.extend(
                [
                    transforms.RandomAffine(
                        degrees=10,
                        translate=(0.05, 0.05),
                        scale=(0.95, 1.05),
                    ),
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                    ),
                ]
            )

        transform_list.append(transforms.ToTensor())

        if self.normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )

        return transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Returns:
            Dictionary with keys:
                - image: 3-channel tensor [C, H, W] constructed based on series_types:
                    - Both T1+T2: [T2, T1, T2]
                    - T2 only: [T2, T2, T2]
                    - T1 only: [T1, T1, T1]
                - targets: Dict with only the selected target labels
                - level_idx: Level index (0-4)
                - metadata: Dict with source, patient_id, level
        """
        record = self.records[idx]

        # Load T1 and/or T2 crops based on what's available
        t1_arr = None
        t2_arr = None

        if record["t1_path"] is not None:
            t1_img = Image.open(record["t1_path"]).convert("L")
            t1_arr = np.array(t1_img)

        if record["t2_path"] is not None:
            t2_img = Image.open(record["t2_path"]).convert("L")
            t2_arr = np.array(t2_img)

        # Construct 3-channel image
        rgb_image = construct_3channel(t2_arr, t1_arr)

        # Convert to PIL for transforms
        pil_image = Image.fromarray(rgb_image)
        image_tensor = self.transform(pil_image)

        # Build targets (only for selected labels)
        all_targets = {
            # Pfirrmann: convert 1-5 to 0-4 for CrossEntropy
            "pfirrmann": record["pfirrmann"] - 1,
            "modic": record["modic"],
            "herniation": [float(record["herniation"])],
            "bulging": [float(record["bulging"])],
            "upper_endplate": [float(record["upper_endplate"])],
            "lower_endplate": [float(record["lower_endplate"])],
            "spondy": [float(record["spondylolisthesis"])],
            "narrowing": [float(record["narrowing"])],
        }

        # Filter to only include target labels
        targets = {k: v for k, v in all_targets.items() if k in self.target_labels}

        return {
            "image": image_tensor,
            "targets": targets,
            "level_idx": record["level_idx"],
            "metadata": {
                "source": record["source"],
                "patient_id": record["patient_id"],
                "level": IDX_TO_LEVEL.get(record["level_idx"], ""),
                "ivd": record["ivd_level"],
            },
        }

    def get_stats(self) -> dict[str, Any]:
        """Get dataset statistics."""
        levels = [IDX_TO_LEVEL.get(r["level_idx"], "") for r in self.records]
        pfirrmann = [r["pfirrmann"] for r in self.records]
        modic = [r["modic"] for r in self.records]
        sources = [r["source"] for r in self.records]

        return {
            "num_samples": len(self.records),
            "num_patients": len(self._get_unique_patients()),
            "levels": dict(Counter(levels)),
            "pfirrmann": dict(Counter(pfirrmann)),
            "modic": dict(Counter(modic)),
            "sources": dict(Counter(sources)),
            "series_types": list(self.series_types),
            "target_labels": self.target_labels,
            "split": self.split,
        }

    def get_label_distribution(self) -> dict[str, dict[int | str, int]]:
        """Get distribution of each label in the dataset.

        Returns counts for each class within each target label.
        Useful for visualizing label balance across train/val/test splits.

        Returns:
            Dictionary mapping label names to class counts.
            For multiclass labels (pfirrmann, modic): {class_value: count}
            For binary labels: {0: count, 1: count}
        """
        distribution: dict[str, dict[int | str, int]] = {}

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

        for label in self.target_labels:
            record_key = label_to_record_key.get(label, label)
            values = [r[record_key] for r in self.records]
            distribution[label] = dict(Counter(values))

        return distribution

    def compute_class_weights(self) -> dict[str, torch.Tensor]:
        """Compute class weights for imbalanced classification.

        Uses inverse frequency weighting: weight = n_samples / (n_classes * count).
        For binary tasks, computes pos_weight for BCEWithLogitsLoss.

        Only computes weights for labels in `target_labels`.

        Returns:
            Dictionary with class weights for each target label.
        """
        n_samples = len(self.records)
        weights: dict[str, torch.Tensor] = {}

        # Compute binary pos_weight: n_negative / n_positive
        def pos_weight(n_pos: int) -> float:
            n_neg = n_samples - n_pos
            return n_neg / max(n_pos, 1)  # Avoid division by zero

        # Multiclass labels
        if "pfirrmann" in self.target_labels:
            pfirrmann_counts = Counter(r["pfirrmann"] - 1 for r in self.records)
            pfirrmann_weights = torch.zeros(5)
            for i in range(5):
                count = pfirrmann_counts.get(i, 1)
                pfirrmann_weights[i] = n_samples / (5 * count)
            weights["pfirrmann"] = pfirrmann_weights

        if "modic" in self.target_labels:
            modic_counts = Counter(r["modic"] for r in self.records)
            modic_weights = torch.zeros(4)
            for i in range(4):
                count = modic_counts.get(i, 1)
                modic_weights[i] = n_samples / (4 * count)
            weights["modic"] = modic_weights

        # Binary labels
        binary_labels = {
            "herniation": "herniation",
            "bulging": "bulging",
            "upper_endplate": "upper_endplate",
            "lower_endplate": "lower_endplate",
            "spondy": "spondylolisthesis",
            "narrowing": "narrowing",
        }

        for label_name, record_key in binary_labels.items():
            if label_name in self.target_labels:
                n_pos = sum(r[record_key] for r in self.records)
                weights[label_name] = torch.tensor([pos_weight(n_pos)])

        return weights


def create_weighted_sampler(
    dataset: ClassificationDataset,
    target_label: str,
) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler for handling class imbalance.

    Computes sample weights based on inverse class frequency (1 / N_c) where N_c
    is the count of samples in class c. This ensures that samples from minority
    classes are sampled more frequently during training.

    Args:
        dataset: The ClassificationDataset to sample from.
        target_label: The label key to use for computing class weights.
            For multiclass labels (pfirrmann, modic), uses the class index.
            For binary labels, uses 0/1 values.

    Returns:
        WeightedRandomSampler with replacement=True for balanced sampling.

    Raises:
        ValueError: If target_label is not valid.

    Example:
        >>> dataset = ClassificationDataset(data_path, split="train")
        >>> sampler = create_weighted_sampler(dataset, "pfirrmann")
        >>> loader = DataLoader(dataset, sampler=sampler, batch_size=32)
    """
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

    if target_label not in label_to_record_key:
        raise ValueError(
            f"Invalid target_label: {target_label}. "
            f"Valid labels: {list(label_to_record_key.keys())}"
        )

    record_key = label_to_record_key[target_label]

    # Extract target values for every sample
    # For pfirrmann, values are 1-5 in records, convert to 0-4 for consistency
    if target_label == "pfirrmann":
        target_values = [r[record_key] - 1 for r in dataset.records]
    else:
        target_values = [r[record_key] for r in dataset.records]

    # Compute class counts
    class_counts = Counter(target_values)

    # Compute weight per class: 1 / N_c
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}

    # Map weights to every sample
    sample_weights = [class_weights[val] for val in target_values]

    # Create and return the sampler
    # WeightedRandomSampler expects Sequence[float], so we pass the list directly
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


class DynamicTargets:
    """Container for dynamic multi-task targets.

    Unlike MTLTargets which has fixed fields, this class works with any subset
    of labels. Provides the same interface (to(), to_dict()) for compatibility.

    Example:
        targets = DynamicTargets({"pfirrmann": tensor1, "modic": tensor2})
        targets = targets.to("cuda:0")
        targets_dict = targets.to_dict()
    """

    def __init__(self, data: dict[str, torch.Tensor]) -> None:
        """Initialize with target tensors.

        Args:
            data: Dictionary mapping label names to tensors.
        """
        self._data = data

    def to(self, device: torch.device | str) -> "DynamicTargets":
        """Move all tensors to the specified device."""
        return DynamicTargets({k: v.to(device) for k, v in self._data.items()})

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert to dictionary format."""
        return self._data

    def __getattr__(self, name: str) -> torch.Tensor:
        """Allow attribute-style access to targets."""
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"No target named '{name}'")

    def __contains__(self, name: str) -> bool:
        """Check if a target exists."""
        return name in self._data

    @property
    def labels(self) -> list[str]:
        """Get list of label names."""
        return list(self._data.keys())


class ClassificationCollator:
    """Custom collator for classification dataset.

    Batches samples and creates DynamicTargets with only the labels present
    in the samples. Works with any subset of labels.
    """

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate samples into batch."""
        images = torch.stack([s["image"] for s in samples])

        # Get which labels are present in samples
        target_labels = list(samples[0]["targets"].keys())

        # Stack targets into tensors (only for present labels)
        targets_dict: dict[str, torch.Tensor] = {}

        for label in target_labels:
            task = get_task(label)
            if task.is_multiclass:
                dtype = torch.long
            else:
                dtype = torch.float32

            targets_dict[label] = torch.tensor(
                [s["targets"][label] for s in samples],
                dtype=dtype,
            )

        targets = DynamicTargets(targets_dict)

        level_idx = torch.tensor(
            [s["level_idx"] for s in samples],
            dtype=torch.long,
        )

        metadata = [s["metadata"] for s in samples]

        return {
            "image": images,
            "targets": targets,
            "level_idx": level_idx,
            "metadata": metadata,
        }

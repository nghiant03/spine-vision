"""Classification Dataset for lumbar spine multi-task learning.

Loads pre-extracted IVD crops from the classification dataset CLI.
Supports dual-modality (T1+T2) input construction with 3-channel output.

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
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms

from spine_vision.training.models import MTLTargets


# IVD level mapping (L1/L2 to L5/S1)
LEVEL_TO_IDX = {
    "L1/L2": 0,
    "L2/L3": 1,
    "L3/L4": 2,
    "L4/L5": 3,
    "L5/S1": 4,
}
IDX_TO_LEVEL = {v: k for k, v in LEVEL_TO_IDX.items()}


def construct_3channel(
    t2_crop: np.ndarray,
    t1_crop: np.ndarray,
) -> np.ndarray:
    """Construct 3-channel image from T1 and T2 crops.

    Channel layout: [T2, T1, T2] (RGB-like).

    Args:
        t2_crop: T2 crop, shape (H, W), uint8.
        t1_crop: T1 crop, shape (H, W), uint8.

    Returns:
        3-channel image, shape (H, W, 3), uint8.
    """
    return np.stack([t2_crop, t1_crop, t2_crop], axis=-1)


class ClassificationDataset(Dataset[dict[str, Any]]):
    """Dataset for lumbar spine classification using pre-extracted crops.

    Loads crops created by the `spine-vision dataset classification` CLI command.
    Automatically pairs T1 and T2 crops for the same patient/level to create
    3-channel inputs.

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
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
        levels: list[str] | None = None,
        output_size: tuple[int, int] = (224, 224),
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
            output_size: Final output size after resizing (H, W).
            augment: Apply data augmentation (training only).
            normalize: Apply ImageNet normalization.
            seed: Random seed for splitting.
        """
        self.data_path = Path(data_path)
        self.split = split
        self.output_size = output_size
        self.augment = augment and split == "train"
        self.normalize = normalize

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

        # Second pass: keep only complete pairs (both T1 and T2)
        records = []
        for group in groups.values():
            if group["t1_path"] is not None and group["t2_path"] is not None:
                records.append(group)

        return records

    def _get_unique_patients(self) -> list[str]:
        """Get list of unique patient keys (source_patientid)."""
        return list(set(r["patient_key"] for r in self.records))

    def _get_patient_stratify_labels(self, patients: list[str]) -> np.ndarray:
        """Get stratification labels for each patient.

        Uses the most frequent Pfirrmann grade across all IVD levels for the patient.
        This ensures class distribution is preserved across splits.
        """
        patient_to_pfirrmann: dict[str, list[int]] = {}
        for record in self.records:
            patient_key = record["patient_key"]
            if patient_key in set(patients):
                if patient_key not in patient_to_pfirrmann:
                    patient_to_pfirrmann[patient_key] = []
                patient_to_pfirrmann[patient_key].append(record["pfirrmann"])

        # Use most frequent Pfirrmann grade as the stratification label
        labels = []
        for patient in patients:
            grades = patient_to_pfirrmann.get(patient, [3])  # Default to grade 3
            most_common = Counter(grades).most_common(1)[0][0]
            labels.append(most_common)

        return np.array(labels)

    def _split_patients(
        self,
        patients: list[str],
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> tuple[set[str], set[str], set[str]]:
        """Split patients into train/val/test sets with stratification.

        Uses stratified splitting based on Pfirrmann grade distribution
        to ensure balanced class representation across splits.
        """
        patients_arr = np.array(patients)
        stratify_labels = self._get_patient_stratify_labels(patients)

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
            # Adjust val_ratio for remaining data
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

    def _build_transforms(self) -> Callable[[Image.Image], torch.Tensor]:
        """Build image transforms."""
        transform_list: list[Any] = [
            transforms.Resize(self.output_size),
        ]

        if self.augment:
            transform_list.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
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
                - image: 3-channel tensor [C, H, W] constructed as [T2, T1, T2]
                - targets: Dict compatible with MTLTargets
                - level_idx: Level index (0-4)
                - metadata: Dict with source, patient_id, level
        """
        record = self.records[idx]

        # Load T1 and T2 crops
        t1_img = Image.open(record["t1_path"]).convert("L")
        t2_img = Image.open(record["t2_path"]).convert("L")

        t1_arr = np.array(t1_img)
        t2_arr = np.array(t2_img)

        # Construct 3-channel image [T2, T1, T2]
        rgb_image = construct_3channel(t2_arr, t1_arr)

        # Convert to PIL for transforms
        pil_image = Image.fromarray(rgb_image)
        image_tensor = self.transform(pil_image)

        # Build targets
        targets = {
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
            "split": self.split,
        }

    def compute_class_weights(self) -> dict[str, torch.Tensor]:
        """Compute class weights for imbalanced classification.

        Uses inverse frequency weighting: weight = n_samples / (n_classes * count).
        For binary tasks, computes pos_weight for BCEWithLogitsLoss.

        Returns:
            Dictionary with class weights for each task:
                - pfirrmann: [5] tensor for 5 grades
                - modic: [4] tensor for 4 types
                - herniation: [1] tensor (pos_weight)
                - bulging: [1] tensor (pos_weight)
                - upper_endplate: [1] tensor (pos_weight)
                - lower_endplate: [1] tensor (pos_weight)
                - spondy: [1] tensor (pos_weight)
                - narrowing: [1] tensor (pos_weight)
        """
        n_samples = len(self.records)

        # Count multiclass labels
        pfirrmann_counts = Counter(r["pfirrmann"] - 1 for r in self.records)  # 0-4
        modic_counts = Counter(r["modic"] for r in self.records)  # 0-3

        # Count binary labels
        herniation_pos = sum(r["herniation"] for r in self.records)
        bulging_pos = sum(r["bulging"] for r in self.records)
        upper_endplate_pos = sum(r["upper_endplate"] for r in self.records)
        lower_endplate_pos = sum(r["lower_endplate"] for r in self.records)
        spondy_pos = sum(r["spondylolisthesis"] for r in self.records)
        narrowing_pos = sum(r["narrowing"] for r in self.records)

        # Compute multiclass weights: n_samples / (n_classes * count)
        pfirrmann_weights = torch.zeros(5)
        for i in range(5):
            count = pfirrmann_counts.get(i, 1)  # Avoid division by zero
            pfirrmann_weights[i] = n_samples / (5 * count)

        modic_weights = torch.zeros(4)
        for i in range(4):
            count = modic_counts.get(i, 1)
            modic_weights[i] = n_samples / (4 * count)

        # Compute binary pos_weight: n_negative / n_positive
        def pos_weight(n_pos: int) -> float:
            n_neg = n_samples - n_pos
            return n_neg / max(n_pos, 1)  # Avoid division by zero

        herniation_weights = torch.tensor([pos_weight(herniation_pos)])
        bulging_weights = torch.tensor([pos_weight(bulging_pos)])
        upper_endplate_weights = torch.tensor([pos_weight(upper_endplate_pos)])
        lower_endplate_weights = torch.tensor([pos_weight(lower_endplate_pos)])
        spondy_weights = torch.tensor([pos_weight(spondy_pos)])
        narrowing_weights = torch.tensor([pos_weight(narrowing_pos)])

        return {
            "pfirrmann": pfirrmann_weights,
            "modic": modic_weights,
            "herniation": herniation_weights,
            "bulging": bulging_weights,
            "upper_endplate": upper_endplate_weights,
            "lower_endplate": lower_endplate_weights,
            "spondy": spondy_weights,
            "narrowing": narrowing_weights,
        }


class ClassificationCollator:
    """Custom collator for classification dataset.

    Batches samples and creates MTLTargets.
    """

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate samples into batch."""
        images = torch.stack([s["image"] for s in samples])

        # Stack targets into tensors
        pfirrmann = torch.tensor(
            [s["targets"]["pfirrmann"] for s in samples],
            dtype=torch.long,
        )
        modic = torch.tensor(
            [s["targets"]["modic"] for s in samples],
            dtype=torch.long,
        )
        herniation = torch.tensor(
            [s["targets"]["herniation"] for s in samples],
            dtype=torch.float32,
        )
        bulging = torch.tensor(
            [s["targets"]["bulging"] for s in samples],
            dtype=torch.float32,
        )
        upper_endplate = torch.tensor(
            [s["targets"]["upper_endplate"] for s in samples],
            dtype=torch.float32,
        )
        lower_endplate = torch.tensor(
            [s["targets"]["lower_endplate"] for s in samples],
            dtype=torch.float32,
        )
        spondy = torch.tensor(
            [s["targets"]["spondy"] for s in samples],
            dtype=torch.float32,
        )
        narrowing = torch.tensor(
            [s["targets"]["narrowing"] for s in samples],
            dtype=torch.float32,
        )

        targets = MTLTargets(
            pfirrmann=pfirrmann,
            modic=modic,
            herniation=herniation,
            bulging=bulging,
            upper_endplate=upper_endplate,
            lower_endplate=lower_endplate,
            spondy=spondy,
            narrowing=narrowing,
        )

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

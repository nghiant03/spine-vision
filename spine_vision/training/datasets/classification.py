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
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from spine_vision.core.tasks import AVAILABLE_TASK_NAMES, get_task
from spine_vision.datasets.levels import IDX_TO_LEVEL
from spine_vision.training.datasets.sampling import create_weighted_sampler
from spine_vision.training.datasets.stratification import split_patients


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
            data_path: Path to dataset directory.
            split: Data split ('train', 'val', 'test', 'all').
            val_ratio: Fraction for validation.
            test_ratio: Fraction for testing.
            levels: Filter to specific IVD levels (e.g., ["L4/L5", "L5/S1"]).
            series_types: Filter to specific series types.
            target_labels: Filter to specific labels.
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
            self.series_types = valid_series

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
        train_patients, val_patients, test_patients = split_patients(
            unique_patients,
            self.records,
            self.target_labels,
            val_ratio,
            test_ratio,
            seed,
        )

        if split == "train":
            self.records = [
                r for r in self.records if r["patient_key"] in train_patients
            ]
        elif split == "val":
            self.records = [r for r in self.records if r["patient_key"] in val_patients]
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
                        "level_idx": ivd_level - 1,
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

            if require_t1 and require_t2:
                if has_t1 and has_t2:
                    records.append(group)
            elif require_t1:
                if has_t1:
                    records.append(group)
            elif require_t2:
                if has_t2:
                    records.append(group)

        return records

    def _get_unique_patients(self) -> list[str]:
        """Get list of unique patient keys."""
        return list(set(r["patient_key"] for r in self.records))

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
        """Get a single sample."""
        record = self.records[idx]

        # Load T1 and/or T2 crops
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
        """Get distribution of each label in the dataset."""
        distribution: dict[str, dict[int | str, int]] = {}

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
        """Compute class weights for imbalanced classification."""
        n_samples = len(self.records)
        weights: dict[str, torch.Tensor] = {}

        def pos_weight(n_pos: int) -> float:
            n_neg = n_samples - n_pos
            return n_neg / max(n_pos, 1)

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


class DynamicTargets:
    """Container for dynamic multi-task targets.

    Provides the same interface (to(), to_dict()) for compatibility with
    fixed-field containers.
    """

    def __init__(self, data: dict[str, torch.Tensor]) -> None:
        """Initialize with target tensors."""
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

    Batches samples and creates DynamicTargets with only the labels present.
    """

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate samples into batch."""
        images = torch.stack([s["image"] for s in samples])

        target_labels = list(samples[0]["targets"].keys())

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


# Re-export for backwards compatibility
__all__ = [
    "ClassificationDataset",
    "ClassificationCollator",
    "DynamicTargets",
    "construct_3channel",
    "create_weighted_sampler",
]

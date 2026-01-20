"""Localization Dataset for training.

Loads images and annotations from the localization dataset created by
spine_vision.datasets.localization.

The dataset groups all level annotations per image, returning all 5 IVD
level coordinates in a single sample. This allows the model to predict
all locations in a single forward pass.
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# IVD level mapping (L1/L2 to L5/S1)
LEVEL_TO_IDX = {
    "L1/L2": 0,
    "L2/L3": 1,
    "L3/L4": 2,
    "L4/L5": 3,
    "L5/S1": 4,
}
IDX_TO_LEVEL = {v: k for k, v in LEVEL_TO_IDX.items()}
NUM_LEVELS = len(LEVEL_TO_IDX)

# Series type mapping
SERIES_TYPE_TO_IDX = {
    "sag_t1": 0,
    "sag_t2": 1,
    "ct": 2,
}


class LocalizationDataset(Dataset[dict[str, Any]]):
    """Dataset for coordinate localization.

    Loads images and relative (x, y) coordinates for all IVD levels.
    Each sample contains one image with coordinates for all 5 levels.

    Annotations CSV format:
        image_path, level, relative_x, relative_y, series_type, source

    Returns per sample:
        - image: Tensor [C, H, W]
        - coords: Tensor [5, 2] with coordinates for all levels
        - mask: Tensor [5] indicating valid levels (1=valid, 0=missing)
        - series_type_idx: Series type index
        - metadata: Dict with image_path, source, series_type
    """

    def __init__(
        self,
        data_path: Path,
        split: Literal["train", "val", "test", "all"] = "all",
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
        series_types: list[str] | None = None,
        sources: list[str] | None = None,
        image_size: tuple[int, int] = (224, 224),
        augment: bool = True,
        normalize: bool = True,
        seed: int = 42,
    ) -> None:
        """Initialize dataset.

        Args:
            data_path: Path to dataset directory (contains images/ and annotations.csv).
            split: Data split to use ('train', 'val', 'test', 'all').
            val_ratio: Fraction of data for validation.
            test_ratio: Fraction of data for testing.
            series_types: Filter to specific series types (e.g., ['sag_t1', 'sag_t2']).
            sources: Filter to specific sources (e.g., ['rsna', 'pretrain_spider']).
            image_size: Target image size (H, W).
            augment: Apply data augmentation (for training).
            normalize: Apply ImageNet normalization.
            seed: Random seed for splitting.
        """
        self.data_path = Path(data_path)
        self.split = split
        self.image_size = image_size
        self.augment = augment and split == "train"
        self.normalize = normalize

        # Load annotations
        annotations_path = self.data_path / "annotations.csv"
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations not found: {annotations_path}")

        raw_records = self._load_annotations(annotations_path)

        # Filter by criteria
        if series_types:
            raw_records = [r for r in raw_records if r["series_type"] in series_types]
        if sources:
            raw_records = [r for r in raw_records if r["source"] in sources]

        # Group records by image path
        self.image_records = self._group_by_image(raw_records)

        # Get unique images and split
        unique_images = list(self.image_records.keys())
        train_images, val_images, test_images = self._split_images(
            unique_images, val_ratio, test_ratio, seed
        )

        # Filter by split
        if split == "train":
            self.image_list = [img for img in unique_images if img in train_images]
        elif split == "val":
            self.image_list = [img for img in unique_images if img in val_images]
        elif split == "test":
            self.image_list = [img for img in unique_images if img in test_images]
        else:
            self.image_list = unique_images

        # Build transforms
        self.transform = self._build_transforms()

    def _load_annotations(self, path: Path) -> list[dict[str, Any]]:
        """Load annotations from CSV."""
        records = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(
                    {
                        "image_path": row["image_path"],
                        "level": row["level"],
                        "relative_x": float(row["relative_x"]),
                        "relative_y": float(row["relative_y"]),
                        "series_type": row["series_type"],
                        "source": row["source"],
                    }
                )
        return records

    def _group_by_image(
        self, records: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Group records by image path.

        Returns:
            Dict mapping image_path to:
                - coords: Dict[level_idx, (x, y)]
                - series_type: str
                - source: str
        """
        grouped: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"coords": {}, "series_type": "", "source": ""}
        )

        for record in records:
            img_path = record["image_path"]
            level_idx = LEVEL_TO_IDX.get(record["level"])
            if level_idx is not None:
                grouped[img_path]["coords"][level_idx] = (
                    record["relative_x"],
                    record["relative_y"],
                )
                grouped[img_path]["series_type"] = record["series_type"]
                grouped[img_path]["source"] = record["source"]

        return dict(grouped)

    def _split_images(
        self,
        images: list[str],
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> tuple[set[str], set[str], set[str]]:
        """Split images into train/val/test sets."""
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(images))

        n_test = int(len(images) * test_ratio)
        n_val = int(len(images) * val_ratio)

        test_idx = indices[:n_test]
        val_idx = indices[n_test : n_test + n_val]
        train_idx = indices[n_test + n_val :]

        return (
            set(images[i] for i in train_idx),
            set(images[i] for i in val_idx),
            set(images[i] for i in test_idx),
        )

    def _build_transforms(self) -> transforms.Compose:
        """Build image transforms."""
        transform_list: list[Any] = [
            transforms.Resize(self.image_size),
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

        transform_list.extend(
            [
                transforms.ToTensor(),
            ]
        )

        if self.normalize:
            # ImageNet normalization
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )

        return transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Returns:
            Dictionary with keys:
                - image: Transformed image tensor [C, H, W]
                - coords: Coordinates tensor [5, 2] for all levels
                - mask: Validity mask [5] (1=valid, 0=missing)
                - series_type_idx: Series type index
                - metadata: Dict with image_path, source, series_type
        """
        image_path = self.image_list[idx]
        record = self.image_records[image_path]

        # Load image
        full_path = self.data_path / image_path
        image = Image.open(full_path).convert("RGB")

        # Apply transforms
        image_tensor = self.transform(image)

        # Build coordinates array [5, 2] and mask [5]
        coords = torch.zeros(NUM_LEVELS, 2, dtype=torch.float32)
        mask = torch.zeros(NUM_LEVELS, dtype=torch.float32)

        for level_idx, (x, y) in record["coords"].items():
            coords[level_idx, 0] = x
            coords[level_idx, 1] = y
            mask[level_idx] = 1.0

        # Series type encoding
        series_type_idx = SERIES_TYPE_TO_IDX.get(record["series_type"], 0)

        return {
            "image": image_tensor,
            "coords": coords,
            "mask": mask,
            "series_type_idx": series_type_idx,
            "metadata": {
                "image_path": image_path,
                "source": record["source"],
                "series_type": record["series_type"],
            },
        }

    def get_stats(self) -> dict[str, Any]:
        """Get dataset statistics."""
        from collections import Counter

        series_types: list[str] = []
        sources: list[str] = []
        level_counts: dict[int, int] = defaultdict(int)
        total_annotations = 0

        for image_path in self.image_list:
            record = self.image_records[image_path]
            series_types.append(record["series_type"])
            sources.append(record["source"])
            for level_idx in record["coords"]:
                level_counts[level_idx] += 1
                total_annotations += 1

        # Convert level counts to level names
        level_distribution = {
            IDX_TO_LEVEL[idx]: count for idx, count in sorted(level_counts.items())
        }

        return {
            "num_images": len(self.image_list),
            "num_annotations": total_annotations,
            "levels": level_distribution,
            "series_types": dict(Counter(series_types)),
            "sources": dict(Counter(sources)),
            "split": self.split,
        }


class LocalizationCollator:
    """Custom collator for localization dataset.

    Batches samples and handles variable-length metadata.
    """

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate samples into batch."""
        images = torch.stack([s["image"] for s in samples])
        coords = torch.stack([s["coords"] for s in samples])  # [B, 5, 2]
        mask = torch.stack([s["mask"] for s in samples])  # [B, 5]
        series_type_idx = torch.tensor(
            [s["series_type_idx"] for s in samples], dtype=torch.long
        )
        metadata = [s["metadata"] for s in samples]

        return {
            "image": images,
            "coords": coords,
            "mask": mask,
            "series_type_idx": series_type_idx,
            "metadata": metadata,
        }

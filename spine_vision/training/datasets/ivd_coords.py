"""IVD Coordinates Dataset for localization training.

Loads images and annotations from the IVD coordinates dataset created by
spine_vision.datasets.ivd_coords.
"""

import csv
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


class IVDCoordsDataset(Dataset[dict[str, Any]]):
    """Dataset for IVD coordinate localization.

    Loads images and relative (x, y) coordinates for IVD localization.
    Supports filtering by series type, level, and data source.

    Annotations CSV format:
        image_path, level, relative_x, relative_y, series_type, source
    """

    def __init__(
        self,
        data_path: Path,
        split: Literal["train", "val", "test", "all"] = "all",
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
        series_types: list[str] | None = None,
        levels: list[str] | None = None,
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
            levels: Filter to specific levels (e.g., ['L4/L5', 'L5/S1']).
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

        self.records = self._load_annotations(annotations_path)

        # Filter by criteria
        if series_types:
            self.records = [r for r in self.records if r["series_type"] in series_types]
        if levels:
            self.records = [r for r in self.records if r["level"] in levels]
        if sources:
            self.records = [r for r in self.records if r["source"] in sources]

        # Group by unique images for proper splitting
        unique_images = self._get_unique_images()
        train_images, val_images, test_images = self._split_images(
            unique_images, val_ratio, test_ratio, seed
        )

        # Filter records by split
        if split == "train":
            self.records = [r for r in self.records if r["image_path"] in train_images]
        elif split == "val":
            self.records = [r for r in self.records if r["image_path"] in val_images]
        elif split == "test":
            self.records = [r for r in self.records if r["image_path"] in test_images]
        # else: keep all

        # Build transforms
        self.transform = self._build_transforms()

    def _load_annotations(self, path: Path) -> list[dict[str, Any]]:
        """Load annotations from CSV."""
        records = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "image_path": row["image_path"],
                    "level": row["level"],
                    "relative_x": float(row["relative_x"]),
                    "relative_y": float(row["relative_y"]),
                    "series_type": row["series_type"],
                    "source": row["source"],
                })
        return records

    def _get_unique_images(self) -> list[str]:
        """Get list of unique image paths."""
        return list(set(r["image_path"] for r in self.records))

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
            transform_list.extend([
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
            ])

        transform_list.extend([
            transforms.ToTensor(),
        ])

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
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Returns:
            Dictionary with keys:
                - image: Transformed image tensor [C, H, W]
                - coords: Coordinates tensor [2] (relative_x, relative_y)
                - level_idx: Level index (0-4)
                - series_type_idx: Series type index
                - metadata: Dict with image_path, level, source
        """
        record = self.records[idx]

        # Load image
        image_path = self.data_path / record["image_path"]
        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        image_tensor = self.transform(image)

        # Coordinates
        coords = torch.tensor(
            [record["relative_x"], record["relative_y"]],
            dtype=torch.float32,
        )

        # Level encoding
        level_idx = LEVEL_TO_IDX.get(record["level"], 0)

        # Series type encoding
        series_type_idx = SERIES_TYPE_TO_IDX.get(record["series_type"], 0)

        return {
            "image": image_tensor,
            "coords": coords,
            "level_idx": level_idx,
            "series_type_idx": series_type_idx,
            "metadata": {
                "image_path": record["image_path"],
                "level": record["level"],
                "source": record["source"],
                "series_type": record["series_type"],
            },
        }

    def get_stats(self) -> dict[str, Any]:
        """Get dataset statistics."""
        levels = [r["level"] for r in self.records]
        series_types = [r["series_type"] for r in self.records]
        sources = [r["source"] for r in self.records]

        from collections import Counter

        return {
            "num_samples": len(self.records),
            "num_unique_images": len(self._get_unique_images()),
            "levels": dict(Counter(levels)),
            "series_types": dict(Counter(series_types)),
            "sources": dict(Counter(sources)),
            "split": self.split,
        }


class IVDCoordsCollator:
    """Custom collator for IVD coordinates dataset.

    Batches samples and handles variable-length metadata.
    """

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate samples into batch."""
        images = torch.stack([s["image"] for s in samples])
        coords = torch.stack([s["coords"] for s in samples])
        level_idx = torch.tensor([s["level_idx"] for s in samples], dtype=torch.long)
        series_type_idx = torch.tensor(
            [s["series_type_idx"] for s in samples], dtype=torch.long
        )
        metadata = [s["metadata"] for s in samples]

        return {
            "image": images,
            "coords": coords,
            "level_idx": level_idx,
            "series_type_idx": series_type_idx,
            "metadata": metadata,
        }

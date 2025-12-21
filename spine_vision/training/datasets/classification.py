"""Classification Dataset for lumbar spine multi-task learning.

Loads high-resolution crops from original DICOM images based on localization
coordinates, supporting dual-modality (T1+T2) input construction.

Key Features:
- Coordinate mapping from 224x224 localization space to original resolution
- Dual-modality cropping (T1 and T2 slices)
- 3-channel construction: [T2, T1, T2] for RGB-like input
- Support for 13 clinical labels across 6 task heads
"""

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from spine_vision.training.models.resnet_mtl import MTLTargets


# IVD level mapping (L1/L2 to L5/S1)
LEVEL_TO_IDX = {
    "L1/L2": 0,
    "L2/L3": 1,
    "L3/L4": 2,
    "L4/L5": 3,
    "L5/S1": 4,
}
IDX_TO_LEVEL = {v: k for k, v in LEVEL_TO_IDX.items()}
LEVEL_TO_IVD = {
    "L1/L2": 1,
    "L2/L3": 2,
    "L3/L4": 3,
    "L4/L5": 4,
    "L5/S1": 5,
}

# Label column names in annotations
LABEL_COLUMNS = {
    "pfirrmann": "pfirrmann",  # int 1-5
    "modic": "modic",  # int 0-3
    "herniation": "herniation",  # int 0 or 1
    "bulging": "bulging",  # int 0 or 1
    "upper_endplate": "upper_endplate",  # int 0 or 1
    "lower_endplate": "lower_endplate",  # int 0 or 1
    "spondylolisthesis": "spondylolisthesis",  # int 0 or 1
    "narrowing": "narrowing",  # int 0 or 1
}


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize array to uint8 range [0, 255]."""
    arr = arr.astype(np.float32)
    min_val, max_val = arr.min(), arr.max()
    if max_val - min_val > 0:
        arr = (arr - min_val) / (max_val - min_val) * 255
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.uint8)


def map_coordinates(
    coords: tuple[float, float],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int]:
    """Map coordinates from source to target resolution.
    
    Args:
        coords: (x, y) relative coordinates in [0, 1] or pixel coords.
        source_size: (width, height) of source image (localization model).
        target_size: (width, height) of target image (original DICOM).
    
    Returns:
        (x, y) pixel coordinates in target resolution.
    """
    x, y = coords
    src_w, src_h = source_size
    tgt_w, tgt_h = target_size
    
    # If coords are relative (0-1), convert to source pixels first
    if 0 <= x <= 1 and 0 <= y <= 1:
        x = x * src_w
        y = y * src_h
    
    # Scale to target resolution
    scale_x = tgt_w / src_w
    scale_y = tgt_h / src_h
    
    return int(x * scale_x), int(y * scale_y)


def crop_square(
    image: np.ndarray,
    center: tuple[int, int],
    size: int,
    fill_value: int = 0,
) -> np.ndarray:
    """Crop a square region from image centered at given point.
    
    Handles edge cases with padding.
    
    Args:
        image: 2D numpy array.
        center: (x, y) center point for crop.
        size: Side length of square crop.
        fill_value: Value to use for padding at edges.
    
    Returns:
        Square crop of shape (size, size).
    """
    h, w = image.shape[:2]
    x, y = center
    half = size // 2
    
    # Calculate source region bounds
    x1, y1 = x - half, y - half
    x2, y2 = x + half, y + half
    
    # Calculate valid source region (clipped to image bounds)
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)
    
    # Calculate corresponding destination region
    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = size - (x2 - src_x2)
    dst_y2 = size - (y2 - src_y2)
    
    # Create padded output
    crop = np.full((size, size), fill_value, dtype=image.dtype)
    
    # Copy valid region
    if src_x2 > src_x1 and src_y2 > src_y1:
        crop[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
    
    return crop


def load_dicom_slice(path: Path, slice_idx: int | None = None) -> np.ndarray:
    """Load a single 2D slice from DICOM file or directory.
    
    Args:
        path: Path to DICOM file or directory containing series.
        slice_idx: Slice index for 3D volumes. If None, uses middle slice.
    
    Returns:
        2D numpy array.
    """
    if path.is_dir():
        # Load DICOM series from directory
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(str(path))
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {path}")
        reader.SetFileNames(dicom_files)
        image = reader.Execute()
    else:
        # Load single DICOM file
        image = sitk.ReadImage(str(path))
    
    arr = sitk.GetArrayFromImage(image)
    
    # Handle 3D volumes
    if arr.ndim == 3:
        if slice_idx is None:
            slice_idx = arr.shape[0] // 2
        arr = arr[slice_idx]
    
    return arr


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
    """Dataset for lumbar spine classification with dual-modality crops.
    
    Loads high-resolution crops from original DICOM/NIfTI images based on
    localization coordinates from a separate localization model.
    
    Expected directory structure:
        data_path/
            images/
                <study_id>/
                    sag_t1/ or sag_t1.nii.gz
                    sag_t2/ or sag_t2.nii.gz
            annotations.csv  (or annotations.json)
    
    Annotations format (CSV):
        study_id, level, coord_x, coord_y, pfirrmann, modic, herniation,
        bulging, upper_endplate, lower_endplate, spondylolisthesis, narrowing,
        [optional: t1_path, t2_path, slice_idx]
    
    Annotations format (JSON):
        [
            {
                "study_id": "...",
                "patient_id": "...",
                "level": "L4/L5",
                "coord_x": 0.5,
                "coord_y": 0.6,
                "pfirrmann": 3,
                "modic": 0,
                ...
            },
            ...
        ]
    """
    
    def __init__(
        self,
        data_path: Path,
        split: Literal["train", "val", "test", "all"] = "all",
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
        levels: list[str] | None = None,
        localization_size: tuple[int, int] = (224, 224),
        crop_size: int = 128,
        output_size: tuple[int, int] = (224, 224),
        augment: bool = True,
        normalize: bool = True,
        seed: int = 42,
        t1_subdir: str = "sag_t1",
        t2_subdir: str = "sag_t2",
    ) -> None:
        """Initialize dataset.
        
        Args:
            data_path: Path to dataset directory.
            split: Data split ('train', 'val', 'test', 'all').
            val_ratio: Fraction for validation.
            test_ratio: Fraction for testing.
            levels: Filter to specific IVD levels.
            localization_size: Size used by localization model (for coord mapping).
            crop_size: Size of square crop from original image (pixels).
            output_size: Final output size after resizing (H, W).
            augment: Apply data augmentation (training only).
            normalize: Apply ImageNet normalization.
            seed: Random seed for splitting.
            t1_subdir: Subdirectory or suffix for T1 images.
            t2_subdir: Subdirectory or suffix for T2 images.
        """
        self.data_path = Path(data_path)
        self.split = split
        self.localization_size = localization_size
        self.crop_size = crop_size
        self.output_size = output_size
        self.augment = augment and split == "train"
        self.normalize = normalize
        self.t1_subdir = t1_subdir
        self.t2_subdir = t2_subdir
        
        # Load annotations
        self.records = self._load_annotations()
        
        # Filter by level
        if levels:
            self.records = [r for r in self.records if r["level"] in levels]
        
        # Split by study_id to avoid data leakage
        unique_studies = self._get_unique_studies()
        train_studies, val_studies, test_studies = self._split_studies(
            unique_studies, val_ratio, test_ratio, seed
        )
        
        if split == "train":
            self.records = [r for r in self.records if r["study_id"] in train_studies]
        elif split == "val":
            self.records = [r for r in self.records if r["study_id"] in val_studies]
        elif split == "test":
            self.records = [r for r in self.records if r["study_id"] in test_studies]
        
        # Build transforms
        self.transform = self._build_transforms()
        
        # Cache for loaded images (optional)
        self._image_cache: dict[str, np.ndarray] = {}
    
    def _load_annotations(self) -> list[dict[str, Any]]:
        """Load annotations from CSV or JSON file."""
        csv_path = self.data_path / "annotations.csv"
        json_path = self.data_path / "annotations.json"
        
        if csv_path.exists():
            return self._load_csv_annotations(csv_path)
        elif json_path.exists():
            return self._load_json_annotations(json_path)
        else:
            raise FileNotFoundError(
                f"No annotations found at {csv_path} or {json_path}"
            )
    
    def _load_csv_annotations(self, path: Path) -> list[dict[str, Any]]:
        """Load annotations from CSV file."""
        records = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                record = {
                    "study_id": row["study_id"],
                    "patient_id": row.get("patient_id", row["study_id"]),
                    "level": row["level"],
                    "coord_x": float(row["coord_x"]),
                    "coord_y": float(row["coord_y"]),
                    # Labels
                    "pfirrmann": int(row["pfirrmann"]),
                    "modic": int(row["modic"]),
                    "herniation": int(row["herniation"]),
                    "bulging": int(row["bulging"]),
                    "upper_endplate": int(row["upper_endplate"]),
                    "lower_endplate": int(row["lower_endplate"]),
                    "spondylolisthesis": int(row["spondylolisthesis"]),
                    "narrowing": int(row["narrowing"]),
                }
                # Optional fields
                if "t1_path" in row:
                    record["t1_path"] = row["t1_path"]
                if "t2_path" in row:
                    record["t2_path"] = row["t2_path"]
                if "slice_idx" in row:
                    record["slice_idx"] = int(row["slice_idx"])
                
                records.append(record)
        return records
    
    def _load_json_annotations(self, path: Path) -> list[dict[str, Any]]:
        """Load annotations from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        records = []
        for item in data:
            record = {
                "study_id": item["study_id"],
                "patient_id": item.get("patient_id", item["study_id"]),
                "level": item["level"],
                "coord_x": float(item["coord_x"]),
                "coord_y": float(item["coord_y"]),
                "pfirrmann": int(item["pfirrmann"]),
                "modic": int(item["modic"]),
                "herniation": int(item["herniation"]),
                "bulging": int(item["bulging"]),
                "upper_endplate": int(item["upper_endplate"]),
                "lower_endplate": int(item["lower_endplate"]),
                "spondylolisthesis": int(item["spondylolisthesis"]),
                "narrowing": int(item["narrowing"]),
            }
            # Optional fields
            for key in ["t1_path", "t2_path", "slice_idx"]:
                if key in item:
                    record[key] = item[key]
            records.append(record)
        return records
    
    def _get_unique_studies(self) -> list[str]:
        """Get list of unique study IDs."""
        return list(set(r["study_id"] for r in self.records))
    
    def _split_studies(
        self,
        studies: list[str],
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> tuple[set[str], set[str], set[str]]:
        """Split studies into train/val/test sets."""
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(studies))
        
        n_test = int(len(studies) * test_ratio)
        n_val = int(len(studies) * val_ratio)
        
        test_idx = indices[:n_test]
        val_idx = indices[n_test : n_test + n_val]
        train_idx = indices[n_test + n_val :]
        
        return (
            set(studies[i] for i in train_idx),
            set(studies[i] for i in val_idx),
            set(studies[i] for i in test_idx),
        )
    
    def _build_transforms(self) -> Callable[[Image.Image], torch.Tensor]:
        """Build image transforms."""
        transform_list: list[Any] = [
            transforms.Resize(self.output_size),
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
        
        transform_list.append(transforms.ToTensor())
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
        
        return transforms.Compose(transform_list)
    
    def _load_image(
        self,
        study_id: str,
        modality: str,
        record: dict[str, Any],
    ) -> np.ndarray:
        """Load and cache image for a study/modality.
        
        Args:
            study_id: Study identifier.
            modality: "t1" or "t2".
            record: Record dict (may contain custom path).
        
        Returns:
            2D numpy array (H, W).
        """
        cache_key = f"{study_id}_{modality}"
        if cache_key in self._image_cache:
            return self._image_cache[cache_key]
        
        # Check for custom path in record
        path_key = f"{modality}_path"
        if path_key in record:
            image_path = self.data_path / record[path_key]
        else:
            # Default path structure
            subdir = self.t1_subdir if modality == "t1" else self.t2_subdir
            study_path = self.data_path / "images" / study_id
            
            # Try directory (DICOM series)
            image_path = study_path / subdir
            if not image_path.exists():
                # Try NIfTI file
                image_path = study_path / f"{subdir}.nii.gz"
                if not image_path.exists():
                    image_path = study_path / f"{subdir}.nii"
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        slice_idx = record.get("slice_idx", None)
        arr = load_dicom_slice(image_path, slice_idx)
        
        # Normalize to uint8
        arr = normalize_to_uint8(arr)
        
        # Cache (be careful with memory for large datasets)
        # self._image_cache[cache_key] = arr
        
        return arr
    
    def _get_crop(
        self,
        study_id: str,
        modality: str,
        coord_x: float,
        coord_y: float,
        record: dict[str, Any],
    ) -> np.ndarray:
        """Get cropped region from image.
        
        Maps coordinates from localization space to original image space,
        then extracts a square crop.
        
        Args:
            study_id: Study identifier.
            modality: "t1" or "t2".
            coord_x: X coordinate in localization space (relative 0-1).
            coord_y: Y coordinate in localization space (relative 0-1).
            record: Full record dict.
        
        Returns:
            Square crop, shape (crop_size, crop_size), uint8.
        """
        # Load original image
        image = self._load_image(study_id, modality, record)
        h, w = image.shape[:2]
        
        # Map coordinates from localization space to original
        loc_w, loc_h = self.localization_size
        x = int(coord_x * w)
        y = int(coord_y * h)
        
        # Extract crop
        crop = crop_square(image, (x, y), self.crop_size)
        
        return crop
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.
        
        Returns:
            Dictionary with keys:
                - image: 3-channel tensor [C, H, W]
                - targets: MTLTargets compatible dict
                - level_idx: Level index (0-4)
                - metadata: Dict with study_id, patient_id, level
        """
        record = self.records[idx]
        
        # Get crops for both modalities
        t2_crop = self._get_crop(
            record["study_id"],
            "t2",
            record["coord_x"],
            record["coord_y"],
            record,
        )
        t1_crop = self._get_crop(
            record["study_id"],
            "t1",
            record["coord_x"],
            record["coord_y"],
            record,
        )
        
        # Construct 3-channel image [T2, T1, T2]
        rgb_image = construct_3channel(t2_crop, t1_crop)
        
        # Convert to PIL for transforms
        pil_image = Image.fromarray(rgb_image)
        image_tensor = self.transform(pil_image)
        
        # Build targets
        level_idx = LEVEL_TO_IDX.get(record["level"], 0)
        
        targets = {
            # Pfirrmann: convert 1-5 to 0-4 for CrossEntropy
            "pfirrmann": record["pfirrmann"] - 1,
            "modic": record["modic"],
            "herniation": [
                float(record["herniation"]),
                float(record["bulging"]),
            ],
            "endplate": [
                float(record["upper_endplate"]),
                float(record["lower_endplate"]),
            ],
            "spondy": [float(record["spondylolisthesis"])],
            "narrowing": [float(record["narrowing"])],
        }
        
        return {
            "image": image_tensor,
            "targets": targets,
            "level_idx": level_idx,
            "metadata": {
                "study_id": record["study_id"],
                "patient_id": record["patient_id"],
                "level": record["level"],
                "ivd": LEVEL_TO_IVD.get(record["level"], 0),
            },
        }
    
    def get_stats(self) -> dict[str, Any]:
        """Get dataset statistics."""
        from collections import Counter
        
        levels = [r["level"] for r in self.records]
        pfirrmann = [r["pfirrmann"] for r in self.records]
        modic = [r["modic"] for r in self.records]
        
        return {
            "num_samples": len(self.records),
            "num_studies": len(self._get_unique_studies()),
            "levels": dict(Counter(levels)),
            "pfirrmann": dict(Counter(pfirrmann)),
            "modic": dict(Counter(modic)),
            "split": self.split,
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
        endplate = torch.tensor(
            [s["targets"]["endplate"] for s in samples],
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
            endplate=endplate,
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


class CropOnlyDataset(Dataset[dict[str, Any]]):
    """Dataset that uses pre-extracted crops instead of loading from DICOM.
    
    Use this when crops have already been extracted and saved as image files.
    Faster than loading from DICOM each time.
    
    Expected structure:
        data_path/
            crops/
                <study_id>_<level>_t1.png
                <study_id>_<level>_t2.png
            annotations.csv
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
        """Initialize dataset with pre-extracted crops."""
        self.data_path = Path(data_path)
        self.split = split
        self.output_size = output_size
        self.augment = augment and split == "train"
        self.normalize = normalize
        
        # Load annotations
        self.records = self._load_annotations()
        
        if levels:
            self.records = [r for r in self.records if r["level"] in levels]
        
        # Split by study
        unique_studies = list(set(r["study_id"] for r in self.records))
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(unique_studies))
        
        n_test = int(len(unique_studies) * test_ratio)
        n_val = int(len(unique_studies) * val_ratio)
        
        test_studies = set(unique_studies[i] for i in indices[:n_test])
        val_studies = set(unique_studies[i] for i in indices[n_test:n_test + n_val])
        train_studies = set(unique_studies[i] for i in indices[n_test + n_val:])
        
        if split == "train":
            self.records = [r for r in self.records if r["study_id"] in train_studies]
        elif split == "val":
            self.records = [r for r in self.records if r["study_id"] in val_studies]
        elif split == "test":
            self.records = [r for r in self.records if r["study_id"] in test_studies]
        
        self.transform = self._build_transforms()
    
    def _load_annotations(self) -> list[dict[str, Any]]:
        """Load annotations from CSV."""
        csv_path = self.data_path / "annotations.csv"
        records = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "study_id": row["study_id"],
                    "patient_id": row.get("patient_id", row["study_id"]),
                    "level": row["level"],
                    "pfirrmann": int(row["pfirrmann"]),
                    "modic": int(row["modic"]),
                    "herniation": int(row["herniation"]),
                    "bulging": int(row["bulging"]),
                    "upper_endplate": int(row["upper_endplate"]),
                    "lower_endplate": int(row["lower_endplate"]),
                    "spondylolisthesis": int(row["spondylolisthesis"]),
                    "narrowing": int(row["narrowing"]),
                })
        return records
    
    def _build_transforms(self) -> Callable[[Image.Image], torch.Tensor]:
        """Build transforms."""
        transform_list: list[Any] = [transforms.Resize(self.output_size)]
        
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        
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
        """Load pre-extracted crops and construct 3-channel input."""
        record = self.records[idx]
        study_id = record["study_id"]
        level = record["level"].replace("/", "_")  # L4/L5 -> L4_L5
        
        # Load crops
        t1_path = self.data_path / "crops" / f"{study_id}_{level}_t1.png"
        t2_path = self.data_path / "crops" / f"{study_id}_{level}_t2.png"
        
        t1_img = Image.open(t1_path).convert("L")
        t2_img = Image.open(t2_path).convert("L")
        
        t1_arr = np.array(t1_img)
        t2_arr = np.array(t2_img)
        
        # Construct 3-channel
        rgb = construct_3channel(t2_arr, t1_arr)
        pil_img = Image.fromarray(rgb)
        
        image_tensor = self.transform(pil_img)
        
        level_idx = LEVEL_TO_IDX.get(record["level"], 0)
        
        targets = {
            "pfirrmann": record["pfirrmann"] - 1,
            "modic": record["modic"],
            "herniation": [float(record["herniation"]), float(record["bulging"])],
            "endplate": [float(record["upper_endplate"]), float(record["lower_endplate"])],
            "spondy": [float(record["spondylolisthesis"])],
            "narrowing": [float(record["narrowing"])],
        }
        
        return {
            "image": image_tensor,
            "targets": targets,
            "level_idx": level_idx,
            "metadata": {
                "study_id": record["study_id"],
                "patient_id": record["patient_id"],
                "level": record["level"],
                "ivd": LEVEL_TO_IVD.get(record["level"], 0),
            },
        }
    
    def get_stats(self) -> dict[str, Any]:
        """Get dataset statistics."""
        from collections import Counter
        return {
            "num_samples": len(self.records),
            "num_studies": len(set(r["study_id"] for r in self.records)),
            "levels": dict(Counter(r["level"] for r in self.records)),
            "split": self.split,
        }

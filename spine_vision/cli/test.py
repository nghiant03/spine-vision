"""Test inference CLI for trained models.

Supports testing models with images or DICOM slices. Task-dependent
inference is performed based on the model type (localization, classification).
"""

from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
import SimpleITK as sitk
import torch
from loguru import logger
from PIL import Image
from pydantic import model_validator

from spine_vision.core import BaseConfig, setup_logger
from spine_vision.io import normalize_to_uint8, read_medical_image

if TYPE_CHECKING:
    pass

TModel = TypeVar("TModel", bound=torch.nn.Module)


class TestConfig(BaseConfig):
    """Configuration for model testing.

    Supports localization and classification tasks with various input formats.
    """

    # Required parameters
    model_path: Path
    """Path to the trained model checkpoint (.pt file)."""

    inputs: list[Path]
    """List of input image or DICOM file paths."""

    # Task configuration
    task: Literal["localization", "classification"] = "localization"
    """Task type determines model architecture and output format."""

    model_variant: Literal[
        "tiny", "small", "base", "large", "xlarge",
        "v2_tiny", "v2_small", "v2_base", "v2_large", "v2_huge",
    ] = "base"
    """ConvNext model variant to use."""

    # Localization-specific options
    level_indices: list[int] | None = None
    """IVD level indices (0-4) for localization. One per input image."""

    num_levels: int = 5
    """Number of IVD levels the model was trained on."""

    use_level_embedding: bool = True
    """Whether the model uses level embedding."""

    # Classification-specific options
    num_classes: int = 4
    """Number of classes for classification task."""

    class_names: list[str] | None = None
    """Optional class names for output formatting."""

    # Inference parameters
    image_size: tuple[int, int] = (224, 224)
    """Target image size (H, W) for model input."""

    device: str = "cuda:0"
    """Device to run inference on."""

    # Output options
    output_path: Path | None = None
    """Optional path to save results (JSON format)."""

    visualize: bool = False
    """Generate visualization of predictions."""

    @model_validator(mode="after")
    def validate_inputs(self) -> "TestConfig":
        """Validate input configuration."""
        # Validate level_indices for localization
        if self.task == "localization" and self.level_indices is not None:
            if len(self.level_indices) != len(self.inputs):
                raise ValueError(
                    f"level_indices ({len(self.level_indices)}) must match "
                    f"number of inputs ({len(self.inputs)})"
                )
        return self


def _load_checkpoint(model: TModel, config: TestConfig) -> TModel:
    """Load checkpoint weights into model.

    Args:
        model: Model instance to load weights into.
        config: Test configuration with model path and device.

    Returns:
        Model with loaded weights.
    """
    checkpoint = torch.load(config.model_path, map_location=config.device, weights_only=True)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)

    model.to(config.device)
    model.eval()
    return model


def load_images(paths: list[Path]) -> list[np.ndarray]:
    """Load images from various formats.

    Supports:
    - Standard image formats (PNG, JPEG, etc.)
    - DICOM files (.dcm)
    - NIfTI files (.nii, .nii.gz)
    - MHA/MHD files

    Args:
        paths: List of input file paths.
        image_size: Target image size for resizing.

    Returns:
        List of numpy arrays in RGB format.
    """
    images: list[np.ndarray] = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        suffix = path.suffix.lower()

        # Check for medical imaging formats
        if suffix in {".dcm", ".mha", ".mhd", ".nrrd"} or suffix == ".nii" or path.name.endswith(".nii.gz"):
            logger.debug(f"Loading medical image: {path}")
            sitk_img = read_medical_image(path)
            arr = sitk.GetArrayFromImage(sitk_img)

            # Handle 3D volumes (take middle slice or first slice)
            if arr.ndim == 3:
                if arr.shape[0] == 1:
                    arr = arr[0]
                else:
                    # Take middle slice for 3D volumes
                    mid_idx = arr.shape[0] // 2
                    arr = arr[mid_idx]
                    logger.info(f"3D volume detected, using slice {mid_idx}")

            # Normalize to uint8
            arr = normalize_to_uint8(arr)

            # Convert to RGB
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)

            images.append(arr)

        else:
            # Standard image format
            logger.debug(f"Loading image: {path}")
            pil_img = Image.open(path).convert("RGB")
            images.append(np.array(pil_img))

    return images


def format_localization_results(
    result: dict,
    paths: list[Path],
    level_indices: list[int] | None,
) -> dict:
    """Format localization inference results.

    Args:
        result: Raw inference result from model.
        paths: Input file paths.
        level_indices: Optional level indices.

    Returns:
        Formatted result dictionary.
    """
    predictions = result["predictions"]
    coords_pixel = result["coords_pixel"]

    formatted = {
        "task": "localization",
        "num_images": len(paths),
        "inference_time_ms": result["inference_time_ms"],
        "device": result["device"],
        "results": [],
    }

    level_names = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

    for i, path in enumerate(paths):
        entry = {
            "file": str(path),
            "prediction_normalized": predictions[i].tolist(),
            "prediction_pixel": coords_pixel[i].tolist(),
        }
        if level_indices is not None:
            entry["level"] = level_names[level_indices[i]]
        formatted["results"].append(entry)

    return formatted


def format_classification_results(
    result: dict,
    paths: list[Path],
    class_names: list[str] | None,
) -> dict:
    """Format classification inference results.

    Args:
        result: Raw inference result from model.
        paths: Input file paths.
        class_names: Optional class names.

    Returns:
        Formatted result dictionary.
    """
    predictions = result["predictions"]
    probabilities = result.get("probabilities")

    formatted = {
        "task": "classification",
        "num_images": len(paths),
        "inference_time_ms": result["inference_time_ms"],
        "device": result["device"],
        "results": [],
    }

    for i, path in enumerate(paths):
        pred_class = int(predictions[i])
        entry = {
            "file": str(path),
            "predicted_class": pred_class,
        }

        if class_names is not None and pred_class < len(class_names):
            entry["class_name"] = class_names[pred_class]

        if probabilities is not None:
            entry["probabilities"] = probabilities[i].tolist()
            entry["confidence"] = float(probabilities[i][pred_class])

        formatted["results"].append(entry)

    return formatted


def visualize_localization(
    images: list[np.ndarray],
    coords_pixel: np.ndarray,
    paths: list[Path],
    output_path: Path,
) -> None:
    """Generate visualization for localization predictions.

    Args:
        images: Input images as numpy arrays.
        coords_pixel: Predicted pixel coordinates [N, 2].
        paths: Input file paths.
        output_path: Output directory for visualizations.
    """
    import matplotlib.pyplot as plt

    output_path.mkdir(parents=True, exist_ok=True)

    for i, (img, coords) in enumerate(zip(images, coords_pixel)):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        ax.scatter(coords[0], coords[1], c="red", s=100, marker="x", linewidths=2)
        ax.set_title(f"Prediction: ({coords[0]:.1f}, {coords[1]:.1f})")
        ax.axis("off")

        out_file = output_path / f"pred_{paths[i].stem}.png"
        fig.savefig(out_file, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"Saved visualization: {out_file}")


def main(config: TestConfig) -> dict:
    """Run model testing.

    Args:
        config: Test configuration.

    Returns:
        Formatted inference results.
    """
    from spine_vision.training.models import ConvNextClassifier, ConvNextLocalization

    setup_logger(verbose=config.verbose)

    logger.info(f"Testing {config.task} model: {config.model_path}")
    logger.info(f"Inputs: {len(config.inputs)} files")

    # Load images
    images = load_images(config.inputs)
    logger.info(f"Loaded {len(images)} images")

    # Cast images to the expected type
    image_inputs: list[np.ndarray] = images

    # Run task-specific inference
    if config.task == "localization":
        model = ConvNextLocalization(
            variant=config.model_variant,
            pretrained=False,
            num_levels=config.num_levels if config.use_level_embedding else 1,
        )
        model = _load_checkpoint(model, config)
        logger.info(f"Model loaded: {model.name}")

        result = model.test_inference(
            images=image_inputs,
            image_size=config.image_size,
            device=config.device,
            level_indices=config.level_indices,
        )
        formatted = format_localization_results(
            result, config.inputs, config.level_indices
        )

        # Visualization
        if config.visualize and config.output_path is not None:
            visualize_localization(
                images, result["coords_pixel"], config.inputs, config.output_path
            )

    else:
        model = ConvNextClassifier(
            variant=config.model_variant,
            num_classes=config.num_classes,
            pretrained=False,
        )
        model = _load_checkpoint(model, config)
        logger.info(f"Model loaded: {model.name}")

        result = model.test_inference(
            images=image_inputs,
            image_size=config.image_size,
            device=config.device,
            return_probabilities=True,
        )
        formatted = format_classification_results(
            result, config.inputs, config.class_names
        )

    # Print results
    logger.info(f"Inference completed in {result['inference_time_ms']:.2f}ms")

    for entry in formatted["results"]:
        if config.task == "localization":
            level_str = f" [{entry.get('level', '')}]" if "level" in entry else ""
            logger.info(
                f"  {Path(entry['file']).name}{level_str}: "
                f"({entry['prediction_pixel'][0]:.1f}, {entry['prediction_pixel'][1]:.1f})"
            )
        else:
            class_str = entry.get("class_name", f"Class {entry['predicted_class']}")
            conf_str = f" ({entry.get('confidence', 0):.1%})" if "confidence" in entry else ""
            logger.info(f"  {Path(entry['file']).name}: {class_str}{conf_str}")

    # Save results
    if config.output_path is not None:
        import json

        config.output_path.mkdir(parents=True, exist_ok=True)
        results_file = config.output_path / "results.json"
        with open(results_file, "w") as f:
            json.dump(formatted, f, indent=2)
        logger.info(f"Results saved to: {results_file}")

    return formatted

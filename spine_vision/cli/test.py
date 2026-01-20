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
    task: Literal["localization", "classification", "mtl_classification"] = "localization"
    """Task type determines model architecture and output format.

    - localization: CoordinateRegressor model (outputs all 5 levels at once)
    - classification: Classifier model (single-task)
    - mtl_classification: Classifier model (multi-task)
    """

    backbone: str = "convnext_base"
    """Backbone architecture (see BackboneFactory for options)."""

    # Localization-specific options
    num_levels: int = 5
    """Number of IVD levels the model outputs (default 5)."""

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


def _load_checkpoint(model: TModel, config: TestConfig) -> TModel:
    """Load checkpoint weights into model.

    Args:
        model: Model instance to load weights into.
        config: Test configuration with model path and device.

    Returns:
        Model with loaded weights.
    """
    checkpoint = torch.load(config.model_path, map_location=config.device, weights_only=False)

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
) -> dict:
    """Format localization inference results.

    Args:
        result: Raw inference result from model.
            - predictions: [N, num_levels, 2] normalized coords.
            - coords_pixel: [N, num_levels, 2] pixel coords.
        paths: Input file paths.

    Returns:
        Formatted result dictionary with all levels for each image.
    """
    predictions = result["predictions"]  # [N, num_levels, 2]
    coords_pixel = result["coords_pixel"]  # [N, num_levels, 2]

    level_names = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
    num_levels = predictions.shape[1]

    formatted: dict = {
        "task": "localization",
        "num_images": len(predictions),
        "num_levels": num_levels,
        "inference_time_ms": result["inference_time_ms"],
        "device": result["device"],
        "results": [],
    }

    for i, path in enumerate(paths):
        file_entry: dict = {
            "file": str(path),
            "levels": [],
        }
        for level_idx in range(num_levels):
            file_entry["levels"].append({
                "level": level_names[level_idx] if level_idx < len(level_names) else f"Level_{level_idx}",
                "prediction_normalized": predictions[i, level_idx].tolist(),
                "prediction_pixel": coords_pixel[i, level_idx].tolist(),
            })
        formatted["results"].append(file_entry)

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
        images: Input images as numpy arrays [N, H, W, 3].
        coords_pixel: Predicted pixel coordinates [N, num_levels, 2].
        paths: Input file paths.
        output_path: Output directory for visualizations.
    """
    import matplotlib.pyplot as plt

    output_path.mkdir(parents=True, exist_ok=True)
    level_names = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
    colors = ["red", "orange", "yellow", "green", "blue"]

    num_levels = coords_pixel.shape[1]

    for i, (img, path) in enumerate(zip(images, paths)):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)

        # Plot all levels for this image
        for level_idx in range(num_levels):
            coords = coords_pixel[i, level_idx]
            color = colors[level_idx] if level_idx < len(colors) else "purple"
            label = level_names[level_idx] if level_idx < len(level_names) else f"Level_{level_idx}"
            ax.scatter(
                coords[0], coords[1],
                c=color, s=100, marker="x", linewidths=2,
                label=label,
            )

        ax.legend(loc="upper right")
        ax.set_title(f"IVD Localization - {path.stem}")
        ax.axis("off")

        out_file = output_path / f"pred_{path.stem}.png"
        fig.savefig(out_file, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"Saved visualization: {out_file}")


def format_mtl_classification_results(
    result: dict,
    paths: list[Path],
) -> dict:
    """Format MTL classification inference results.

    Args:
        result: Raw inference result from ResNet50MTL.predict().
        paths: Input file paths.

    Returns:
        Formatted result dictionary with all 13 labels.
    """
    predictions = result["predictions"]
    probabilities = result.get("probabilities", {})

    formatted = {
        "task": "mtl_classification",
        "num_images": len(paths),
        "inference_time_ms": result["inference_time_ms"],
        "device": result["device"],
        "results": [],
    }

    for i, path in enumerate(paths):
        entry = {
            "file": str(path),
            "pfirrmann": int(predictions["pfirrmann"][i]),
            "modic": int(predictions["modic"][i]),
            "herniation": int(predictions["herniation"][i, 0]),
            "bulging": int(predictions["herniation"][i, 1]),
            "upper_endplate": int(predictions["endplate"][i, 0]),
            "lower_endplate": int(predictions["endplate"][i, 1]),
            "spondylolisthesis": int(predictions["spondy"][i]),
            "narrowing": int(predictions["narrowing"][i]),
        }

        if probabilities:
            entry["probabilities"] = {
                "pfirrmann": probabilities["pfirrmann"][i].tolist(),
                "modic": probabilities["modic"][i].tolist(),
                "herniation": probabilities["herniation"][i].tolist(),
                "endplate": probabilities["endplate"][i].tolist(),
                "spondy": float(probabilities["spondy"][i, 0]),
                "narrowing": float(probabilities["narrowing"][i, 0]),
            }

        formatted["results"].append(entry)

    return formatted


def main(config: TestConfig) -> dict:
    """Run model testing.

    Args:
        config: Test configuration.

    Returns:
        Formatted inference results.
    """
    from spine_vision.training.models import Classifier, CoordinateRegressor

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
        model = CoordinateRegressor(
            backbone=config.backbone,
            pretrained=False,
            num_levels=config.num_levels,
        )
        model = _load_checkpoint(model, config)
        logger.info(f"Model loaded: {model.name}")

        # Model outputs all levels at once - no need to replicate images
        result = model.test_inference(
            images=image_inputs,
            image_size=config.image_size,
            device=config.device,
        )
        formatted = format_localization_results(result, config.inputs)

        # Visualization
        if config.visualize and config.output_path is not None:
            processed_images = list(result["images"])
            visualize_localization(
                processed_images,
                result["coords_pixel"],
                config.inputs,
                config.output_path,
            )

    elif config.task == "mtl_classification":
        mtl_model = Classifier(
            backbone=config.backbone,
            pretrained=False,
        )
        mtl_model = _load_checkpoint(mtl_model, config)
        logger.info(f"Model loaded: {mtl_model.name}")

        result = mtl_model.test_inference(
            images=image_inputs,
            image_size=config.image_size,
            device=config.device,
        )
        formatted = format_mtl_classification_results(result, config.inputs)

    else:
        # Single-task classification using Classifier with one task
        from spine_vision.training.models import TaskConfig

        task_name = "class"
        model = Classifier(
            backbone=config.backbone,
            tasks=[TaskConfig(name=task_name, num_classes=config.num_classes)],
            pretrained=False,
        )
        model = _load_checkpoint(model, config)
        logger.info(f"Model loaded: {model.name}")

        result = model.test_inference(
            images=image_inputs,
            image_size=config.image_size,
            device=config.device,
        )

        # Extract single task results for format_classification_results
        # Classifier returns dict format, convert to flat arrays
        single_task_result = {
            "predictions": result["predictions"][task_name],
            "probabilities": result["probabilities"][task_name],
            "images": result["images"],
            "inference_time_ms": result["inference_time_ms"],
            "num_images": result["num_images"],
            "device": result["device"],
        }
        formatted = format_classification_results(
            single_task_result, config.inputs, config.class_names
        )

    # Print results
    logger.info(f"Inference completed in {result['inference_time_ms']:.2f}ms")

    for entry in formatted["results"]:
        if config.task == "localization":
            # All levels output for each file
            logger.info(f"  {Path(entry['file']).name}:")
            for level_entry in entry["levels"]:
                logger.info(
                    f"    {level_entry['level']}: "
                    f"({level_entry['prediction_pixel'][0]:.1f}, {level_entry['prediction_pixel'][1]:.1f})"
                )
        elif config.task == "mtl_classification":
            # Multi-task classification output
            logger.info(f"  {Path(entry['file']).name}:")
            logger.info(f"    Pfirrmann: Grade {entry['pfirrmann']}")
            logger.info(f"    Modic: Type {entry['modic']}")
            logger.info(f"    Herniation: {entry['herniation']}, Bulging: {entry['bulging']}")
            logger.info(f"    Endplate: Upper={entry['upper_endplate']}, Lower={entry['lower_endplate']}")
            logger.info(f"    Spondylolisthesis: {entry['spondylolisthesis']}")
            logger.info(f"    Narrowing: {entry['narrowing']}")
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

"""Visualize segmentation results with nnU-Net inference."""

import warnings
from pathlib import Path
from typing import Annotated, Literal

import SimpleITK as sitk
import tyro
from loguru import logger
from pydantic import BaseModel, computed_field
from tqdm.rich import tqdm
from tqdm.std import TqdmExperimentalWarning

from spine_vision.core.logging import setup_logger
from spine_vision.inference.segmentation import NNUNetSegmentation
from spine_vision.io.readers import read_medical_image
from spine_vision.io.writers import write_medical_image
from spine_vision.visualization.plotly_viewer import PlotlyViewer


class VisualizeConfig(BaseModel):
    """Configuration for visualization."""

    input_path: Path = Path.cwd() / "data/gold/classification/images/250029976/SAG T1"
    output_path: Path = Path.cwd() / "results/segmentation"
    model_path: Path = Path.cwd() / "weights/segmentation/Dataset501_Spider/nnUNetTrainer__nnUNetPlans__2d"
    dataset_id: int = 501
    configuration: str = "2d"
    fold: int = 0
    save_probabilities: bool = False
    enable_tta: bool = False
    verbose: Annotated[bool, tyro.conf.arg(aliases=["-v"])] = False
    output_mode: Literal["browser", "html", "image"] = "html"
    batch: bool = False

    @computed_field
    @property
    def temp_input_path(self) -> Path:
        return self.output_path / "temp_input"

    @computed_field
    @property
    def temp_output_path(self) -> Path:
        return self.output_path / "temp_prediction"


def visualize_single(
    config: VisualizeConfig,
    segmentation_model: NNUNetSegmentation,
    viewer: PlotlyViewer,
) -> None:
    """Visualize a single image with segmentation."""
    config.temp_input_path.mkdir(parents=True, exist_ok=True)
    config.temp_output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading image from {config.input_path}")
    image_sitk = read_medical_image(config.input_path)

    nifti_input_path = config.temp_input_path / "test_case_0000.nii.gz"
    write_medical_image(image_sitk, nifti_input_path)
    logger.info(f"Converted to NIfTI: {nifti_input_path}")

    result = segmentation_model.predict_from_path(
        config.temp_input_path,
        config.temp_output_path,
    )

    mask = result.prediction if isinstance(result.prediction, sitk.Image) else None
    viewer.visualize(
        image=image_sitk,
        mask=mask,
        title="Spine Segmentation (Interactive)",
        filename="visualization",
    )

    logger.info("Visualization complete")


def visualize_batch(
    config: VisualizeConfig,
    segmentation_model: NNUNetSegmentation,
    viewer: PlotlyViewer,
) -> None:
    """Visualize multiple images from a directory."""
    input_dirs = [d for d in config.input_path.iterdir() if d.is_dir()]

    if not input_dirs:
        input_dirs = [config.input_path]

    logger.info(f"Processing {len(input_dirs)} cases in batch mode")

    for input_dir in tqdm(input_dirs, desc="Processing", unit="case"):
        try:
            case_id = input_dir.name
            temp_input = config.output_path / "temp" / case_id / "input"
            temp_output = config.output_path / "temp" / case_id / "output"
            temp_input.mkdir(parents=True, exist_ok=True)
            temp_output.mkdir(parents=True, exist_ok=True)

            image_sitk = read_medical_image(input_dir)
            nifti_path = temp_input / f"{case_id}_0000.nii.gz"
            write_medical_image(image_sitk, nifti_path)

            result = segmentation_model.predict_from_path(temp_input, temp_output)

            mask = result.prediction if isinstance(result.prediction, sitk.Image) else None
            viewer.visualize(
                image=image_sitk,
                mask=mask,
                title=f"Segmentation - {case_id}",
                filename=f"visualization_{case_id}",
            )

        except Exception as e:
            logger.error(f"Failed to process {input_dir}: {e}")
            continue

    logger.info("Batch visualization complete")


def main(config: VisualizeConfig) -> None:
    """Run visualization pipeline."""
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

    setup_logger(verbose=config.verbose)

    config.output_path.mkdir(parents=True, exist_ok=True)

    segmentation_model = NNUNetSegmentation(
        model_path=config.model_path,
        dataset_id=config.dataset_id,
        configuration=config.configuration,
        fold=config.fold,
        save_probabilities=config.save_probabilities,
        enable_tta=config.enable_tta,
    )

    viewer = PlotlyViewer(
        output_path=config.output_path,
        output_mode=config.output_mode,
    )

    if config.batch:
        visualize_batch(config, segmentation_model, viewer)
    else:
        visualize_single(config, segmentation_model, viewer)

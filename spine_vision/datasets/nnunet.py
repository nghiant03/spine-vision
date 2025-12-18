"""Convert datasets to nnU-Net format."""

import json
import warnings
from pathlib import Path
from typing import Annotated

import SimpleITK as sitk
import tyro
from loguru import logger
from pydantic import BaseModel, computed_field
from tqdm.rich import tqdm
from tqdm.std import TqdmExperimentalWarning

from spine_vision.core.logging import setup_logger
from spine_vision.labels.mapping import (
    LabelSchema,
    generate_nnunet_labels,
    load_label_schema,
    remap_labels,
)


class ConvertConfig(BaseModel):
    """Configuration for dataset conversion."""

    input_path: Path = Path.cwd() / "data/raw/SPIDER"
    output_path: Path = Path.cwd() / "data/silver/SPIDER/Dataset501_Spider"
    schema_path: Path | None = None
    dataset_name: str = "Spider"
    channel_name: str = "MRI"
    file_extension: str = ".mha"
    verbose: Annotated[bool, tyro.conf.arg(aliases=["-v"])] = False

    @computed_field
    @property
    def input_images_path(self) -> Path:
        return self.input_path / "images"

    @computed_field
    @property
    def input_masks_path(self) -> Path:
        return self.input_path / "masks"

    @computed_field
    @property
    def output_images_path(self) -> Path:
        return self.output_path / "imagesTr"

    @computed_field
    @property
    def output_labels_path(self) -> Path:
        return self.output_path / "labelsTr"


def convert_image(
    image_path: Path,
    mask_path: Path,
    output_images: Path,
    output_labels: Path,
    case_id: str,
    schema: LabelSchema,
) -> bool:
    """Convert a single image-mask pair.

    Args:
        image_path: Path to source image.
        mask_path: Path to source mask.
        output_images: Output directory for images.
        output_labels: Output directory for labels.
        case_id: Case identifier.
        schema: Label schema for remapping.

    Returns:
        True if successful, False otherwise.
    """
    image = sitk.ReadImage(str(image_path))
    sitk.WriteImage(image, str(output_images / f"{case_id}_0000.nii.gz"))

    if not mask_path.exists():
        logger.warning(f"No mask found for {case_id}")
        return False

    mask = sitk.ReadImage(str(mask_path))
    mask_array = sitk.GetArrayFromImage(mask)

    remapped_array = remap_labels(mask_array, schema.mapping)

    remapped_mask = sitk.GetImageFromArray(remapped_array)
    remapped_mask.CopyInformation(image)

    sitk.WriteImage(remapped_mask, str(output_labels / f"{case_id}.nii.gz"))
    return True


def generate_dataset_json(
    output_path: Path,
    num_training: int,
    channel_name: str,
    schema: LabelSchema,
) -> None:
    """Generate nnU-Net dataset.json file.

    Args:
        output_path: Dataset output directory.
        num_training: Number of training samples.
        channel_name: Name of the image channel.
        schema: Label schema for label names.
    """
    label_names = generate_nnunet_labels(schema)

    dataset_json = {
        "channel_names": {"0": channel_name},
        "labels": label_names,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
    }

    json_path = output_path / "dataset.json"
    json_path.write_text(json.dumps(dataset_json, indent=4))
    logger.info(f"Created {json_path}")


def main(config: ConvertConfig) -> None:
    """Run dataset conversion."""
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

    setup_logger(verbose=config.verbose)

    config.output_images_path.mkdir(parents=True, exist_ok=True)
    config.output_labels_path.mkdir(parents=True, exist_ok=True)

    if config.schema_path:
        schema = load_label_schema(config.schema_path)
    else:
        schema = load_label_schema("spider")

    image_files = sorted(config.input_images_path.glob(f"*{config.file_extension}"))

    if not image_files:
        logger.error(
            f"No {config.file_extension} files found in {config.input_images_path}"
        )
        return

    logger.info(f"Found {len(image_files)} files. Starting conversion...")

    successful = 0
    for image_path in tqdm(image_files, desc="Converting", unit="file"):
        case_id = image_path.stem
        mask_path = config.input_masks_path / image_path.name

        if convert_image(
            image_path,
            mask_path,
            config.output_images_path,
            config.output_labels_path,
            case_id,
            schema,
        ):
            successful += 1

    generate_dataset_json(config.output_path, successful, config.channel_name, schema)
    logger.info(f"Conversion complete: {successful}/{len(image_files)} files processed")

import json
import warnings
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import tyro
from loguru import logger
from tqdm.rich import tqdm
from tqdm.std import TqdmExperimentalWarning

from config import ConvertConfig

LABEL_MAPPING: dict[int, int] = {
    0: 0,
    **{i: i for i in range(1, 26)},
    100: 26,
    **{201 + i: 27 + i for i in range(25)},
}

LABEL_NAMES: dict[str, int] = {
    "background": 0,
    **{f"Vertebra_{i}": i for i in range(1, 26)},
    "Spinal_Canal": 26,
    **{f"Disc_{201 + i}": 27 + i for i in range(25)},
}


def remap_labels(mask_array: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    remapped = np.zeros_like(mask_array)
    for old_label, new_label in mapping.items():
        remapped[mask_array == old_label] = new_label
    return remapped


def convert_image(
    image_path: Path,
    mask_path: Path,
    output_images: Path,
    output_labels: Path,
    case_id: str,
) -> bool:
    image = sitk.ReadImage(str(image_path))
    sitk.WriteImage(image, str(output_images / f"{case_id}_0000.nii.gz"))

    if not mask_path.exists():
        logger.warning(f"No mask found for {case_id}")
        return False

    mask = sitk.ReadImage(str(mask_path))
    mask_array = sitk.GetArrayFromImage(mask)

    remapped_array = remap_labels(mask_array, LABEL_MAPPING)

    remapped_mask = sitk.GetImageFromArray(remapped_array)
    remapped_mask.CopyInformation(image)

    sitk.WriteImage(remapped_mask, str(output_labels / f"{case_id}.nii.gz"))
    return True


def generate_dataset_json(
    output_path: Path,
    num_training: int,
    channel_name: str,
    label_names: dict[str, int],
) -> None:
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
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

    log_level = "DEBUG" if config.verbose else "INFO"
    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
    )

    config.output_images_path.mkdir(parents=True, exist_ok=True)
    config.output_labels_path.mkdir(parents=True, exist_ok=True)

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
        ):
            successful += 1

    generate_dataset_json(
        config.output_path, successful, config.channel_name, LABEL_NAMES
    )
    logger.info(f"Conversion complete: {successful}/{len(image_files)} files processed")


if __name__ == "__main__":
    convert_config = tyro.cli(ConvertConfig)
    main(convert_config)

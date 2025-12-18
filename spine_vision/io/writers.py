"""Medical image writers supporting multiple formats."""

from pathlib import Path

import SimpleITK as sitk
from loguru import logger


def write_medical_image(
    image: sitk.Image,
    output_path: Path,
    use_compression: bool = True,
) -> None:
    """Write medical image to file.

    Format is determined by file extension. Supported formats:
    - NIfTI: .nii, .nii.gz
    - MHA: .mha
    - MHD: .mhd (creates companion .raw file)
    - NRRD: .nrrd

    Args:
        image: SimpleITK Image to write.
        output_path: Output file path with appropriate extension.
        use_compression: Whether to use compression (default True).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Writing image to: {output_path}")
    sitk.WriteImage(image, str(output_path), useCompression=use_compression)


def convert_format(
    input_path: Path,
    output_path: Path,
    use_compression: bool = True,
) -> None:
    """Convert medical image from one format to another.

    Args:
        input_path: Input image path.
        output_path: Output image path (format determined by extension).
        use_compression: Whether to use compression.
    """
    from spine_vision.io.readers import read_medical_image

    logger.info(f"Converting {input_path} -> {output_path}")
    image = read_medical_image(input_path)
    write_medical_image(image, output_path, use_compression)

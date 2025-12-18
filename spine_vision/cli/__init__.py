"""Unified CLI for spine-vision.

Usage:
    spine-vision dataset nnunet [OPTIONS]      # Convert to nnUNet format
    spine-vision dataset ivd-coords [OPTIONS]  # Create IVD coordinates dataset
    spine-vision dataset phenikaa [OPTIONS]    # Preprocess Phenikaa dataset
    spine-vision visualize [OPTIONS]           # Visualize segmentation results
"""

import dataclasses
from typing import Annotated, Union

import tyro

from spine_vision.datasets.nnunet import ConvertConfig, main as nnunet_main
from spine_vision.datasets.ivd_coords import IVDDatasetConfig, main as ivd_coords_main
from spine_vision.datasets.phenikaa import PreprocessConfig, main as phenikaa_main
from spine_vision.cli.visualize import VisualizeConfig, main as visualize_main


DatasetSubcommand = Annotated[
    Union[
        Annotated[
            ConvertConfig,
            tyro.conf.subcommand("nnunet", prefix_name=False, description="Convert datasets to nnU-Net format"),
        ],
        Annotated[
            IVDDatasetConfig,
            tyro.conf.subcommand("ivd-coords", prefix_name=False, description="Create IVD coordinates dataset"),
        ],
        Annotated[
            PreprocessConfig,
            tyro.conf.subcommand("phenikaa", prefix_name=False, description="Preprocess Phenikaa dataset (OCR + matching)"),
        ],
    ],
    tyro.conf.arg(name=""),
]


@dataclasses.dataclass
class DatasetCommand:
    """Dataset creation and conversion commands."""

    cmd: DatasetSubcommand


Command = Union[
    Annotated[
        DatasetCommand,
        tyro.conf.subcommand("dataset", description="Dataset creation and conversion"),
    ],
    Annotated[
        VisualizeConfig,
        tyro.conf.subcommand("visualize", description="Visualize segmentation results"),
    ],
]


def cli() -> None:
    """CLI entry point for spine-vision."""
    config: Command = tyro.cli(Command)  # type: ignore[assignment]

    match config:
        case DatasetCommand(cmd=cmd):
            match cmd:
                case ConvertConfig():
                    nnunet_main(cmd)
                case IVDDatasetConfig():
                    ivd_coords_main(cmd)
                case PreprocessConfig():
                    phenikaa_main(cmd)
        case VisualizeConfig():
            visualize_main(config)


if __name__ == "__main__":
    cli()

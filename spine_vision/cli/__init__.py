"""Unified CLI for spine-vision.

Usage:
    spine-vision dataset nnunet [OPTIONS]          # Convert to nnUNet format
    spine-vision dataset localization [OPTIONS]    # Create localization dataset
    spine-vision dataset phenikaa [OPTIONS]        # Preprocess Phenikaa dataset
    spine-vision dataset classification [OPTIONS]  # Create classification dataset
    spine-vision train localization [OPTIONS]      # Train localization model
    spine-vision train classification [OPTIONS]    # Train classification model (MTL)
    spine-vision test [OPTIONS]                    # Test trained models
    spine-vision evaluate [OPTIONS]                # Evaluate on test set with visualization
    spine-vision analyze [OPTIONS]                 # Analyze classification dataset
    spine-vision visualize [OPTIONS]               # Visualize segmentation results
"""

import dataclasses
from typing import Annotated, Union

import tyro
from loguru import logger

from spine_vision.cli.analyze import AnalyzeConfig
from spine_vision.cli.analyze import main as analyze_main
from spine_vision.cli.evaluate import EvaluateConfig
from spine_vision.cli.evaluate import main as evaluate_main
from spine_vision.cli.test import TestConfig
from spine_vision.cli.test import main as test_main
from spine_vision.cli.train import main as train_main
from spine_vision.datasets import (
    ClassificationDatasetConfig,
    ClassificationDatasetProcessor,
    IVDCoordsDatasetProcessor,
    IVDDatasetConfig,
    PhenikkaaProcessor,
    PreprocessConfig,
)
from spine_vision.training.trainers.classification import ClassificationConfig
from spine_vision.training.trainers.localization import LocalizationConfig

DatasetSubcommand = Annotated[
    Union[
        Annotated[
            IVDDatasetConfig,
            tyro.conf.subcommand(
                "localization",
                prefix_name=False,
                description="Create localization dataset (IVD coordinates)",
            ),
        ],
        Annotated[
            PreprocessConfig,
            tyro.conf.subcommand(
                "phenikaa",
                prefix_name=False,
                description="Preprocess Phenikaa dataset (OCR + matching)",
            ),
        ],
        Annotated[
            ClassificationDatasetConfig,
            tyro.conf.subcommand(
                "classification",
                prefix_name=False,
                description="Create classification dataset (Phenikaa + SPIDER)",
            ),
        ],
    ],
    tyro.conf.arg(name=""),
]


TrainSubcommand = Annotated[
    Union[
        Annotated[
            LocalizationConfig,
            tyro.conf.subcommand(
                "localization",
                prefix_name=False,
                description="Train localization model (ConvNext)",
            ),
        ],
        Annotated[
            ClassificationConfig,
            tyro.conf.subcommand(
                "classification",
                prefix_name=False,
                description="Train classification model (ResNet50-MTL)",
            ),
        ],
    ],
    tyro.conf.arg(name=""),
]


@dataclasses.dataclass
class DatasetCommand:
    """Dataset creation and conversion commands."""

    cmd: DatasetSubcommand


@dataclasses.dataclass
class TrainCommand:
    """Model training commands."""

    cmd: TrainSubcommand


Command = Union[
    Annotated[
        DatasetCommand,
        tyro.conf.subcommand("dataset", description="Dataset creation and conversion"),
    ],
    Annotated[
        TrainCommand,
        tyro.conf.subcommand("train", description="Model training"),
    ],
    Annotated[
        TestConfig,
        tyro.conf.subcommand("test", description="Test trained models with images or DICOM"),
    ],
    Annotated[
        EvaluateConfig,
        tyro.conf.subcommand("evaluate", description="Evaluate on test set with visualization"),
    ],
    Annotated[
        AnalyzeConfig,
        tyro.conf.subcommand("analyze", description="Analyze classification dataset"),
    ],
]


def cli() -> None:
    """CLI entry point for spine-vision."""
    config: Command = tyro.cli(Command)  # type: ignore[assignment]

    match config:
        case DatasetCommand(cmd=cmd):
            match cmd:
                case IVDDatasetConfig():
                    processor = IVDCoordsDatasetProcessor(cmd)
                    result = processor.process()
                    logger.info(result.summary)
                case PreprocessConfig():
                    processor = PhenikkaaProcessor(cmd)
                    result = processor.process()
                    logger.info(result.summary)
                case ClassificationDatasetConfig():
                    processor = ClassificationDatasetProcessor(cmd)
                    result = processor.process()
                    logger.info(result.summary)
        case TrainCommand(cmd=cmd):
            match cmd:
                case LocalizationConfig():
                    train_main(cmd)
                case ClassificationConfig():
                    train_main(cmd)
        case TestConfig():
            test_main(config)
        case EvaluateConfig():
            evaluate_main(config)
        case AnalyzeConfig():
            analyze_main(config)


if __name__ == "__main__":
    cli()

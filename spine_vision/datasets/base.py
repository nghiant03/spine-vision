"""Base classes for dataset creation pipelines.

Provides abstract base classes for all dataset processors,
mirroring the training module architecture for consistency.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel

TConfig = TypeVar("TConfig", bound=BaseModel)


@dataclass
class ProcessingResult:
    """Result of dataset processing.

    Contains statistics and metadata about the processed dataset.
    """

    num_samples: int
    """Total number of samples processed."""

    output_path: Path
    """Path to output dataset."""

    summary: str = ""
    """Human-readable summary of processing."""


class BaseProcessor(ABC, Generic[TConfig]):
    """Abstract base class for dataset processors.

    Mirrors the BaseTrainer architecture for consistency across the codebase.
    Provides hooks for custom behavior without requiring method overrides.

    Usage:
        class MyProcessor(BaseProcessor[MyConfig]):
            def process(self) -> ProcessingResult:
                # Processing logic here
                return ProcessingResult(...)

        config = MyConfig(...)
        processor = MyProcessor(config)
        result = processor.process()
    """

    def __init__(self, config: TConfig) -> None:
        """Initialize processor with configuration.

        Args:
            config: Dataset-specific configuration.
        """
        self.config = config

    @abstractmethod
    def process(self) -> ProcessingResult:
        """Execute dataset processing pipeline.

        This method must be implemented by subclasses to define the
        dataset creation or conversion logic.

        Returns:
            ProcessingResult with statistics and output path.
        """
        ...

    def on_process_begin(self) -> None:
        """Hook called before processing begins.

        Override this method to add custom setup logic, such as
        validating inputs or initializing resources.
        """
        pass

    def on_process_end(self, result: ProcessingResult) -> None:
        """Hook called after processing completes.

        Override this method to add custom cleanup or reporting logic.

        Args:
            result: Processing result with statistics.
        """
        pass


__all__ = [
    "BaseProcessor",
    "ProcessingResult",
]

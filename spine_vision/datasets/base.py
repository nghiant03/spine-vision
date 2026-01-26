"""Base classes for dataset creation pipelines.

Provides shared result types for dataset processing functions.
"""

from dataclasses import dataclass
from pathlib import Path


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


__all__ = [
    "ProcessingResult",
]

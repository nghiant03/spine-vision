"""Core utilities for logging, configuration, and pipeline orchestration."""

from spine_vision.core.logging import setup_logger
from spine_vision.core.config import BaseConfig

__all__ = ["setup_logger", "BaseConfig"]

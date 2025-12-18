"""Core utilities for logging, configuration, and pipeline orchestration."""

from spine_vision.core.config import BaseConfig
from spine_vision.core.logging import setup_logger

__all__ = ["setup_logger", "BaseConfig"]

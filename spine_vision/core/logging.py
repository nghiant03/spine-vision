"""Centralized logging configuration."""

import logging
from pathlib import Path

from loguru import logger
from tqdm.rich import tqdm


def setup_logger(
    verbose: bool = False,
) -> None:
    """Configure loguru logger with tqdm integration.

    Args:
        verbose: If True, set log level to DEBUG; otherwise INFO.
        enable_file_log: If True, also write logs to a file.
        log_path: Directory for log files (required if enable_file_log is True).
        log_filename: Name of the log file.
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
    )


def add_file_log(
    log_path: Path | str | None = None, log_filename: str = "spine_vision.log"
):
    if log_path is None:
        log_path = Path.cwd() / "logs"

    if isinstance(log_path, str):
        log_path = Path(log_path)

    log_path.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path / log_filename,
        level="DEBUG",
        rotation="10 MB",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
        encoding="utf-8",
    )
    logger.info(f"Logging to {log_path}")

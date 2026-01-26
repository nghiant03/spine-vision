"""Core utilities for logging, configuration, and task definitions."""

from spine_vision.core.config import BaseConfig
from spine_vision.core.logging import add_file_log, setup_logger
from spine_vision.core.tasks import (
    AVAILABLE_TASK_NAMES,
    TASK_REGISTRY,
    TaskConfig,
    TaskStrategy,
    TaskType,
    compute_predictions_for_tasks,
    compute_probabilities_for_tasks,
    create_loss_functions,
    get_strategy,
    get_task,
    get_task_color,
    get_task_colors,
    get_task_display_name,
    get_task_display_names,
    get_tasks,
    register_task,
)

__all__ = [
    # Config and logging
    "BaseConfig",
    "add_file_log",
    "setup_logger",
    # Task system
    "AVAILABLE_TASK_NAMES",
    "TASK_REGISTRY",
    "TaskConfig",
    "TaskStrategy",
    "TaskType",
    "compute_predictions_for_tasks",
    "compute_probabilities_for_tasks",
    "create_loss_functions",
    "get_strategy",
    "get_task",
    "get_task_color",
    "get_task_colors",
    "get_task_display_name",
    "get_task_display_names",
    "get_tasks",
    "register_task",
]

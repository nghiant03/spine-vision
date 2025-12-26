"""Model and trainer registry for extensible training infrastructure.

Provides decorator-based registration of models and trainers, enabling:
- Dynamic model/trainer discovery without manual imports
- Factory methods for creating instances from config
- CLI integration without modifying Union types

Usage:
    # Register a model
    @register_model("convnext_loc")
    class ConvNextLocalization(BaseModel):
        ...

    # Register a trainer
    @register_trainer("localization")
    class LocalizationTrainer(BaseTrainer):
        ...

    # Create instances
    model = ModelRegistry.create("convnext_loc", variant="base")
    trainer = TrainerRegistry.create("localization", config)

    # List available options
    ModelRegistry.list_models()
    TrainerRegistry.list_trainers()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar

from loguru import logger

if TYPE_CHECKING:
    from spine_vision.training.base import BaseModel, BaseTrainer, TrainingConfig

T = TypeVar("T")
ModelT = TypeVar("ModelT", bound="BaseModel")
TrainerT = TypeVar("TrainerT", bound="BaseTrainer[Any, Any, Any]")
ConfigT = TypeVar("ConfigT", bound="TrainingConfig")


class ModelRegistry:
    """Registry for model classes.

    Enables dynamic model discovery and instantiation without hardcoded imports.
    """

    _models: dict[str, type[BaseModel]] = {}
    _metadata: dict[str, dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        *,
        task: str | None = None,
        description: str = "",
        aliases: list[str] | None = None,
    ) -> Callable[[type[ModelT]], type[ModelT]]:
        """Decorator to register a model class.

        Args:
            name: Unique identifier for the model.
            task: Task type (e.g., "localization", "classification").
            description: Human-readable description.
            aliases: Alternative names for the model.

        Returns:
            Decorator function.

        Example:
            @ModelRegistry.register("convnext_loc", task="localization")
            class ConvNextLocalization(BaseModel):
                ...
        """

        def decorator(model_cls: type[ModelT]) -> type[ModelT]:
            if name in cls._models:
                logger.warning(f"Model '{name}' already registered, overwriting")

            cls._models[name] = model_cls  # type: ignore[assignment]
            cls._metadata[name] = {
                "task": task,
                "description": description,
                "class": model_cls.__name__,
                "module": model_cls.__module__,
            }

            # Register aliases
            if aliases:
                for alias in aliases:
                    cls._models[alias] = model_cls  # type: ignore[assignment]
                    cls._metadata[alias] = cls._metadata[name]

            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseModel:
        """Create a model instance by name.

        Args:
            name: Registered model name.
            **kwargs: Arguments to pass to model constructor.

        Returns:
            Model instance.

        Raises:
            KeyError: If model not found.
        """
        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise KeyError(f"Model '{name}' not found. Available: {available}")

        model_cls = cls._models[name]
        return model_cls(**kwargs)

    @classmethod
    def get(cls, name: str) -> type[BaseModel]:
        """Get model class by name.

        Args:
            name: Registered model name.

        Returns:
            Model class.

        Raises:
            KeyError: If model not found.
        """
        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        return cls._models[name]

    @classmethod
    def list_models(cls, task: str | None = None) -> list[str]:
        """List registered model names.

        Args:
            task: Optional filter by task type.

        Returns:
            List of model names.
        """
        if task is None:
            return list(cls._models.keys())
        return [
            name
            for name, meta in cls._metadata.items()
            if meta.get("task") == task
        ]

    @classmethod
    def get_metadata(cls, name: str) -> dict[str, Any]:
        """Get model metadata.

        Args:
            name: Model name.

        Returns:
            Metadata dictionary.
        """
        return cls._metadata.get(name, {})

    @classmethod
    def clear(cls) -> None:
        """Clear all registered models (for testing)."""
        cls._models.clear()
        cls._metadata.clear()


class TrainerRegistry:
    """Registry for trainer classes.

    Enables dynamic trainer discovery and instantiation.
    """

    _trainers: dict[str, type[BaseTrainer[Any, Any, Any]]] = {}
    _configs: dict[str, type[TrainingConfig]] = {}
    _metadata: dict[str, dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        *,
        config_cls: type[TrainingConfig] | None = None,
        description: str = "",
    ) -> Callable[[type[TrainerT]], type[TrainerT]]:
        """Decorator to register a trainer class.

        Args:
            name: Unique identifier for the trainer (usually matches task name).
            config_cls: Configuration class for this trainer.
            description: Human-readable description.

        Returns:
            Decorator function.

        Example:
            @TrainerRegistry.register("localization", config_cls=LocalizationConfig)
            class LocalizationTrainer(BaseTrainer):
                ...
        """

        def decorator(trainer_cls: type[TrainerT]) -> type[TrainerT]:
            if name in cls._trainers:
                logger.warning(f"Trainer '{name}' already registered, overwriting")

            cls._trainers[name] = trainer_cls  # type: ignore[assignment]
            cls._metadata[name] = {
                "description": description,
                "class": trainer_cls.__name__,
                "module": trainer_cls.__module__,
            }

            if config_cls is not None:
                cls._configs[name] = config_cls

            return trainer_cls

        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        config: TrainingConfig,
        **kwargs: Any,
    ) -> BaseTrainer[Any, Any, Any]:
        """Create a trainer instance by name.

        Args:
            name: Registered trainer name.
            config: Training configuration.
            **kwargs: Additional arguments to pass to trainer constructor.

        Returns:
            Trainer instance.

        Raises:
            KeyError: If trainer not found.
        """
        if name not in cls._trainers:
            available = ", ".join(cls._trainers.keys())
            raise KeyError(f"Trainer '{name}' not found. Available: {available}")

        trainer_cls = cls._trainers[name]
        return trainer_cls(config, **kwargs)

    @classmethod
    def create_from_config(
        cls,
        config: TrainingConfig,
        **kwargs: Any,
    ) -> BaseTrainer[Any, Any, Any]:
        """Create a trainer instance from config's task field.

        Uses config.task to determine which trainer to instantiate.

        Args:
            config: Training configuration with task field.
            **kwargs: Additional arguments to pass to trainer constructor.

        Returns:
            Trainer instance.

        Raises:
            KeyError: If no trainer registered for config's task.
        """
        task = config.task
        if task not in cls._trainers:
            available = ", ".join(cls._trainers.keys())
            raise KeyError(
                f"No trainer registered for task '{task}'. Available: {available}"
            )

        trainer_cls = cls._trainers[task]
        return trainer_cls(config, **kwargs)

    @classmethod
    def get(cls, name: str) -> type[BaseTrainer[Any, Any, Any]]:
        """Get trainer class by name.

        Args:
            name: Registered trainer name.

        Returns:
            Trainer class.

        Raises:
            KeyError: If trainer not found.
        """
        if name not in cls._trainers:
            available = ", ".join(cls._trainers.keys())
            raise KeyError(f"Trainer '{name}' not found. Available: {available}")
        return cls._trainers[name]

    @classmethod
    def get_config_class(cls, name: str) -> type[TrainingConfig] | None:
        """Get configuration class for a trainer.

        Args:
            name: Trainer name.

        Returns:
            Configuration class or None if not registered.
        """
        return cls._configs.get(name)

    @classmethod
    def list_trainers(cls) -> list[str]:
        """List registered trainer names.

        Returns:
            List of trainer names.
        """
        return list(cls._trainers.keys())

    @classmethod
    def get_metadata(cls, name: str) -> dict[str, Any]:
        """Get trainer metadata.

        Args:
            name: Trainer name.

        Returns:
            Metadata dictionary.
        """
        return cls._metadata.get(name, {})

    @classmethod
    def clear(cls) -> None:
        """Clear all registered trainers (for testing)."""
        cls._trainers.clear()
        cls._configs.clear()
        cls._metadata.clear()


class MetricsRegistry:
    """Registry for metrics classes.

    Enables dynamic metrics discovery and instantiation.
    """

    _metrics: dict[str, type[Any]] = {}
    _metadata: dict[str, dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        *,
        task: str | None = None,
        description: str = "",
    ) -> Callable[[type[T]], type[T]]:
        """Decorator to register a metrics class.

        Args:
            name: Unique identifier for the metrics.
            task: Task type this metrics is for.
            description: Human-readable description.

        Returns:
            Decorator function.
        """

        def decorator(metrics_cls: type[T]) -> type[T]:
            if name in cls._metrics:
                logger.warning(f"Metrics '{name}' already registered, overwriting")

            cls._metrics[name] = metrics_cls
            cls._metadata[name] = {
                "task": task,
                "description": description,
                "class": metrics_cls.__name__,
                "module": metrics_cls.__module__,
            }
            return metrics_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> Any:
        """Create a metrics instance by name."""
        if name not in cls._metrics:
            available = ", ".join(cls._metrics.keys())
            raise KeyError(f"Metrics '{name}' not found. Available: {available}")
        return cls._metrics[name](**kwargs)

    @classmethod
    def get(cls, name: str) -> type[Any]:
        """Get metrics class by name."""
        if name not in cls._metrics:
            available = ", ".join(cls._metrics.keys())
            raise KeyError(f"Metrics '{name}' not found. Available: {available}")
        return cls._metrics[name]

    @classmethod
    def list_metrics(cls, task: str | None = None) -> list[str]:
        """List registered metrics names."""
        if task is None:
            return list(cls._metrics.keys())
        return [
            name
            for name, meta in cls._metadata.items()
            if meta.get("task") == task
        ]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered metrics (for testing)."""
        cls._metrics.clear()
        cls._metadata.clear()


# Convenience decorators
def register_model(
    name: str,
    *,
    task: str | None = None,
    description: str = "",
    aliases: list[str] | None = None,
) -> Callable[[type[ModelT]], type[ModelT]]:
    """Convenience decorator for registering models.

    Example:
        @register_model("convnext_loc", task="localization")
        class ConvNextLocalization(BaseModel):
            ...
    """
    return ModelRegistry.register(
        name, task=task, description=description, aliases=aliases
    )


def register_trainer(
    name: str,
    *,
    config_cls: type[TrainingConfig] | None = None,
    description: str = "",
) -> Callable[[type[TrainerT]], type[TrainerT]]:
    """Convenience decorator for registering trainers.

    Example:
        @register_trainer("localization", config_cls=LocalizationConfig)
        class LocalizationTrainer(BaseTrainer):
            ...
    """
    return TrainerRegistry.register(name, config_cls=config_cls, description=description)


def register_metrics(
    name: str,
    *,
    task: str | None = None,
    description: str = "",
) -> Callable[[type[T]], type[T]]:
    """Convenience decorator for registering metrics.

    Example:
        @register_metrics("localization", task="localization")
        class LocalizationMetrics(BaseMetrics):
            ...
    """
    return MetricsRegistry.register(name, task=task, description=description)

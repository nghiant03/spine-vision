"""Model and trainer registry for extensible training infrastructure.

Simplified registry system with essential functionality only.

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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from spine_vision.training.base import BaseModel, BaseTrainer, TrainingConfig

T = TypeVar("T")
ModelT = TypeVar("ModelT", bound="BaseModel")
TrainerT = TypeVar("TrainerT", bound="BaseTrainer[Any, Any, Any]")


class ModelRegistry:
    """Simple registry for model classes."""

    _models: dict[str, type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[ModelT]], type[ModelT]]:
        """Register a model class.

        Args:
            name: Unique identifier for the model.

        Returns:
            Decorator function.
        """

        def decorator(model_cls: type[ModelT]) -> type[ModelT]:
            cls._models[name] = model_cls  # type: ignore[assignment]
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

        return cls._models[name](**kwargs)

    @classmethod
    def get(cls, name: str) -> type[BaseModel]:
        """Get model class by name."""
        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        return cls._models[name]

    @classmethod
    def list_models(cls) -> list[str]:
        """List registered model names."""
        return list(cls._models.keys())


class TrainerRegistry:
    """Simple registry for trainer classes."""

    _trainers: dict[str, type[BaseTrainer[Any, Any, Any]]] = {}
    _configs: dict[str, type[TrainingConfig]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        *,
        config_cls: type[TrainingConfig] | None = None,
    ) -> Callable[[type[TrainerT]], type[TrainerT]]:
        """Register a trainer class.

        Args:
            name: Unique identifier for the trainer.
            config_cls: Optional configuration class for this trainer.

        Returns:
            Decorator function.
        """

        def decorator(trainer_cls: type[TrainerT]) -> type[TrainerT]:
            cls._trainers[name] = trainer_cls  # type: ignore[assignment]
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
        """Create a trainer instance by name."""
        if name not in cls._trainers:
            available = ", ".join(cls._trainers.keys())
            raise KeyError(f"Trainer '{name}' not found. Available: {available}")

        return cls._trainers[name](config, **kwargs)

    @classmethod
    def create_from_config(
        cls,
        config: TrainingConfig,
        **kwargs: Any,
    ) -> BaseTrainer[Any, Any, Any]:
        """Create a trainer from config's task field."""
        task = config.task
        if task not in cls._trainers:
            available = ", ".join(cls._trainers.keys())
            raise KeyError(
                f"No trainer registered for task '{task}'. Available: {available}"
            )

        return cls._trainers[task](config, **kwargs)

    @classmethod
    def get(cls, name: str) -> type[BaseTrainer[Any, Any, Any]]:
        """Get trainer class by name."""
        if name not in cls._trainers:
            available = ", ".join(cls._trainers.keys())
            raise KeyError(f"Trainer '{name}' not found. Available: {available}")
        return cls._trainers[name]

    @classmethod
    def get_config_class(cls, name: str) -> type[TrainingConfig] | None:
        """Get configuration class for a trainer."""
        return cls._configs.get(name)

    @classmethod
    def list_trainers(cls) -> list[str]:
        """List registered trainer names."""
        return list(cls._trainers.keys())


class MetricsRegistry:
    """Simple registry for metrics classes."""

    _metrics: dict[str, type[Any]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[T]], type[T]]:
        """Register a metrics class."""

        def decorator(metrics_cls: type[T]) -> type[T]:
            cls._metrics[name] = metrics_cls
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
    def list_metrics(cls) -> list[str]:
        """List registered metrics names."""
        return list(cls._metrics.keys())


# Convenience decorators
def register_model(name: str) -> Callable[[type[ModelT]], type[ModelT]]:
    """Decorator for registering models.

    Example:
        @register_model("convnext_loc")
        class ConvNextLocalization(BaseModel):
            ...
    """
    return ModelRegistry.register(name)


def register_trainer(
    name: str,
    *,
    config_cls: type[TrainingConfig] | None = None,
) -> Callable[[type[TrainerT]], type[TrainerT]]:
    """Decorator for registering trainers.

    Example:
        @register_trainer("localization", config_cls=LocalizationConfig)
        class LocalizationTrainer(BaseTrainer):
            ...
    """
    return TrainerRegistry.register(name, config_cls=config_cls)


def register_metrics(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator for registering metrics.

    Example:
        @register_metrics("localization")
        class LocalizationMetrics(BaseMetrics):
            ...
    """
    return MetricsRegistry.register(name)

"""Base classes for training infrastructure.

Provides abstract interfaces for models and trainers that can be extended
for various tasks (localization, classification, segmentation).

Uses HuggingFace Accelerate for distributed training and mixed precision,
with optional Trackio logging.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Generic, Literal, Sequence, TypeVar

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from loguru import logger
from PIL import Image
from pydantic import model_validator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from spine_vision.core import BaseConfig


def _create_worker_init_fn(seed: int) -> Callable[[int], None]:
    """Create a worker init function that seeds each worker deterministically.

    Each DataLoader worker gets a unique seed based on base seed + worker_id.
    This ensures reproducible data loading and augmentation across runs.

    Args:
        seed: Base random seed.

    Returns:
        Worker init function for DataLoader.
    """
    import random

    def worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn


# Type variables for generic trainer
TConfig = TypeVar("TConfig", bound="TrainingConfig")
TModel = TypeVar("TModel", bound=nn.Module)
TDataset = TypeVar("TDataset", bound=Dataset[Any])


def generate_run_id() -> str:
    """Generate a unique run ID with timestamp and short UUID.

    Format: YYYYMMDD_HHMMSS_<short_uuid>
    Example: 20231215_143022_a1b2c3
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"{timestamp}_{short_uuid}"


class TrainingConfig(BaseConfig):
    """Base configuration for training.

    Subclass this for task-specific configurations.

    Output structure:
        weights/<task>/<run_id>/
            best_model.pt
            checkpoint_epoch_N.pt
            config.yaml
            logs/
                training_curves.html
                predictions_epoch_N.html
                error_distribution.html
    """

    # Run identification
    run_id: str = ""
    """Unique run identifier. Auto-generated if not provided."""

    task: str = "training"
    """Task name for organizing outputs (e.g., 'localization', 'classification')."""

    # Data paths
    data_path: Path = Path("data/processed/ivd_coords")
    output_path: Path | None = None
    """Output directory. Defaults to weights/<task>/<run_id> if not specified."""

    checkpoint_path: Path | None = None

    # Training parameters
    batch_size: int = 32
    num_epochs: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float | None = 1.0

    # Scheduler
    scheduler_type: Literal["cosine", "step", "plateau", "none"] = "cosine"
    scheduler_patience: int = 10
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    warmup_epochs: int = 5

    # Early stopping
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4

    # Validation
    val_split: float = 0.2
    val_frequency: int = 1

    # Hardware
    device: str = "cuda:0"
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True

    # Logging
    log_frequency: int = 10
    save_frequency: int = 10

    # Trackio logging
    use_trackio: bool = False
    use_space: bool = True
    trackio_project: str = "spine-vision"
    trackio_run_name: str | None = None
    

    # Reproducibility
    seed: int = 42

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def setup_paths(self) -> "TrainingConfig":
        """Set up run_id and output_path if not provided."""
        # Generate run_id if not provided
        if not self.run_id:
            object.__setattr__(self, "run_id", generate_run_id())

        # Set output_path if not provided
        if self.output_path is None:
            object.__setattr__(
                self, "output_path", Path("weights") / self.task / self.run_id
            )

        # Sync trackio_run_name with run_id if not provided
        if self.use_trackio and self.trackio_run_name is None:
            object.__setattr__(self, "trackio_run_name", self.run_id)

        return self

    @property
    def logs_path(self) -> Path:
        """Path for logs and visualizations."""
        assert self.output_path is not None
        return self.output_path / "logs"

    @property
    def config_path(self) -> Path:
        """Path for saving configuration file."""
        assert self.output_path is not None
        return self.output_path / "config.yaml"

    def save_config(self) -> None:
        """Save configuration to YAML file."""
        import yaml

        assert self.output_path is not None
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Convert to dict with Path objects as strings
        config_dict = self.model_dump()
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)

        with open(self.config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved config to: {self.config_path}")


@dataclass
class TrainingResult:
    """Container for training results."""

    best_epoch: int
    best_metric: float
    final_train_loss: float
    final_val_loss: float
    history: dict[str, list[float]] = field(default_factory=dict)
    checkpoint_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpochResult:
    """Container for single epoch results."""

    epoch: int
    train_loss: float
    val_loss: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    lr: float = 0.0


class BaseModel(nn.Module, ABC):
    """Abstract base class for trainable models.

    All models should inherit from this class and implement:
    - forward(): The forward pass
    - get_loss(): Compute training loss
    - predict(): Run inference (no gradients)
    - test_inference(): Test model with sample images
    """

    def __init__(self) -> None:
        super().__init__()
        self._is_initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this model."""
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs: Any) -> Any:
        """Forward pass.

        Args:
            x: Input tensor.
            **kwargs: Additional model-specific arguments.

        Returns:
            Model output (tensor or dict of tensors for multi-task models).
        """
        ...

    @abstractmethod
    def get_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute training loss.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional arguments (e.g., weights, masks).

        Returns:
            Loss tensor (scalar).
        """
        ...

    def predict(self, x: torch.Tensor, **kwargs: Any) -> Any:
        """Run inference without gradients.

        Args:
            x: Input tensor.
            **kwargs: Additional model-specific arguments.

        Returns:
            Model predictions.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x, **kwargs)

    def test_inference(
        self,
        images: Sequence[str | Path | Image.Image | np.ndarray],
        image_size: tuple[int, int] = (224, 224),
        device: str | torch.device | None = None,
    ) -> dict[str, Any]:
        """Test inference with a list of images.

        This method provides a convenient way to test the model with
        various image inputs (file paths, PIL Images, or numpy arrays).

        Args:
            images: List of images - can be file paths, PIL Images, or numpy arrays.
            image_size: Target size for resizing images (H, W).
            device: Device to run inference on. If None, uses model's current device.

        Returns:
            Dictionary containing:
                - predictions: Model predictions as numpy array
                - images: Preprocessed images as numpy array (for visualization)
                - inference_time_ms: Total inference time in milliseconds
        """
        import time

        # Determine device
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)

        # Build preprocessing transform
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Process images
        processed_tensors: list[torch.Tensor] = []
        processed_images: list[np.ndarray] = []

        for img in images:
            # Convert to PIL Image
            if isinstance(img, (str, Path)):
                pil_img = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                pil_img = Image.fromarray(img).convert("RGB")
            elif isinstance(img, Image.Image):
                pil_img = img.convert("RGB")
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

            # Store original (resized) for visualization
            resized = pil_img.resize((image_size[1], image_size[0]))
            processed_images.append(np.array(resized))

            # Transform and add to batch
            tensor = transform(pil_img)
            processed_tensors.append(tensor)

        # Create batch
        batch = torch.stack(processed_tensors).to(device)

        # Run inference
        self.eval()
        start_time = time.perf_counter()
        with torch.no_grad():
            predictions = self.forward(batch)
        end_time = time.perf_counter()

        inference_time_ms = (end_time - start_time) * 1000

        return {
            "predictions": predictions.cpu().numpy(),
            "images": np.stack(processed_images),
            "inference_time_ms": inference_time_ms,
            "num_images": len(images),
            "device": str(device),
        }

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters (if applicable)."""
        pass

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters (if applicable)."""
        pass


class BaseTrainer(ABC, Generic[TConfig, TModel, TDataset]):
    """Abstract base class for model trainers.

    Provides common training loop, validation, checkpointing, and logging.
    Uses HuggingFace Accelerate for distributed training and mixed precision.
    Supports optional trackio logging for experiment tracking.

    Subclass this for task-specific training logic.

    Training Hooks (override these for custom behavior):
        - on_epoch_end(epoch, metrics): Called at end of each epoch for custom logging/visualization
        - on_train_end(result): Called after training completes for final processing
        - get_metric_for_checkpoint(val_loss, metrics): Return metric value for saving best model
    """

    def __init__(
        self,
        config: TConfig,
        model: TModel,
        train_dataset: TDataset,
        val_dataset: TDataset | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            config: Training configuration.
            model: Model to train.
            train_dataset: Training dataset.
            val_dataset: Optional validation dataset.
        """
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Initialize Accelerator for distributed training and mixed precision
        mixed_precision = "fp16" if config.mixed_precision else "no"
        log_with = "trackio" if config.use_trackio else None

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            log_with=log_with,
            gradient_accumulation_steps=1,
        )

        # Device is managed by accelerator
        self.device = self.accelerator.device

        # Create data loaders
        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        self.val_loader = (
            self._create_dataloader(val_dataset, shuffle=False) if val_dataset else None
        )

        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Prepare with accelerator (handles device placement, distributed, etc.)
        prepared = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
        )
        self.model = prepared[0]
        self.optimizer = prepared[1]
        self.train_loader = prepared[2]

        if self.val_loader:
            self.val_loader = self.accelerator.prepare(self.val_loader)

        if self.scheduler:
            self.scheduler = self.accelerator.prepare(self.scheduler)

        # Training state
        self.current_epoch = 0
        self.best_metric = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

        # Create output directories
        assert self.config.output_path is not None
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        self.config.logs_path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        if self.accelerator.is_main_process:
            self.config.save_config()

        # Set seed for reproducibility
        self._set_seed(config.seed)

        # Initialize trackio if enabled
        self._trackio_initialized = False
        if config.use_trackio and self.accelerator.is_main_process:
            self._init_trackio()

    def _init_trackio(self) -> None:
        """Initialize trackio tracking."""
        try:
            import trackio  # noqa: F401
        except ImportError:
            logger.warning("trackio not installed. Disabling trackio logging.")
            logger.info("Install with: uv sync --extra trackio")
            self.config.use_trackio = False
            return

        # Convert config to dict with Path objects as strings for trackio
        trackio_config = {}
        for key, value in self.config.model_dump().items():
            if isinstance(value, Path):
                trackio_config[key] = str(value)
            else:
                trackio_config[key] = value

        trackio_config["model_name"] = getattr(self.model, "name", "unknown")
        trackio_config["num_parameters"] = self._count_model_parameters()

        # Explicitly log run_id and output_path for mapping
        trackio_config["run_id"] = self.config.run_id
        trackio_config["output_path"] = str(self.config.output_path)
        trackio_config["logs_path"] = str(self.config.logs_path)

        space_id = self.config.trackio_project if self.config.use_space else None
        self.accelerator.init_trackers(
            project_name=self.config.trackio_project,
            config=trackio_config,
            init_kwargs={
                "trackio": {
                    "name": self.config.trackio_run_name,
                    "space_id": space_id
                }
            },
        )
        self._trackio_initialized = True
        logger.info(f"Initialized trackio project: {self.config.trackio_project}")
        logger.info(f"Trackio run name: {self.config.trackio_run_name}")
        logger.info(f"Run ID: {self.config.run_id}")

    def _count_model_parameters(self) -> int:
        """Count model parameters (handles accelerate-wrapped models)."""
        model = self.accelerator.unwrap_model(self.model)
        if hasattr(model, "count_parameters"):
            return model.count_parameters()
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _log_to_trackio(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log metrics to trackio if enabled."""
        if self._trackio_initialized and self.accelerator.is_main_process:
            self.accelerator.log(metrics, step=step)

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Enable benchmark for performance (auto-tune algorithms)
            torch.backends.cudnn.benchmark = True

    def _create_dataloader(
        self,
        dataset: TDataset,
        shuffle: bool = True,
    ) -> DataLoader[Any]:
        """Create a DataLoader from dataset with deterministic worker seeding."""
        # Create generator for reproducible shuffling
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=shuffle,
            worker_init_fn=_create_worker_init_fn(self.config.seed),
            generator=generator,
        )

    def _create_optimizer(self) -> Optimizer:
        """Create optimizer. Override for custom optimizers."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _create_scheduler(self) -> LRScheduler | None:
        """Create learning rate scheduler."""
        if self.config.scheduler_type == "none":
            return None

        total_steps = len(self.train_loader) * self.config.num_epochs

        if self.config.scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.config.learning_rate * 0.01,
            )
        elif self.config.scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma,
            )
        elif self.config.scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.scheduler_gamma,
                patience=self.config.scheduler_patience,
            )
        return None

    def train(self) -> TrainingResult:
        """Run full training loop.

        Returns:
            TrainingResult with final metrics and checkpoint path.
        """
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        model_name = getattr(unwrapped_model, "name", "Model")

        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Model: {model_name}")
        logger.info(f"Parameters: {self._count_model_parameters():,}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train samples: {len(self.train_dataset)}")  # type: ignore[arg-type]
        if self.val_dataset:
            logger.info(f"Val samples: {len(self.val_dataset)}")  # type: ignore[arg-type]
        if self.config.use_trackio:
            logger.info(f"Logging to trackio project: {self.config.trackio_project}")

        # Load checkpoint if specified
        if self.config.checkpoint_path:
            self._load_checkpoint(self.config.checkpoint_path)

        # Hook: training begins
        self.on_train_begin()

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            # Hook: epoch begins
            self.on_epoch_begin(epoch)

            # Training epoch
            train_loss = self._train_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # Validation
            val_loss: float | None = None
            metrics: dict[str, float] = {}
            if self.val_loader and (epoch + 1) % self.config.val_frequency == 0:
                val_loss, metrics = self._validate_epoch()
                self.history["val_loss"].append(val_loss)

                # Update metrics history
                for key, value in metrics.items():
                    if key not in self.history:
                        self.history[key] = []
                    self.history[key].append(value)

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    if val_loss is not None:
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Logging
            self._log_epoch(epoch, train_loss, val_loss, metrics)

            # Log to trackio
            trackio_metrics = {
                "train/loss": train_loss,
                "train/lr": self.optimizer.param_groups[0]["lr"],
            }
            if val_loss is not None:
                trackio_metrics["val/loss"] = val_loss
            for key, value in metrics.items():
                trackio_metrics[f"val/{key}"] = value
            self._log_to_trackio(trackio_metrics, step=epoch)

            # Hook: epoch ends (with all metrics for custom processing)
            epoch_metrics = {"train_loss": train_loss, "val_loss": val_loss, **metrics}
            self.on_epoch_end(epoch, epoch_metrics)

            # Checkpointing - use hook to get metric
            metric_for_checkpoint = self.get_metric_for_checkpoint(val_loss, metrics)
            is_best = metric_for_checkpoint < self.best_metric - self.config.min_delta
            if is_best:
                self.best_metric = metric_for_checkpoint
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1

            if (epoch + 1) % self.config.save_frequency == 0:
                self._save_checkpoint(is_best=False)

            # Early stopping
            if (
                self.config.early_stopping
                and self.patience_counter >= self.config.patience
            ):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        assert self.config.output_path is not None
        best_checkpoint = self.config.output_path / "best_model.pt"
        if best_checkpoint.exists():
            self._load_checkpoint(best_checkpoint)

        result = TrainingResult(
            best_epoch=self.best_epoch,
            best_metric=self.best_metric,
            final_train_loss=self.history["train_loss"][-1],
            final_val_loss=self.history["val_loss"][-1]
            if self.history["val_loss"]
            else 0.0,
            history=self.history,
            checkpoint_path=best_checkpoint,
        )

        # Hook: training ends
        self.on_train_end(result)

        # End trackio run
        if self._trackio_initialized:
            self.accelerator.end_training()
            self._trackio_initialized = False

        return result

    def _train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            loss = self._train_step(batch)
            total_loss += loss
            num_batches += 1

            if (batch_idx + 1) % self.config.log_frequency == 0:
                avg_loss = total_loss / num_batches
                logger.debug(
                    f"Epoch {self.current_epoch} [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {avg_loss:.6f}"
                )

        return total_loss / num_batches

    def _train_step(self, batch: Any) -> float:
        """Single training step.

        Args:
            batch: Batch from dataloader.

        Returns:
            Loss value for this batch.
        """
        inputs, targets = self._unpack_batch(batch)

        self.optimizer.zero_grad()

        # Accelerator handles mixed precision automatically
        with self.accelerator.autocast():
            predictions = self.model(inputs)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            loss = unwrapped_model.get_loss(predictions, targets)

        self.accelerator.backward(loss)

        if self.config.grad_clip:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )

        self.optimizer.step()

        return loss.item()

    def _validate_epoch(self) -> tuple[float, dict[str, float]]:
        """Run validation.

        Returns:
            Tuple of (average loss, metrics dict).
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in self.val_loader:  # type: ignore[union-attr]
                inputs, targets = self._unpack_batch(batch)

                with self.accelerator.autocast():
                    predictions = self.model(inputs)
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    loss = unwrapped_model.get_loss(predictions, targets)

                total_loss += loss.item()
                num_batches += 1

                # Gather predictions from all processes
                all_preds: torch.Tensor = self.accelerator.gather(predictions)  # type: ignore[assignment]
                all_tgts: torch.Tensor = self.accelerator.gather(targets)  # type: ignore[assignment]

                all_predictions.append(all_preds.cpu())
                all_targets.append(all_tgts.cpu())

        avg_loss = total_loss / num_batches
        predictions_cat = torch.cat(all_predictions, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        metrics = self._compute_metrics(predictions_cat, targets_cat)

        return avg_loss, metrics

    @abstractmethod
    def _unpack_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack batch into inputs and targets.

        Args:
            batch: Batch from dataloader.

        Returns:
            Tuple of (inputs, targets).
        """
        ...

    @abstractmethod
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        """Compute validation metrics.

        Args:
            predictions: All predictions from validation.
            targets: All targets from validation.

        Returns:
            Dictionary of metric names to values.
        """
        ...

    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None,
        metrics: dict[str, float],
    ) -> None:
        """Log epoch results."""
        lr = self.optimizer.param_groups[0]["lr"]
        msg = (
            f"Epoch {epoch + 1}/{self.config.num_epochs} - Train Loss: {train_loss:.6f}"
        )
        if val_loss is not None:
            msg += f" - Val Loss: {val_loss:.6f}"
        for key, value in metrics.items():
            msg += f" - {key}: {value:.4f}"
        msg += f" - LR: {lr:.2e}"
        logger.info(msg)

    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        # Only save on main process
        if not self.accelerator.is_main_process:
            return

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "config": self.config.model_dump(),
        }

        if is_best:
            assert self.config.output_path is not None
            path = self.config.output_path / "best_model.pt"
        else:
            assert self.config.output_path is not None
            path = (
                self.config.output_path
                / f"checkpoint_epoch_{self.current_epoch + 1}.pt"
            )

        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint: {path}")

    def _load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"] + 1
        self.best_metric = checkpoint["best_metric"]
        self.best_epoch = checkpoint["best_epoch"]
        self.history = checkpoint["history"]

        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")

    # ==================== Training Hooks ====================
    # Override these methods for custom behavior

    def on_train_begin(self) -> None:
        """Called before training starts.

        Use this for dataset logging, initial visualizations,
        or any setup that should happen after model initialization.
        """
        pass

    def on_epoch_begin(self, epoch: int) -> None:
        """Called at the beginning of each epoch.

        Args:
            epoch: Current epoch number (0-indexed).

        Use this for per-epoch setup like unfreezing layers.
        """
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Called at the end of each epoch.

        Args:
            epoch: Current epoch number (0-indexed).
            metrics: Combined metrics including 'train_loss', 'val_loss', and validation metrics.

        Use this for custom logging, visualization, or state updates.
        Override for task-specific behavior like saving visualizations.
        """
        pass

    def on_train_end(self, result: TrainingResult) -> None:
        """Called after training completes.

        Args:
            result: Training result with metrics and checkpoint path.

        Use this for final visualizations, cleanup, or post-processing.
        """
        pass

    def get_metric_for_checkpoint(
        self,
        val_loss: float | None,
        metrics: dict[str, float],
    ) -> float:
        """Get the metric value to use for checkpointing.

        Override this to use a different metric for best model selection.
        Lower values are considered better (for loss-like metrics).

        Args:
            val_loss: Validation loss (None if no validation).
            metrics: Dictionary of validation metrics.

        Returns:
            Metric value for checkpoint comparison.
        """
        # Default: use validation loss, fall back to train loss
        if val_loss is not None:
            return val_loss
        return self.history["train_loss"][-1] if self.history["train_loss"] else float("inf")

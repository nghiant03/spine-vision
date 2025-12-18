"""Base classes for training infrastructure.

Provides abstract interfaces for models and trainers that can be extended
for various tasks (localization, classification, segmentation).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import torch
import torch.nn as nn
from loguru import logger
from pydantic import BaseModel as PydanticBaseModel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset

# Type variables for generic trainer
TConfig = TypeVar("TConfig", bound="TrainingConfig")
TModel = TypeVar("TModel", bound=nn.Module)
TDataset = TypeVar("TDataset", bound=Dataset[Any])


class TrainingConfig(PydanticBaseModel):
    """Base configuration for training.

    Subclass this for task-specific configurations.
    """

    # Data paths
    data_path: Path = Path("data/gold/ivd_coords")
    output_path: Path = Path("outputs/training")
    checkpoint_path: Path | None = None

    # Training parameters
    batch_size: int = 32
    num_epochs: int = 100
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
    verbose: bool = False

    # Reproducibility
    seed: int = 42

    model_config = {"arbitrary_types_allowed": True}


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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Model output tensor.
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

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference without gradients.

        Args:
            x: Input tensor.

        Returns:
            Model predictions.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)

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
    Subclass this for task-specific training logic.
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

        # Set device
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Create data loaders
        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        self.val_loader = (
            self._create_dataloader(val_dataset, shuffle=False) if val_dataset else None
        )

        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.scaler = (
            torch.cuda.amp.GradScaler()  # type: ignore[attr-defined]
            if config.mixed_precision and self.device.type == "cuda"
            else None
        )

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

        # Create output directory
        self.config.output_path.mkdir(parents=True, exist_ok=True)

        # Set seed for reproducibility
        self._set_seed(config.seed)

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_dataloader(
        self,
        dataset: TDataset,
        shuffle: bool = True,
    ) -> DataLoader[Any]:
        """Create a DataLoader from dataset."""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=shuffle,
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
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Model: {self.model.name}")
        logger.info(f"Parameters: {self.model.count_parameters():,}")  # type: ignore[union-attr]
        logger.info(f"Device: {self.device}")
        logger.info(f"Train samples: {len(self.train_dataset)}")  # type: ignore[arg-type]
        if self.val_dataset:
            logger.info(f"Val samples: {len(self.val_dataset)}")  # type: ignore[arg-type]

        # Load checkpoint if specified
        if self.config.checkpoint_path:
            self._load_checkpoint(self.config.checkpoint_path)

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch

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

            # Checkpointing
            metric_for_checkpoint = val_loss if val_loss is not None else train_loss
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
        best_checkpoint = self.config.output_path / "best_model.pt"
        if best_checkpoint.exists():
            self._load_checkpoint(best_checkpoint)

        return TrainingResult(
            best_epoch=self.best_epoch,
            best_metric=self.best_metric,
            final_train_loss=self.history["train_loss"][-1],
            final_val_loss=self.history["val_loss"][-1]
            if self.history["val_loss"]
            else 0.0,
            history=self.history,
            checkpoint_path=best_checkpoint,
        )

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
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        self.optimizer.zero_grad()

        if self.scaler:
            with torch.cuda.amp.autocast():  # type: ignore[attr-defined]
                predictions = self.model(inputs)
                loss = self.model.get_loss(predictions, targets)  # type: ignore[union-attr]

            self.scaler.scale(loss).backward()  # type: ignore[union-attr]
            if self.config.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            predictions = self.model(inputs)
            loss = self.model.get_loss(predictions, targets)  # type: ignore[union-attr]
            loss.backward()
            if self.config.grad_clip:
                torch.nn.utils.clip_grad_norm_(
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
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                if self.scaler:
                    with torch.cuda.amp.autocast():  # type: ignore[attr-defined]
                        predictions = self.model(inputs)
                        loss = self.model.get_loss(predictions, targets)  # type: ignore[union-attr]
                else:
                    predictions = self.model(inputs)
                    loss = self.model.get_loss(predictions, targets)  # type: ignore[union-attr]

                total_loss += loss.item()
                num_batches += 1

                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

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
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
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
            path = self.config.output_path / "best_model.pt"
        else:
            path = (
                self.config.output_path
                / f"checkpoint_epoch_{self.current_epoch + 1}.pt"
            )

        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint: {path}")

    def _load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"] + 1
        self.best_metric = checkpoint["best_metric"]
        self.best_epoch = checkpoint["best_epoch"]
        self.history = checkpoint["history"]

        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")

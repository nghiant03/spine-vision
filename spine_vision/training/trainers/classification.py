"""Classification trainer for multi-task lumbar spine grading.

Specialized trainer for ResNet50-MTL model training with dual-modality input.
Uses HuggingFace Accelerate and supports wandb logging.
"""

from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader

from spine_vision.training.base import BaseTrainer, TrainingConfig, TrainingResult
from spine_vision.training.datasets.classification import (
    ClassificationCollator,
    ClassificationDataset,
    CropOnlyDataset,
    IDX_TO_LEVEL,
    LEVEL_TO_IDX,
)
from spine_vision.training.metrics import ClassificationMetrics
from spine_vision.training.models.resnet_mtl import MTLPredictions, MTLTargets, ResNet50MTL
from spine_vision.training.visualization import TrainingVisualizer


class MTLClassificationMetrics:
    """Metrics calculator for multi-task classification.
    
    Computes metrics for each task head:
    - Pfirrmann: 5-class accuracy
    - Modic: 4-class accuracy
    - Binary heads: AUC, F1, Precision, Recall
    """
    
    def __init__(self) -> None:
        """Initialize metrics for all heads."""
        self.pfirrmann_metrics = ClassificationMetrics(
            num_classes=5,
            class_names=[f"Grade_{i+1}" for i in range(5)],
        )
        self.modic_metrics = ClassificationMetrics(
            num_classes=4,
            class_names=[f"Type_{i}" for i in range(4)],
        )
        
        # Accumulators for binary heads
        self._herniation_preds: list[np.ndarray] = []
        self._herniation_targets: list[np.ndarray] = []
        self._endplate_preds: list[np.ndarray] = []
        self._endplate_targets: list[np.ndarray] = []
        self._spondy_preds: list[np.ndarray] = []
        self._spondy_targets: list[np.ndarray] = []
        self._narrowing_preds: list[np.ndarray] = []
        self._narrowing_targets: list[np.ndarray] = []
    
    def reset(self) -> None:
        """Reset all accumulators."""
        self.pfirrmann_metrics.reset()
        self.modic_metrics.reset()
        self._herniation_preds = []
        self._herniation_targets = []
        self._endplate_preds = []
        self._endplate_targets = []
        self._spondy_preds = []
        self._spondy_targets = []
        self._narrowing_preds = []
        self._narrowing_targets = []
    
    def update(
        self,
        predictions: MTLPredictions,
        targets: MTLTargets,
    ) -> None:
        """Accumulate predictions from a batch."""
        # Multiclass heads
        pfirrmann_pred = predictions.pfirrmann.argmax(dim=1).cpu().numpy()
        modic_pred = predictions.modic.argmax(dim=1).cpu().numpy()
        
        self.pfirrmann_metrics.update(
            pfirrmann_pred,
            targets.pfirrmann.cpu().numpy(),
        )
        self.modic_metrics.update(
            modic_pred,
            targets.modic.cpu().numpy(),
        )
        
        # Binary heads (sigmoid probabilities)
        self._herniation_preds.append(
            torch.sigmoid(predictions.herniation).cpu().numpy()
        )
        self._herniation_targets.append(targets.herniation.cpu().numpy())
        
        self._endplate_preds.append(
            torch.sigmoid(predictions.endplate).cpu().numpy()
        )
        self._endplate_targets.append(targets.endplate.cpu().numpy())
        
        self._spondy_preds.append(
            torch.sigmoid(predictions.spondy).cpu().numpy()
        )
        self._spondy_targets.append(targets.spondy.cpu().numpy())
        
        self._narrowing_preds.append(
            torch.sigmoid(predictions.narrowing).cpu().numpy()
        )
        self._narrowing_targets.append(targets.narrowing.cpu().numpy())
    
    def compute(self) -> dict[str, float]:
        """Compute all metrics."""
        metrics: dict[str, float] = {}
        
        # Multiclass metrics
        pfirrmann = self.pfirrmann_metrics.compute()
        modic = self.modic_metrics.compute()
        
        metrics["pfirrmann_accuracy"] = pfirrmann.get("accuracy", 0.0)
        metrics["pfirrmann_balanced_acc"] = pfirrmann.get("balanced_accuracy", 0.0)
        metrics["modic_accuracy"] = modic.get("accuracy", 0.0)
        metrics["modic_balanced_acc"] = modic.get("balanced_accuracy", 0.0)
        
        # Binary metrics
        binary_heads = [
            ("herniation", self._herniation_preds, self._herniation_targets, 2),
            ("endplate", self._endplate_preds, self._endplate_targets, 2),
            ("spondy", self._spondy_preds, self._spondy_targets, 1),
            ("narrowing", self._narrowing_preds, self._narrowing_targets, 1),
        ]
        
        for name, preds_list, targets_list, n_outputs in binary_heads:
            if not preds_list:
                continue
            
            preds = np.concatenate(preds_list, axis=0)
            targets = np.concatenate(targets_list, axis=0)
            
            # Compute per-output metrics
            for i in range(n_outputs):
                if n_outputs == 1:
                    p = preds.flatten()
                    t = targets.flatten()
                    suffix = ""
                else:
                    p = preds[:, i]
                    t = targets[:, i]
                    suffix = f"_{i}"
                
                # Binary predictions
                pred_binary = (p > 0.5).astype(int)
                t_binary = t.astype(int)
                
                # Accuracy
                acc = np.mean(pred_binary == t_binary) * 100
                metrics[f"{name}{suffix}_accuracy"] = float(acc)
                
                # Precision, Recall, F1
                tp = np.sum((pred_binary == 1) & (t_binary == 1))
                fp = np.sum((pred_binary == 1) & (t_binary == 0))
                fn = np.sum((pred_binary == 0) & (t_binary == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                metrics[f"{name}{suffix}_precision"] = float(precision)
                metrics[f"{name}{suffix}_recall"] = float(recall)
                metrics[f"{name}{suffix}_f1"] = float(f1)
        
        # Overall average accuracy
        accs = [
            metrics.get("pfirrmann_accuracy", 0),
            metrics.get("modic_accuracy", 0),
            metrics.get("herniation_0_accuracy", metrics.get("herniation_accuracy", 0)),
            metrics.get("endplate_0_accuracy", metrics.get("endplate_accuracy", 0)),
            metrics.get("spondy_accuracy", 0),
            metrics.get("narrowing_accuracy", 0),
        ]
        metrics["overall_accuracy"] = float(np.mean(accs))
        
        return metrics


class ClassificationConfig(TrainingConfig):
    """Configuration for multi-task classification training."""
    
    task: str = "classification"
    
    # Model configuration
    pretrained: bool = True
    dropout: float = 0.3
    freeze_backbone_epochs: int = 0
    label_smoothing: float = 0.1
    
    # Dataset configuration
    use_preextracted_crops: bool = False
    """Use CropOnlyDataset with pre-extracted crops."""
    
    levels: list[str] | None = None
    """Filter to specific IVD levels."""
    
    localization_size: tuple[int, int] = (224, 224)
    """Size used by localization model."""
    
    crop_size: int = 128
    """Size of crop from original image (before resize)."""
    
    output_size: tuple[int, int] = (224, 224)
    """Final input size to model."""
    
    t1_subdir: str = "sag_t1"
    t2_subdir: str = "sag_t2"
    
    augment: bool = True
    
    # Visualization
    visualize_predictions: bool = True
    num_visualization_samples: int = 16


class ClassificationTrainer(
    BaseTrainer[ClassificationConfig, ResNet50MTL, ClassificationDataset | CropOnlyDataset]
):
    """Trainer for multi-task lumbar spine classification.
    
    Uses ResNet50-MTL model with 6 classification heads.
    Supports dual-modality input (T1 + T2 crops).
    """
    
    def __init__(
        self,
        config: ClassificationConfig,
        model: ResNet50MTL | None = None,
        train_dataset: ClassificationDataset | CropOnlyDataset | None = None,
        val_dataset: ClassificationDataset | CropOnlyDataset | None = None,
    ) -> None:
        """Initialize trainer."""
        # Create model if not provided
        if model is None:
            model = ResNet50MTL(
                pretrained=config.pretrained,
                dropout=config.dropout,
                freeze_backbone=config.freeze_backbone_epochs > 0,
                label_smoothing=config.label_smoothing,
            )
        
        # Create datasets if not provided
        if train_dataset is None:
            if config.use_preextracted_crops:
                train_dataset = CropOnlyDataset(
                    data_path=config.data_path,
                    split="train",
                    val_ratio=config.val_split,
                    levels=config.levels,
                    output_size=config.output_size,
                    augment=config.augment,
                )
            else:
                train_dataset = ClassificationDataset(
                    data_path=config.data_path,
                    split="train",
                    val_ratio=config.val_split,
                    levels=config.levels,
                    localization_size=config.localization_size,
                    crop_size=config.crop_size,
                    output_size=config.output_size,
                    augment=config.augment,
                    t1_subdir=config.t1_subdir,
                    t2_subdir=config.t2_subdir,
                )
        
        if val_dataset is None:
            if config.use_preextracted_crops:
                val_dataset = CropOnlyDataset(
                    data_path=config.data_path,
                    split="val",
                    val_ratio=config.val_split,
                    levels=config.levels,
                    output_size=config.output_size,
                    augment=False,
                )
            else:
                val_dataset = ClassificationDataset(
                    data_path=config.data_path,
                    split="val",
                    val_ratio=config.val_split,
                    levels=config.levels,
                    localization_size=config.localization_size,
                    crop_size=config.crop_size,
                    output_size=config.output_size,
                    augment=False,
                    t1_subdir=config.t1_subdir,
                    t2_subdir=config.t2_subdir,
                )
        
        super().__init__(config, model, train_dataset, val_dataset)
        
        # Metrics
        self.metrics = MTLClassificationMetrics()
        
        # Visualizer
        self.visualizer = TrainingVisualizer(
            output_path=config.logs_path,
            output_mode="html",
            use_wandb=config.use_wandb,
        )
        
        # Track backbone freeze state
        self._backbone_unfrozen = config.freeze_backbone_epochs == 0
    
    def _create_dataloader(
        self,
        dataset: ClassificationDataset | CropOnlyDataset,
        shuffle: bool = True,
    ) -> DataLoader[Any]:
        """Create DataLoader with custom collator."""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=shuffle,
            collate_fn=ClassificationCollator(),
        )
    
    def _unpack_batch(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, MTLTargets]:
        """Unpack batch into inputs and targets."""
        return batch["image"], batch["targets"]
    
    def _train_step(self, batch: dict[str, Any]) -> float:
        """Single training step."""
        inputs = batch["image"]
        targets: MTLTargets = batch["targets"]
        
        self.optimizer.zero_grad()
        
        with self.accelerator.autocast():
            predictions: MTLPredictions = self.model(inputs)
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
        """Validate and compute metrics."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        self.metrics.reset()
        
        # For visualization
        sample_images: list[np.ndarray] = []
        sample_predictions: list[MTLPredictions] = []
        sample_targets: list[MTLTargets] = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):  # type: ignore[union-attr]
                inputs = batch["image"]
                targets: MTLTargets = batch["targets"]
                
                with self.accelerator.autocast():
                    predictions: MTLPredictions = self.model(inputs)
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    loss = unwrapped_model.get_loss(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update metrics
                self.metrics.update(predictions, targets)
                
                # Collect samples for visualization
                if (
                    self.config.visualize_predictions
                    and self.accelerator.is_main_process
                    and len(sample_images) < self.config.num_visualization_samples
                ):
                    images_np = self._denormalize_images(inputs)
                    for i, img in enumerate(images_np):
                        if len(sample_images) < self.config.num_visualization_samples:
                            sample_images.append(img)
        
        avg_loss = total_loss / num_batches
        metrics = self.metrics.compute()
        
        return avg_loss, metrics
    
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        """Compute metrics (placeholder for base class compatibility)."""
        return {}
    
    def _denormalize_images(self, images: torch.Tensor) -> list[np.ndarray]:
        """Denormalize ImageNet-normalized images."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        images = images.cpu()
        images = images * std + mean
        images = torch.clamp(images, 0, 1)
        
        images_np = images.permute(0, 2, 3, 1).numpy()
        images_np = (images_np * 255).astype(np.uint8)
        
        return [img for img in images_np]
    
    def train(self) -> TrainingResult:
        """Train with backbone unfreezing support."""
        if self.config.freeze_backbone_epochs > 0:
            logger.info(
                f"Backbone frozen for first {self.config.freeze_backbone_epochs} epochs"
            )
        
        return self._train_loop()
    
    def _train_loop(self) -> TrainingResult:
        """Main training loop."""
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Model: {unwrapped_model.name}")
        logger.info(f"Parameters: {unwrapped_model.count_parameters():,}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train samples: {len(self.train_dataset)}")
        if self.val_dataset:
            logger.info(f"Val samples: {len(self.val_dataset)}")
        if self.config.use_wandb:
            logger.info(f"Logging to wandb: {self.config.wandb_project}")
        
        # Log dataset stats
        stats = self.train_dataset.get_stats()
        logger.info(f"Train dataset stats: {stats}")
        
        if self.config.checkpoint_path:
            self._load_checkpoint(self.config.checkpoint_path)
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Unfreeze backbone if needed
            if (
                not self._backbone_unfrozen
                and epoch >= self.config.freeze_backbone_epochs
            ):
                logger.info(f"Unfreezing backbone at epoch {epoch + 1}")
                unwrapped_model.unfreeze_backbone()
                self._backbone_unfrozen = True
            
            # Training
            train_loss = self._train_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            
            # Validation
            val_loss: float | None = None
            metrics: dict[str, float] = {}
            if self.val_loader and (epoch + 1) % self.config.val_frequency == 0:
                val_loss, metrics = self._validate_epoch()
                self.history["val_loss"].append(val_loss)
                
                for key, value in metrics.items():
                    if key not in self.history:
                        self.history[key] = []
                    self.history[key].append(value)
            
            # Scheduler step
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
            
            # Log to wandb
            wandb_metrics = {
                "train/loss": train_loss,
                "train/lr": self.optimizer.param_groups[0]["lr"],
            }
            if val_loss is not None:
                wandb_metrics["val/loss"] = val_loss
            for key, value in metrics.items():
                wandb_metrics[f"val/{key}"] = value
            self._log_to_wandb(wandb_metrics, step=epoch)
            
            # Checkpointing (use overall accuracy, higher is better)
            metric_for_checkpoint = -metrics.get(
                "overall_accuracy", -(val_loss if val_loss else train_loss)
            )
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
        
        # Generate final visualizations
        if self.accelerator.is_main_process:
            self._generate_final_visualizations()
        
        # End wandb run
        if self._wandb_initialized:
            self.accelerator.end_training()
        
        return TrainingResult(
            best_epoch=self.best_epoch,
            best_metric=-self.best_metric,  # Convert back to positive
            final_train_loss=self.history["train_loss"][-1],
            final_val_loss=self.history["val_loss"][-1]
            if self.history["val_loss"]
            else 0.0,
            history=self.history,
            checkpoint_path=best_checkpoint,
        )
    
    def _generate_final_visualizations(self) -> None:
        """Generate final training visualizations."""
        self.visualizer.plot_training_curves(
            self.history,
            filename="training_curves",
        )
        
        logger.info(f"Visualizations saved to: {self.config.logs_path}")
    
    def evaluate(
        self,
        test_dataset: ClassificationDataset | CropOnlyDataset | None = None,
    ) -> dict[str, float]:
        """Evaluate model on test set."""
        if test_dataset is None:
            if self.config.use_preextracted_crops:
                test_dataset = CropOnlyDataset(
                    data_path=self.config.data_path,
                    split="test",
                    val_ratio=self.config.val_split,
                    levels=self.config.levels,
                    output_size=self.config.output_size,
                    augment=False,
                )
            else:
                test_dataset = ClassificationDataset(
                    data_path=self.config.data_path,
                    split="test",
                    val_ratio=self.config.val_split,
                    levels=self.config.levels,
                    localization_size=self.config.localization_size,
                    crop_size=self.config.crop_size,
                    output_size=self.config.output_size,
                    augment=False,
                    t1_subdir=self.config.t1_subdir,
                    t2_subdir=self.config.t2_subdir,
                )
        
        test_loader = self._create_dataloader(test_dataset, shuffle=False)
        test_loader = self.accelerator.prepare(test_loader)
        
        self.model.eval()
        self.metrics.reset()
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["image"]
                targets: MTLTargets = batch["targets"]
                
                predictions: MTLPredictions = self.model(inputs)
                self.metrics.update(predictions, targets)
        
        metrics = self.metrics.compute()
        
        logger.info("Test Results:")
        for key, value in sorted(metrics.items()):
            logger.info(f"  {key}: {value:.4f}")
        
        # Log to wandb
        if self._wandb_initialized:
            wandb_metrics = {f"test/{key}": value for key, value in metrics.items()}
            self._log_to_wandb(wandb_metrics)
        
        return metrics

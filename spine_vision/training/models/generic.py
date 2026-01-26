"""Generic vision models with configurable backbone and heads.

Provides composable model architectures:
- Classifier: Single-task and multi-task classification with configurable heads
- CoordinateRegressor: Coordinate regression for localization

All models use:
- Configurable backbone via BackboneFactory
- Standard BaseModel interface
- Strategy pattern for task-type-specific behavior

Usage:
    from spine_vision.training.models import Classifier
    from spine_vision.core.tasks import get_task, get_tasks

    # Single-task classification
    task = get_task("pfirrmann")
    model = Classifier(backbone="resnet50", tasks=[task])
    output = model(images)  # {"pfirrmann": logits}

    # Multi-task classification with all lumbar spine tasks
    model = Classifier(backbone="convnext_base", tasks=get_tasks())
    output = model(images)  # {"pfirrmann": logits, "modic": logits, ...}
"""

from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from spine_vision.core.tasks import (
    TaskConfig,
    compute_predictions_for_tasks,
    compute_probabilities_for_tasks,
    create_loss_functions,
    get_strategy,
    get_tasks,
)
from spine_vision.training.base import BaseModel
from spine_vision.training.heads import HeadConfig, create_head
from spine_vision.training.models.backbone import BackboneFactory
from spine_vision.training.registry import register_model


@register_model("classifier")
class Classifier(BaseModel):
    """Generic classifier with configurable backbone and task heads.

    Supports single-task and multi-task classification with a unified API.
    Returns dict[str, Tensor] for consistent interface regardless of task count.

    Architecture:
        - Configurable backbone (ResNet, ConvNeXt, ViT, etc.)
        - Global Average Pooling -> feature_dim
        - Separate heads per task

    Task types handled via strategy pattern:
        - multiclass: CrossEntropyLoss, argmax predictions
        - binary: BCEWithLogitsLoss, sigmoid > 0.5
        - multilabel: BCEWithLogitsLoss, sigmoid > 0.5
        - ordinal: CrossEntropyLoss (extensible for CORAL)
        - regression: MSELoss
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        tasks: list[TaskConfig] | None = None,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize Classifier.

        Args:
            backbone: Backbone name (see BackboneFactory for options).
            tasks: List of task configurations. If None, uses all registered tasks.
            pretrained: Use pretrained weights.
            dropout: Dropout rate for feature layer.
            freeze_backbone: Freeze backbone weights.
        """
        super().__init__()

        self._backbone_name = backbone
        self._pretrained = pretrained
        self._dropout = dropout
        self._freeze_backbone = freeze_backbone

        # Use all registered tasks if none provided
        if tasks is None:
            tasks = get_tasks()
        self._tasks = tasks
        self._task_names = [t.name for t in tasks]

        # Create backbone
        self.backbone, feature_dim = BackboneFactory.create(backbone, pretrained)
        self._feature_dim = feature_dim

        # Shared dropout
        self.dropout = nn.Dropout(dropout)

        # Create task heads
        self.heads = nn.ModuleDict()
        for task in tasks:
            self.heads[task.name] = nn.Linear(feature_dim, task.num_classes)

        # Initialize loss functions using strategies
        self._loss_functions, self._loss_weights = create_loss_functions(tasks)

        if freeze_backbone:
            self.freeze_backbone()

        self._is_initialized = True

    @property
    def name(self) -> str:
        return f"Classifier-{self._backbone_name}"

    @property
    def task_names(self) -> list[str]:
        return self._task_names

    @property
    def tasks(self) -> list[TaskConfig]:
        return self._tasks

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input images [B, C, H, W].

        Returns:
            Dictionary mapping task names to output logits.
        """
        features = self.backbone(x)
        features = self.dropout(features)
        return {name: head(features) for name, head in self.heads.items()}

    def get_loss(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute multi-task loss.

        Total loss = sum(weight_i * loss_i) for each task.
        Targets are formatted via strategy before loss computation.
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        for task in self._tasks:
            if task.name not in predictions or task.name not in targets:
                continue

            pred = predictions[task.name]
            target = targets[task.name]

            # Format target via strategy
            strategy = get_strategy(task)
            target = strategy.format_target(target)

            loss_fn = self._loss_functions[task.name]
            weight = self._loss_weights[task.name]

            task_loss = loss_fn(pred, target)
            total_loss = total_loss + weight * task_loss

        return total_loss

    def get_loss_breakdown(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Get individual loss values for each task."""
        losses: dict[str, torch.Tensor] = {}
        for task in self._tasks:
            if task.name not in predictions or task.name not in targets:
                continue

            strategy = get_strategy(task)
            target = strategy.format_target(targets[task.name])
            losses[task.name] = self._loss_functions[task.name](
                predictions[task.name], target
            )
        return losses

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict(self, x: torch.Tensor, **kwargs: Any) -> dict[str, np.ndarray]:
        """Run inference and return final predictions via strategies."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, **kwargs)
        return compute_predictions_for_tasks(outputs, self._tasks)

    def predict_proba(self, x: torch.Tensor, **kwargs: Any) -> dict[str, np.ndarray]:
        """Run inference and return probabilities via strategies."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, **kwargs)
        return compute_probabilities_for_tasks(outputs, self._tasks)

    def test_inference(
        self,
        images: Sequence[str | Path | Image.Image | np.ndarray],
        image_size: tuple[int, int] = (224, 224),
        device: str | torch.device | None = None,
    ) -> dict[str, Any]:
        """Test inference with images."""
        import time

        from torchvision import transforms

        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        processed_tensors: list[torch.Tensor] = []
        processed_images: list[np.ndarray] = []

        for img in images:
            if isinstance(img, (str, Path)):
                pil_img = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                pil_img = Image.fromarray(img).convert("RGB")
            elif isinstance(img, Image.Image):
                pil_img = img.convert("RGB")
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

            resized = pil_img.resize((image_size[1], image_size[0]))
            processed_images.append(np.array(resized))
            processed_tensors.append(transform(pil_img))

        batch = torch.stack(processed_tensors).to(device)

        self.eval()
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.forward(batch)
        inference_time_ms = (time.perf_counter() - start_time) * 1000

        predictions = compute_predictions_for_tasks(outputs, self._tasks)
        probabilities = compute_probabilities_for_tasks(outputs, self._tasks)

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "images": np.stack(processed_images),
            "inference_time_ms": inference_time_ms,
            "num_images": len(images),
            "device": str(device),
        }


@register_model("coordinate_regressor")
class CoordinateRegressor(BaseModel):
    """Generic coordinate regressor with configurable backbone and head.

    Architecture:
        - Configurable backbone (ResNet, ConvNeXt, ViT, etc.)
        - Global Average Pooling -> feature_dim
        - Regression head outputting all levels at once

    Output: Normalized coordinates in [0, 1] for all levels.
    Shape: [B, num_levels, 2] where 2 is (x, y).
    """

    def __init__(
        self,
        backbone: str = "convnext_base",
        num_outputs: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        head_config: HeadConfig | None = None,
        num_levels: int = 5,
        loss_type: Literal["mse", "smooth_l1", "huber"] = "smooth_l1",
    ) -> None:
        """Initialize CoordinateRegressor.

        Args:
            backbone: Backbone name (see BackboneFactory for options).
            num_outputs: Number of output coordinates per level (default 2 for x,y).
            pretrained: Use pretrained weights.
            dropout: Dropout rate.
            freeze_backbone: Freeze backbone weights.
            head_config: Custom head configuration.
            num_levels: Number of levels to predict (default 5 for IVD levels).
            loss_type: Loss function type.
        """
        super().__init__()

        self._backbone_name = backbone
        self._num_outputs = num_outputs
        self._pretrained = pretrained
        self._dropout = dropout
        self._freeze_backbone = freeze_backbone
        self._num_levels = num_levels
        self._loss_type = loss_type

        # Create backbone
        self.backbone, feature_dim = BackboneFactory.create(backbone, pretrained)
        self._feature_dim = feature_dim

        # Total outputs: num_levels * num_outputs (e.g., 5 levels * 2 coords = 10)
        total_outputs = num_levels * num_outputs

        # Create head
        if head_config is not None:
            self.head = create_head(head_config, feature_dim, total_outputs)
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(256, total_outputs),
                nn.Sigmoid(),
            )

        # Loss function
        if loss_type == "mse":
            self._loss_fn = nn.MSELoss()
        elif loss_type == "smooth_l1":
            self._loss_fn = nn.SmoothL1Loss()
        elif loss_type == "huber":
            self._loss_fn = nn.HuberLoss(delta=0.1)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        if freeze_backbone:
            self.freeze_backbone()

        self._is_initialized = True

    @property
    def name(self) -> str:
        return f"Regressor-{self._backbone_name}"

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def num_levels(self) -> int:
        return self._num_levels

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images [B, C, H, W].

        Returns:
            Predicted coordinates [B, num_levels, 2] in [0, 1].
        """
        features = self.backbone(x)
        output = self.head(features)  # [B, num_levels * 2]
        return output.view(-1, self._num_levels, self._num_outputs)

    def get_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute regression loss.

        Args:
            predictions: Predicted coordinates [B, num_levels, 2].
            targets: Ground truth coordinates [B, num_levels, 2].
            mask: Optional mask [B, num_levels] for valid targets.

        Returns:
            Scalar loss value.
        """
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(predictions)
            valid_preds = predictions[mask_expanded.bool()]
            valid_targets = targets[mask_expanded.bool()]
            if valid_preds.numel() == 0:
                return torch.tensor(0.0, device=predictions.device)
            return self._loss_fn(valid_preds, valid_targets)
        return self._loss_fn(predictions, targets)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def test_inference(
        self,
        images: Sequence[str | Path | Image.Image | np.ndarray],
        image_size: tuple[int, int] = (224, 224),
        device: str | torch.device | None = None,
    ) -> dict[str, Any]:
        """Test inference with images."""
        import time

        from torchvision import transforms

        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        processed_tensors: list[torch.Tensor] = []
        processed_images: list[np.ndarray] = []

        for img in images:
            if isinstance(img, (str, Path)):
                pil_img = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                pil_img = Image.fromarray(img).convert("RGB")
            elif isinstance(img, Image.Image):
                pil_img = img.convert("RGB")
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

            resized = pil_img.resize((image_size[1], image_size[0]))
            processed_images.append(np.array(resized))
            processed_tensors.append(transform(pil_img))

        batch = torch.stack(processed_tensors).to(device)

        self.eval()
        start_time = time.perf_counter()
        with torch.no_grad():
            predictions = self.forward(batch)
        inference_time_ms = (time.perf_counter() - start_time) * 1000

        predictions_np = predictions.cpu().numpy()
        h, w = image_size
        coords_pixel = predictions_np * np.array([w, h])

        return {
            "predictions": predictions_np,
            "coords_pixel": coords_pixel,
            "images": np.stack(processed_images),
            "inference_time_ms": inference_time_ms,
            "num_images": len(images),
            "device": str(device),
        }


def list_backbones(family: str | None = None) -> list[str]:
    """List available backbone names."""
    return BackboneFactory.list_backbones(family)

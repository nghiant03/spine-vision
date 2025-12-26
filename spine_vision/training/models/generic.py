"""Generic vision models with configurable backbone and heads.

Provides composable model architectures:
- ImageClassifier: Single-task classification
- MultiTaskClassifier: Multi-task classification with multiple heads
- CoordinateRegressor: Coordinate regression for localization

All models use:
- Configurable backbone via BackboneFactory
- Configurable head via HeadConfig
- Standard BaseModel interface

Usage:
    from spine_vision.training.models import ImageClassifier, MultiTaskClassifier

    # Single-task classification
    model = ImageClassifier(
        backbone="resnet50",
        num_classes=5,
        head_config=HeadConfig(head_type="mlp", hidden_dims=[512]),
    )

    # Multi-task classification
    model = MultiTaskClassifier(
        backbone="convnext_base",
        tasks=[
            TaskConfig(name="grade", num_classes=5, task_type="multiclass"),
            TaskConfig(name="condition", num_classes=1, task_type="binary"),
        ],
    )
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from spine_vision.training.base import BaseModel
from spine_vision.training.heads import HeadConfig, create_head
from spine_vision.training.models.backbone import BackboneFactory
from spine_vision.training.registry import register_model


@dataclass
class TaskConfig:
    """Configuration for a single task in multi-task learning.

    Attributes:
        name: Task name (used as key in outputs).
        num_classes: Number of output classes/values.
        task_type: Type of task (multiclass, binary, multilabel, regression).
        head_config: Optional custom head configuration.
        loss_weight: Weight for this task's loss in total loss.
        label_smoothing: Label smoothing for cross-entropy (multiclass only).
        class_weights: Optional class weights for imbalanced data.
    """

    name: str
    num_classes: int
    task_type: Literal["multiclass", "binary", "multilabel", "regression"] = "multiclass"
    head_config: HeadConfig | None = None
    loss_weight: float = 1.0
    label_smoothing: float = 0.0
    class_weights: torch.Tensor | None = None


# Predefined task configurations for lumbar spine classification
LUMBAR_SPINE_TASKS: list[TaskConfig] = [
    TaskConfig(name="pfirrmann", num_classes=5, task_type="multiclass", label_smoothing=0.1),
    TaskConfig(name="modic", num_classes=4, task_type="multiclass", label_smoothing=0.1),
    TaskConfig(name="herniation", num_classes=2, task_type="multilabel"),
    TaskConfig(name="endplate", num_classes=2, task_type="multilabel"),
    TaskConfig(name="spondy", num_classes=1, task_type="binary"),
    TaskConfig(name="narrowing", num_classes=1, task_type="binary"),
]


@dataclass
class MTLTargets:
    """Container for multi-task targets.

    For lumbar spine classification with 6 predefined tasks.
    """

    pfirrmann: torch.Tensor  # [B] int64, values 0-4
    modic: torch.Tensor  # [B] int64, values 0-3
    herniation: torch.Tensor  # [B, 2] float32, 0.0 or 1.0
    endplate: torch.Tensor  # [B, 2] float32, 0.0 or 1.0
    spondy: torch.Tensor  # [B, 1] float32, 0.0 or 1.0
    narrowing: torch.Tensor  # [B, 1] float32, 0.0 or 1.0

    def to(self, device: torch.device | str) -> "MTLTargets":
        """Move all tensors to the specified device."""
        return MTLTargets(
            pfirrmann=self.pfirrmann.to(device),
            modic=self.modic.to(device),
            herniation=self.herniation.to(device),
            endplate=self.endplate.to(device),
            spondy=self.spondy.to(device),
            narrowing=self.narrowing.to(device),
        )

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert to dictionary format."""
        return {
            "pfirrmann": self.pfirrmann,
            "modic": self.modic,
            "herniation": self.herniation,
            "endplate": self.endplate,
            "spondy": self.spondy,
            "narrowing": self.narrowing,
        }


@register_model(
    "classifier",
    task="classification",
    description="Single-task image classifier with configurable backbone",
    aliases=["image_classifier"],
)
class ImageClassifier(BaseModel):
    """Generic image classifier with configurable backbone and head.

    Architecture:
        - Configurable backbone (ResNet, ConvNeXt, ViT, etc.)
        - Global Average Pooling -> feature_dim
        - Configurable classification head
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        head_config: HeadConfig | None = None,
        label_smoothing: float = 0.1,
    ) -> None:
        """Initialize ImageClassifier.

        Args:
            backbone: Backbone name (see BackboneFactory for options).
            num_classes: Number of output classes.
            pretrained: Use pretrained weights.
            dropout: Dropout rate (used if no head_config).
            freeze_backbone: Freeze backbone weights.
            head_config: Custom head configuration.
            label_smoothing: Label smoothing for cross-entropy.
        """
        super().__init__()

        self._backbone_name = backbone
        self._num_classes = num_classes
        self._pretrained = pretrained
        self._dropout = dropout
        self._freeze_backbone = freeze_backbone

        # Create backbone
        self.backbone, feature_dim = BackboneFactory.create(backbone, pretrained)
        self._feature_dim = feature_dim

        # Create head
        if head_config is not None:
            self.head = create_head(head_config, feature_dim, num_classes)
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes),
            )

        # Loss function
        self._loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if freeze_backbone:
            self.freeze_backbone()

        self._is_initialized = True

    @property
    def name(self) -> str:
        return f"Classifier-{self._backbone_name}"

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images [B, C, H, W].

        Returns:
            Logits [B, num_classes].
        """
        features = self.backbone(x)
        return self.head(features)

    def get_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute cross-entropy loss."""
        return self._loss_fn(predictions, targets)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without head."""
        return self.backbone(x)

    def test_inference(
        self,
        images: Sequence[str | Path | Image.Image | np.ndarray],
        image_size: tuple[int, int] = (224, 224),
        device: str | torch.device | None = None,
        return_probabilities: bool = True,
    ) -> dict[str, Any]:
        """Test inference with images."""
        result = super().test_inference(images, image_size, device)

        logits = result["predictions"]
        predictions = np.argmax(logits, axis=1)

        result["logits"] = logits
        result["predictions"] = predictions

        if return_probabilities:
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            result["probabilities"] = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return result


@register_model(
    "multi_task_classifier",
    task="mtl_classification",
    description="Multi-task classifier with configurable backbone and task heads",
    aliases=["mtl_classifier", "mtl"],
)
class MultiTaskClassifier(BaseModel):
    """Generic multi-task classifier with configurable backbone and heads.

    Architecture:
        - Configurable backbone (ResNet, ConvNeXt, ViT, etc.)
        - Global Average Pooling -> feature_dim
        - Separate configurable heads per task

    Supports multiple task types:
        - multiclass: CrossEntropyLoss
        - binary: BCEWithLogitsLoss (single output)
        - multilabel: BCEWithLogitsLoss (multiple outputs)
        - regression: MSELoss
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        tasks: list[TaskConfig] | None = None,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
        default_head_config: HeadConfig | None = None,
    ) -> None:
        """Initialize MultiTaskClassifier.

        Args:
            backbone: Backbone name (see BackboneFactory for options).
            tasks: List of task configurations. If None, uses LUMBAR_SPINE_TASKS.
            pretrained: Use pretrained weights.
            dropout: Dropout rate (used if no head_config).
            freeze_backbone: Freeze backbone weights.
            default_head_config: Default head config for tasks without custom config.
        """
        super().__init__()

        self._backbone_name = backbone
        self._pretrained = pretrained
        self._dropout = dropout
        self._freeze_backbone = freeze_backbone

        # Use predefined tasks if none provided
        if tasks is None:
            tasks = [TaskConfig(**t.__dict__) for t in LUMBAR_SPINE_TASKS]
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
            if task.head_config is not None:
                head = create_head(task.head_config, feature_dim, task.num_classes)
            elif default_head_config is not None:
                head = create_head(default_head_config, feature_dim, task.num_classes)
            else:
                head = nn.Linear(feature_dim, task.num_classes)
            self.heads[task.name] = head

        # Initialize loss functions
        self._loss_functions: dict[str, nn.Module] = {}
        self._loss_weights: dict[str, float] = {}
        self._init_loss_functions(tasks)

        if freeze_backbone:
            self.freeze_backbone()

        self._is_initialized = True

    def _init_loss_functions(self, tasks: list[TaskConfig]) -> None:
        """Initialize loss functions for each task."""
        for task in tasks:
            self._loss_weights[task.name] = task.loss_weight

            if task.task_type == "multiclass":
                self._loss_functions[task.name] = nn.CrossEntropyLoss(
                    weight=task.class_weights,
                    label_smoothing=task.label_smoothing,
                )
            elif task.task_type in ("binary", "multilabel"):
                if task.class_weights is not None:
                    self._loss_functions[task.name] = nn.BCEWithLogitsLoss(
                        pos_weight=task.class_weights,
                    )
                else:
                    self._loss_functions[task.name] = nn.BCEWithLogitsLoss()
            elif task.task_type == "regression":
                self._loss_functions[task.name] = nn.MSELoss()
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

    @property
    def name(self) -> str:
        return f"MTLClassifier-{self._backbone_name}"

    @property
    def task_names(self) -> list[str]:
        return self._task_names

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
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
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        for task in self._tasks:
            if task.name not in predictions or task.name not in targets:
                continue

            pred = predictions[task.name]
            target = targets[task.name]
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
            losses[task.name] = self._loss_functions[task.name](
                predictions[task.name], targets[task.name]
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

    def predict(self, x: torch.Tensor) -> dict[str, np.ndarray]:
        """Run inference and return final predictions."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)

        predictions: dict[str, np.ndarray] = {}
        for task in self._tasks:
            logits = outputs[task.name]

            if task.task_type == "multiclass":
                pred = torch.argmax(logits, dim=1)
            elif task.task_type in ("binary", "multilabel"):
                pred = (torch.sigmoid(logits) > 0.5).int()
                if task.task_type == "binary" and pred.shape[-1] == 1:
                    pred = pred.squeeze(-1)
            else:
                pred = logits

            predictions[task.name] = pred.cpu().numpy()

        return predictions

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

        predictions = self.predict(batch)

        probabilities: dict[str, np.ndarray] = {}
        for task in self._tasks:
            logits = outputs[task.name]
            if task.task_type == "multiclass":
                probabilities[task.name] = F.softmax(logits, dim=1).cpu().numpy()
            else:
                probabilities[task.name] = torch.sigmoid(logits).cpu().numpy()

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "images": np.stack(processed_images),
            "inference_time_ms": inference_time_ms,
            "num_images": len(images),
            "device": str(device),
        }


@register_model(
    "coordinate_regressor",
    task="localization",
    description="Coordinate regressor for localization with configurable backbone",
    aliases=["localizer", "regressor"],
)
class CoordinateRegressor(BaseModel):
    """Generic coordinate regressor with configurable backbone and head.

    Architecture:
        - Configurable backbone (ResNet, ConvNeXt, ViT, etc.)
        - Global Average Pooling -> feature_dim
        - Optional level embedding for multi-level localization
        - Configurable regression head with sigmoid output

    Output: Normalized coordinates in [0, 1].
    """

    def __init__(
        self,
        backbone: str = "convnext_base",
        num_outputs: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        head_config: HeadConfig | None = None,
        num_levels: int = 1,
        level_embedding_dim: int = 16,
        loss_type: Literal["mse", "smooth_l1", "huber"] = "smooth_l1",
    ) -> None:
        """Initialize CoordinateRegressor.

        Args:
            backbone: Backbone name (see BackboneFactory for options).
            num_outputs: Number of output coordinates (default 2 for x,y).
            pretrained: Use pretrained weights.
            dropout: Dropout rate (used if no head_config).
            freeze_backbone: Freeze backbone weights.
            head_config: Custom head configuration.
            num_levels: Number of levels for level embedding (1 = no embedding).
            level_embedding_dim: Dimension of level embedding.
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

        # Level embedding (optional)
        self.level_embedding: nn.Embedding | None = None
        if num_levels > 1:
            self.level_embedding = nn.Embedding(num_levels, level_embedding_dim)
            head_input_dim = feature_dim + level_embedding_dim
        else:
            head_input_dim = feature_dim

        # Create head
        if head_config is not None:
            self.head = create_head(head_config, head_input_dim, num_outputs)
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(head_input_dim),
                nn.Dropout(dropout),
                nn.Linear(head_input_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout / 2),
                nn.Linear(256, num_outputs),
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

    def forward(
        self,
        x: torch.Tensor,
        level_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images [B, C, H, W].
            level_idx: Optional level indices [B] for level embedding.

        Returns:
            Predicted coordinates [B, num_outputs] in [0, 1].
        """
        features = self.backbone(x)

        if self.level_embedding is not None and level_idx is not None:
            level_emb = self.level_embedding(level_idx)
            features = torch.cat([features, level_emb], dim=1)

        return self.head(features)

    def get_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute regression loss."""
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
        level_indices: list[int] | None = None,
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

        level_idx_tensor: torch.Tensor | None = None
        if level_indices is not None and self.level_embedding is not None:
            level_idx_tensor = torch.tensor(level_indices, dtype=torch.long, device=device)

        self.eval()
        start_time = time.perf_counter()
        with torch.no_grad():
            predictions = self.forward(batch, level_idx_tensor)
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


# Expose backbone list for convenience
def list_backbones(family: str | None = None) -> list[str]:
    """List available backbone names."""
    return BackboneFactory.list_backbones(family)

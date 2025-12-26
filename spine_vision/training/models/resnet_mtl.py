"""ResNet50-based Multi-Task Learning model for lumbar spine classification.

Implements a multi-head classification model for predicting 13 clinical labels:
- Pfirrmann grade (5 classes)
- Modic type (4 classes)
- Herniation (2 binary: herniation, bulging)
- Endplate (2 binary: upper, lower)
- Spondylolisthesis (1 binary)
- Narrowing (1 binary)

Uses ResNet-50 backbone with separate classification heads for each task.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image

from spine_vision.training.base import BaseModel


@dataclass
class MTLPredictions:
    """Container for multi-task predictions."""
    
    pfirrmann: torch.Tensor  # [B, 5] logits
    modic: torch.Tensor  # [B, 4] logits
    herniation: torch.Tensor  # [B, 2] logits
    endplate: torch.Tensor  # [B, 2] logits
    spondy: torch.Tensor  # [B, 1] logits
    narrowing: torch.Tensor  # [B, 1] logits


@dataclass
class MTLTargets:
    """Container for multi-task targets."""
    
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


class ResNet50MTL(BaseModel):
    """ResNet-50 Multi-Task Learning model for lumbar spine classification.
    
    Architecture:
        - ResNet-50 backbone (pretrained on ImageNet)
        - Global Average Pooling -> 2048-dim features
        - 6 separate classification heads:
            * head_pfirrmann: 5 classes (Pfirrmann grades 1-5)
            * head_modic: 4 classes (Modic types 0-3)
            * head_herniation: 2 outputs (herniation, bulging)
            * head_endplate: 2 outputs (upper, lower)
            * head_spondy: 1 output (yes/no)
            * head_narrowing: 1 output (yes/no)
    
    Loss:
        - CrossEntropyLoss for Pfirrmann and Modic (multiclass)
        - BCEWithLogitsLoss for all other heads (binary/multi-label)
    
    Input:
        3-channel tensor [B, 3, H, W]:
            - Channel R: T2 crop (normalized)
            - Channel G: T1 crop (normalized)
            - Channel B: T2 crop duplicate
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
        label_smoothing: float = 0.1,
        class_weights: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Initialize ResNet50MTL.
        
        Args:
            pretrained: Use ImageNet pretrained weights.
            dropout: Dropout rate before classification heads.
            freeze_backbone: Freeze backbone weights initially.
            label_smoothing: Label smoothing for cross-entropy losses.
            class_weights: Optional class weights for imbalanced data.
                Keys: pfirrmann, modic, herniation, endplate, spondy, narrowing.
                For multiclass (pfirrmann, modic): weight tensor for CrossEntropyLoss.
                For binary tasks: pos_weight tensor for BCEWithLogitsLoss.
        """
        super().__init__()
        
        self._pretrained = pretrained
        self._dropout_rate = dropout
        self._freeze_backbone = freeze_backbone
        self._label_smoothing = label_smoothing
        self._class_weights = class_weights
        
        # Load pretrained ResNet-50 via timm (no classification head)
        self.backbone = timm.create_model(
            "resnet50.a1_in1k",
            pretrained=pretrained,
            num_classes=0,  # Remove classification head, get features
        )
        
        # Feature dimension from ResNet-50 is 2048
        feature_dim: int = self.backbone.num_features  # type: ignore[assignment]
        assert feature_dim == 2048, f"Expected 2048 features, got {feature_dim}"
        
        # Shared dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification heads
        self.head_pfirrmann = nn.Linear(feature_dim, 5)  # Grades 1-5
        self.head_modic = nn.Linear(feature_dim, 4)  # Types 0-3
        self.head_herniation = nn.Linear(feature_dim, 2)  # Herniation, Bulging
        self.head_endplate = nn.Linear(feature_dim, 2)  # Upper, Lower
        self.head_spondy = nn.Linear(feature_dim, 1)  # Yes/No
        self.head_narrowing = nn.Linear(feature_dim, 1)  # Yes/No
        
        # Loss functions with optional class weights
        self._init_loss_functions(label_smoothing, class_weights)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()
        
        self._is_initialized = True

    def _init_loss_functions(
        self,
        label_smoothing: float,
        class_weights: dict[str, torch.Tensor] | None,
    ) -> None:
        """Initialize loss functions with optional class weights."""
        if class_weights is None:
            # Standard losses without weighting
            self._ce_pfirrmann = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self._ce_modic = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self._bce_herniation = nn.BCEWithLogitsLoss()
            self._bce_endplate = nn.BCEWithLogitsLoss()
            self._bce_spondy = nn.BCEWithLogitsLoss()
            self._bce_narrowing = nn.BCEWithLogitsLoss()
        else:
            # Weighted losses for imbalanced data
            self._ce_pfirrmann = nn.CrossEntropyLoss(
                weight=class_weights.get("pfirrmann"),
                label_smoothing=label_smoothing,
            )
            self._ce_modic = nn.CrossEntropyLoss(
                weight=class_weights.get("modic"),
                label_smoothing=label_smoothing,
            )
            self._bce_herniation = nn.BCEWithLogitsLoss(
                pos_weight=class_weights.get("herniation"),
            )
            self._bce_endplate = nn.BCEWithLogitsLoss(
                pos_weight=class_weights.get("endplate"),
            )
            self._bce_spondy = nn.BCEWithLogitsLoss(
                pos_weight=class_weights.get("spondy"),
            )
            self._bce_narrowing = nn.BCEWithLogitsLoss(
                pos_weight=class_weights.get("narrowing"),
            )
    
    @property
    def name(self) -> str:
        return "ResNet50-MTL-Classification"
    
    def forward(self, x: torch.Tensor) -> MTLPredictions:
        """Forward pass.
        
        Args:
            x: Input images [B, 3, H, W].
               Expected channels: [T2, T1, T2] with ImageNet normalization.
        
        Returns:
            MTLPredictions with logits for each head.
        """
        # Extract features via backbone (includes global avg pool)
        features = self.backbone(x)  # [B, 2048]
        
        # Apply dropout
        features = self.dropout(features)
        
        # Pass through each head
        return MTLPredictions(
            pfirrmann=self.head_pfirrmann(features),
            modic=self.head_modic(features),
            herniation=self.head_herniation(features),
            endplate=self.head_endplate(features),
            spondy=self.head_spondy(features),
            narrowing=self.head_narrowing(features),
        )
    
    def get_loss(
        self,
        predictions: MTLPredictions | torch.Tensor,
        targets: MTLTargets | torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute multi-task loss.
        
        Total loss = L_pfirrmann + L_modic + L_herniation + L_endplate + L_spondy + L_narrowing
        
        Args:
            predictions: MTLPredictions from forward pass.
            targets: MTLTargets with ground truth labels.
            **kwargs: Additional arguments (unused).
        
        Returns:
            Total loss (scalar tensor).
        """
        if isinstance(predictions, torch.Tensor):
            raise TypeError(
                "Expected MTLPredictions, got Tensor. Use forward() to get predictions."
            )
        if isinstance(targets, torch.Tensor):
            raise TypeError(
                "Expected MTLTargets, got Tensor. Create MTLTargets from batch."
            )
        
        # CrossEntropy for multiclass
        loss_pfirrmann = self._ce_pfirrmann(predictions.pfirrmann, targets.pfirrmann)
        loss_modic = self._ce_modic(predictions.modic, targets.modic)
        
        # BCEWithLogits for binary/multi-label
        loss_herniation = self._bce_herniation(predictions.herniation, targets.herniation)
        loss_endplate = self._bce_endplate(predictions.endplate, targets.endplate)
        loss_spondy = self._bce_spondy(predictions.spondy, targets.spondy)
        loss_narrowing = self._bce_narrowing(predictions.narrowing, targets.narrowing)
        
        # Sum all losses (equal weighting)
        total_loss = (
            loss_pfirrmann
            + loss_modic
            + loss_herniation
            + loss_endplate
            + loss_spondy
            + loss_narrowing
        )
        
        return total_loss
    
    def get_loss_breakdown(
        self,
        predictions: MTLPredictions,
        targets: MTLTargets,
    ) -> dict[str, torch.Tensor]:
        """Get individual loss values for each task.
        
        Useful for logging and debugging.
        """
        return {
            "pfirrmann": self._ce_pfirrmann(predictions.pfirrmann, targets.pfirrmann),
            "modic": self._ce_modic(predictions.modic, targets.modic),
            "herniation": self._bce_herniation(predictions.herniation, targets.herniation),
            "endplate": self._bce_endplate(predictions.endplate, targets.endplate),
            "spondy": self._bce_spondy(predictions.spondy, targets.spondy),
            "narrowing": self._bce_narrowing(predictions.narrowing, targets.narrowing),
        }
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def predict(self, x: torch.Tensor) -> dict[str, np.ndarray]:
        """Run inference and return final predictions.
        
        Args:
            x: Input images [B, 3, H, W].
        
        Returns:
            Dictionary with predicted labels:
                - pfirrmann: [B] int, values 1-5 (grade)
                - modic: [B] int, values 0-3 (type)
                - herniation: [B, 2] int, 0 or 1
                - endplate: [B, 2] int, 0 or 1
                - spondy: [B] int, 0 or 1
                - narrowing: [B] int, 0 or 1
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
        
        # Convert logits to predictions
        pfirrmann = torch.argmax(outputs.pfirrmann, dim=1) + 1  # 0-4 -> 1-5
        modic = torch.argmax(outputs.modic, dim=1)  # 0-3
        herniation = (torch.sigmoid(outputs.herniation) > 0.5).int()
        endplate = (torch.sigmoid(outputs.endplate) > 0.5).int()
        spondy = (torch.sigmoid(outputs.spondy) > 0.5).int().squeeze(-1)
        narrowing = (torch.sigmoid(outputs.narrowing) > 0.5).int().squeeze(-1)
        
        return {
            "pfirrmann": pfirrmann.cpu().numpy(),
            "modic": modic.cpu().numpy(),
            "herniation": herniation.cpu().numpy(),
            "endplate": endplate.cpu().numpy(),
            "spondy": spondy.cpu().numpy(),
            "narrowing": narrowing.cpu().numpy(),
        }
    
    def to_csv_row(
        self,
        predictions: dict[str, np.ndarray],
        patient_id: str,
        ivd_level: int,  # 1-5 for L1/L2 to L5/S1
        row_idx: int = 0,
    ) -> list[Any]:
        """Convert predictions to CSV row format.
        
        Output columns (13 values):
            PatientID, IVD, Pfirrmann, Modic_0, Modic_1, Modic_2, Modic_3,
            Herniation, Bulging, Upper_Endplate, Lower_Endplate,
            Spondylolisthesis, Narrowing
        
        Args:
            predictions: Output from predict() method.
            patient_id: Patient identifier.
            ivd_level: IVD level (1-5, where 1=L1/L2, ..., 5=L5/S1).
            row_idx: Index into batch for multi-sample predictions.
        
        Returns:
            List of values for CSV row.
        """
        # Convert Modic prediction to one-hot
        modic_pred = predictions["modic"][row_idx]
        modic_one_hot = [0, 0, 0, 0]
        modic_one_hot[modic_pred] = 1
        
        return [
            patient_id,
            ivd_level,
            predictions["pfirrmann"][row_idx],
            modic_one_hot[0],
            modic_one_hot[1],
            modic_one_hot[2],
            modic_one_hot[3],
            predictions["herniation"][row_idx, 0],
            predictions["herniation"][row_idx, 1],
            predictions["endplate"][row_idx, 0],
            predictions["endplate"][row_idx, 1],
            predictions["spondy"][row_idx],
            predictions["narrowing"][row_idx],
        ]
    
    @staticmethod
    def get_csv_header() -> list[str]:
        """Get CSV header for prediction output."""
        return [
            "PatientID",
            "IVD",
            "Pfirrmann",
            "Modic_0",
            "Modic_1",
            "Modic_2",
            "Modic_3",
            "Herniation",
            "Bulging",
            "Upper_Endplate",
            "Lower_Endplate",
            "Spondylolisthesis",
            "Narrowing",
        ]
    
    def test_inference(
        self,
        images: Sequence[str | Path | Image.Image | np.ndarray],
        image_size: tuple[int, int] = (224, 224),
        device: str | torch.device | None = None,
    ) -> dict[str, Any]:
        """Test inference with a list of images.
        
        Args:
            images: List of images (file paths, PIL Images, or numpy arrays).
            image_size: Target size for resizing (H, W).
            device: Device for inference. If None, uses model's current device.
        
        Returns:
            Dictionary containing:
                - predictions: Dict with predicted labels per task
                - probabilities: Dict with probabilities per task
                - images: Preprocessed images as numpy array
                - inference_time_ms: Total inference time
        """
        import time
        from torchvision import transforms
        
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)
        
        # Build preprocessing transform (ImageNet normalization)
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
            
            tensor = transform(pil_img)
            processed_tensors.append(tensor)
        
        # Create batch
        batch = torch.stack(processed_tensors).to(device)
        
        # Run inference
        self.eval()
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.forward(batch)
        end_time = time.perf_counter()
        
        inference_time_ms = (end_time - start_time) * 1000
        
        # Get predictions
        predictions = self.predict(batch.to(device))
        
        # Get probabilities
        probabilities = {
            "pfirrmann": F.softmax(outputs.pfirrmann, dim=1).cpu().numpy(),
            "modic": F.softmax(outputs.modic, dim=1).cpu().numpy(),
            "herniation": torch.sigmoid(outputs.herniation).cpu().numpy(),
            "endplate": torch.sigmoid(outputs.endplate).cpu().numpy(),
            "spondy": torch.sigmoid(outputs.spondy).cpu().numpy(),
            "narrowing": torch.sigmoid(outputs.narrowing).cpu().numpy(),
        }
        
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "images": np.stack(processed_images),
            "inference_time_ms": inference_time_ms,
            "num_images": len(images),
            "device": str(device),
        }

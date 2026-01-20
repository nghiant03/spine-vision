"""Model architectures for training.

Provides composable model architectures with configurable backbones and heads.
Uses timm (PyTorch Image Models) for pretrained backbones.

## Models

- `Classifier`: Single-task and multi-task classification with configurable heads
- `CoordinateRegressor`: Coordinate regression for localization

## Backbone Options

Use `list_backbones()` to see all available backbones:
- ResNet family: resnet18, resnet50, resnet101, resnext50, wide_resnet50, etc.
- ConvNeXt: convnext_tiny, convnext_base, convnextv2_base, etc.
- Vision Transformers: vit_tiny, vit_base, swin_tiny, deit_base, etc.
- EfficientNet: efficientnet_b0-b4, efficientnetv2_s/m/l
- MobileNet: mobilenetv3_small, mobilenetv3_large

## Usage

```python
from spine_vision.training.models import (
    Classifier,
    CoordinateRegressor,
    TaskConfig,
    list_backbones,
)

# Single-task classifier
model = Classifier(
    backbone="resnet50",
    tasks=[TaskConfig(name="grade", num_classes=5)],
)
output = model(images)  # {"grade": logits}

# Multi-task classifier
model = Classifier(
    backbone="convnext_base",
    tasks=[
        TaskConfig(name="grade", num_classes=5, task_type="multiclass"),
        TaskConfig(name="herniation", num_classes=1, task_type="binary"),
    ],
)
output = model(images)  # {"grade": logits, "herniation": logits}

# Coordinate regressor for localization
model = CoordinateRegressor(
    backbone="convnext_base",
    num_outputs=2,
    num_levels=5,  # For IVD level embedding
)

# List available backbones
print(list_backbones())  # All backbones
print(list_backbones("resnet"))  # ResNet family only
```
"""

from spine_vision.training.models.backbone import (
    BACKBONES,
    BackboneFactory,
    BackboneName,
)
from spine_vision.training.models.generic import (
    Classifier,
    CoordinateRegressor,
    LUMBAR_SPINE_TASKS,
    MTLTargets,
    TaskConfig,
    list_backbones,
)

# Backward compatibility alias
MultiTaskClassifier = Classifier

__all__ = [
    # Models
    "Classifier",
    "MultiTaskClassifier",  # Backward compatibility alias
    "CoordinateRegressor",
    # Task configuration
    "TaskConfig",
    "LUMBAR_SPINE_TASKS",
    "MTLTargets",
    # Backbone utilities
    "BackboneFactory",
    "BACKBONES",
    "BackboneName",
    "list_backbones",
]

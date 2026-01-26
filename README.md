# Spine Vision

Medical imaging pipeline for lumbar spine MRI analysis and radiological grading prediction.

## Features

- **Dataset Creation** - Build localization and classification datasets from RSNA, SPIDER, and Phenikaa sources
- **OCR Extraction** - Extract patient information from Vietnamese medical reports using PaddleOCR + VietOCR
- **Fuzzy Matching** - Match patient data across different data sources with configurable thresholds
- **IVD Localization** - Train ConvNext-based models to detect intervertebral disc coordinates
- **Multi-task Classification** - Train models for Pfirrmann grading, Modic changes, herniation detection, and more
- **Experiment Tracking** - Optional trackio integration for training visualization

## Installation

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv) package manager.

```bash
# Clone the repository
git clone https://github.com/yourusername/spine-vision.git
cd spine-vision

# Install dependencies
uv sync

# Include dev dependencies (type stubs)
uv sync --group dev

# Include visualization dependencies (seaborn, trackio)
uv sync --group visualization
```

## Quick Start

### CLI Commands

```bash
# Create localization dataset from RSNA + Lumbar Coords
spine-vision dataset localization --base-path data

# Preprocess Phenikaa dataset (OCR + patient matching)
spine-vision dataset phenikaa --data-path data/raw/Phenikaa

# Create classification dataset (Phenikaa + SPIDER with IVD cropping)
spine-vision dataset classification --base-path data --localization-model-path weights/localization/model.pt

# Train localization model
spine-vision train localization --data-path data/processed/localization --backbone convnext_base

# Train classification model
spine-vision train classification --data-path data/processed/classification --backbone resnet18

# Test trained models
spine-vision test --image-path test_image.png --model-path weights/model.pt
```

### Python API

```python
from pathlib import Path
from spine_vision.io import read_medical_image, write_medical_image, normalize_to_uint8
from spine_vision.datasets import (
    LocalizationDatasetConfig,
    create_localization_dataset,
    ClassificationDatasetConfig,
    create_classification_dataset,
)
from spine_vision.training import (
    LocalizationConfig,
    LocalizationTrainer,
    ClassificationConfig,
    ClassificationTrainer,
)

# Load medical image (auto-detects DICOM, NIfTI, MHA, NRRD)
image = read_medical_image(Path("input.nii.gz"))
write_medical_image(image, Path("output.nii.gz"))

# Create localization dataset
config = LocalizationDatasetConfig(base_path=Path("data"))
result = create_localization_dataset(config)
print(f"Created {result.num_samples} samples")

# Train localization model
config = LocalizationConfig(
    data_path=Path("data/processed/localization"),
    backbone="convnext_base",
    batch_size=32,
    num_epochs=100,
)
trainer = LocalizationTrainer(config)
result = trainer.train()

# Train classification model
config = ClassificationConfig(
    data_path=Path("data/processed/classification"),
    backbone="resnet18",
    output_size=(256, 256),
    dropout=0.3,
)
trainer = ClassificationTrainer(config)
result = trainer.train()
```

## CLI Reference

### `spine-vision dataset localization`

Create localization dataset from RSNA + Lumbar Coords.

| Option | Description | Default |
|--------|-------------|---------|
| `--base-path` | Base data directory | `data` |
| `-v, --verbose` | Debug logging | `False` |

### `spine-vision dataset phenikaa`

Preprocess Phenikaa dataset with OCR and patient matching.

| Option | Description | Default |
|--------|-------------|---------|
| `--data-path` | Input data directory | `data/raw/Phenikaa` |
| `--output-path` | Output directory | `data/interim/Phenikaa` |
| `-g, --use-gpu` | Enable GPU acceleration | `True` |
| `--report-fuzzy-threshold` | OCR matching threshold | `80` |
| `--image-fuzzy-threshold` | Folder matching threshold | `85` |
| `--pdf-dpi` | DPI for PDF rendering | `200` |
| `-v, --verbose` | Debug logging | `False` |

### `spine-vision dataset classification`

Create classification dataset with IVD cropping.

| Option | Description | Default |
|--------|-------------|---------|
| `--base-path` | Base data directory | `data` |
| `--localization-model-path` | Path to trained localization model | None |
| `--crop-size` | Output size of cropped IVD regions (H W) | `128 128` |
| `--crop-delta-mm` | Crop deltas (left right top bottom) in mm | `50.0 20.0 30.0 30.0` |
| `--include-phenikaa` | Include Phenikaa dataset | `True` |
| `--include-spider` | Include SPIDER dataset | `True` |
| `-v, --verbose` | Debug logging | `False` |

### `spine-vision train localization`

Train IVD localization model.

| Option | Description | Default |
|--------|-------------|---------|
| `--data-path` | Localization dataset path | `data/processed/localization` |
| `--backbone` | Backbone model name | `convnext_base` |
| `--image-size` | Input image size | `(512, 512)` |
| `--num-levels` | Number of IVD levels to predict | `5` |
| `--loss-type` | Loss function: `mse`, `smooth_l1`, `huber` | `mse` |
| `--freeze-backbone-epochs` | Epochs to freeze backbone | `0` |
| `--batch-size` | Training batch size | `32` |
| `--num-epochs` | Number of training epochs | `100` |
| `--learning-rate` | Learning rate | `1e-4` |
| `--use-trackio` | Enable trackio logging | `False` |
| `-v, --verbose` | Debug logging | `False` |

### `spine-vision train classification`

Train multi-task classification model.

| Option | Description | Default |
|--------|-------------|---------|
| `--data-path` | Classification dataset path | `data/processed/classification` |
| `--backbone` | Backbone model name | `resnet18` |
| `--output-size` | Final input size to model (H W) | `(256, 256)` |
| `--target-labels` | Filter to specific labels | None (all) |
| `--series-types` | Filter to T1, T2, or both | None (all) |
| `--use-weighted-sampling` | Enable weighted sampling | `True` |
| `--use-focal-loss` | Use focal loss | `False` |
| `--focal-gamma` | Focal loss gamma | `2.0` |
| `--dropout` | Dropout rate | `0.3` |
| `--use-trackio` | Enable trackio logging | `False` |
| `-v, --verbose` | Debug logging | `False` |

### Common Training Options

These options apply to all training commands:

| Option | Description | Default |
|--------|-------------|---------|
| `--batch-size` | Training batch size | `32` |
| `--num-epochs` | Number of training epochs | `15` |
| `--learning-rate` | Learning rate | `1e-4` |
| `--weight-decay` | Weight decay | `1e-5` |
| `--scheduler-type` | LR scheduler: `cosine`, `step`, `plateau`, `none` | `cosine` |
| `--early-stopping` | Enable early stopping | `True` |
| `--patience` | Early stopping patience | `20` |
| `--mixed-precision` | Enable mixed precision training | `True` |

## Supported Classification Tasks

The multi-task classification model supports:

| Task | Type | Classes | Description |
|------|------|---------|-------------|
| `pfirrmann` | Multiclass | 5 | Pfirrmann disc degeneration grade (I-V) |
| `modic` | Multiclass | 4 | Modic endplate changes (0-3) |
| `herniation` | Binary | 2 | Disc herniation presence |
| `upper_endplate` | Binary | 2 | Upper endplate defect |
| `lower_endplate` | Binary | 2 | Lower endplate defect |
| `spondylolisthesis` | Binary | 2 | Vertebral slippage |
| `spinal_stenosis` | Multiclass | 4 | Spinal canal narrowing severity |
| `foraminal_stenosis` | Multiclass | 4 | Neural foraminal narrowing severity |

## Project Structure

```
spine-vision/
├── spine_vision/
│   ├── core/              # Config, logging, task system
│   ├── io/                # Medical image I/O, tabular data
│   ├── datasets/          # Dataset creation pipelines
│   │   ├── phenikaa/      # OCR + patient matching
│   │   └── classification/ # IVD cropping pipeline
│   ├── training/          # Training infrastructure
│   │   ├── models/        # Model architectures
│   │   ├── trainers/      # Task-specific trainers
│   │   └── datasets/      # PyTorch datasets
│   ├── visualization/     # Plotly-based visualization
│   └── cli/               # CLI entry points
├── notebooks/             # Jupyter notebooks
├── data/                  # Data directories (gitignored)
├── weights/               # Model weights (gitignored)
└── pyproject.toml
```

## Requirements

- Python 3.11+
- NumPy < 2 (compatibility with medical imaging libraries)
- CUDA-capable GPU recommended for OCR and training

## Development

### Type Checking

```bash
uv run pyright spine_vision
```

### Linting

```bash
uv run ruff check --fix spine_vision
```

## License

MIT

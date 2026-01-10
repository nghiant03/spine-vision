# AGENTS.md - Spine Vision

## Project Overview

Library for lumbar spine MRI dataset creation, model training, and result visualization. The project handles:
- Dataset creation pipelines (nnUNet format, IVD coordinates, classification datasets)
- OCR-based extraction of patient information from medical reports (for Phenikaa dataset)
- Fuzzy matching of patient data across different data sources
- Model training infrastructure with trackio integration
- Interactive visualization of training results and predictions

## Tech Stack

- **Python**: 3.11+
- **Package Manager**: uv (uses `pyproject.toml` and `uv.lock`)
- **Core Libraries**:
  - `paddleocr` / `vietocr` - Vietnamese OCR for medical report extraction
  - `SimpleITK` - Medical image I/O and processing
  - `opencv-python` - Image manipulation and perspective transforms
  - `pandas` / `openpyxl` - Tabular data handling
  - `rapidfuzz` - Fuzzy string matching
  - `pydantic` - Configuration validation
  - `tyro` - CLI argument parsing
  - `loguru` - Logging
  - `torch` / `torchvision` / `timm` - Deep learning backend
  - `plotly` / `kaleido` - Interactive visualization and image export
  - `pyyaml` - Label schema configuration
  - `rich` / `tqdm` - Progress bars and terminal output
  - `accelerate` - Distributed training and mixed precision

## Commands

### Setup
```bash
uv sync                    # Install dependencies
uv sync --group dev        # Include dev dependencies (pandas-stubs, plotly-stubs)
```

### CLI Entry Points (after install)
```bash
spine-vision dataset nnunet [OPTIONS]          # Convert datasets to nnU-Net format
spine-vision dataset localization [OPTIONS]    # Create localization dataset
spine-vision dataset phenikaa [OPTIONS]        # Preprocess Phenikaa dataset (OCR + matching)
spine-vision dataset classification [OPTIONS]  # Create classification dataset (Phenikaa + SPIDER)
spine-vision train localization [OPTIONS]      # Train localization model (ConvNext)
spine-vision train classification [OPTIONS]    # Train classification model (ResNet50-MTL)
spine-vision test [OPTIONS]                    # Test trained models with images/DICOM
spine-vision evaluate [OPTIONS]                # Evaluate on test set with visualization
spine-vision analyze [OPTIONS]                 # Analyze classification dataset
```

### Running Scripts Directly
```bash
uv run python -m spine_vision.cli [OPTIONS]
```

### Type Checking (REQUIRED)
```bash
uv run pyright spine_vision
```

### Linting (REQUIRED)
```bash
uv run ruff check --fix spine_vision
```

> **IMPORTANT**: Always run the type checker and linter after making code changes. This is mandatory before completing any task.

## Project Structure

```
spine-vision/
├── spine_vision/              # Main package
│   ├── __init__.py
│   ├── core/                  # Core utilities
│   │   ├── __init__.py        # BaseConfig, setup_logger, add_file_log
│   │   └── logging.py         # Centralized logging setup
│   ├── io/                    # I/O utilities
│   │   ├── __init__.py        # normalize_to_uint8 + re-exports
│   │   ├── pdf.py             # PDF to image conversion
│   │   ├── readers.py         # DICOM, NIfTI, MHA, NRRD readers
│   │   ├── writers.py         # Format conversion & export
│   │   └── tabular.py         # Excel/CSV handling
│   ├── datasets/              # Dataset creation pipelines
│   │   ├── __init__.py
│   │   ├── labels.py          # LabelSchema, remap_labels, load_label_schema
│   │   ├── nnunet.py          # nnUNet format conversion
│   │   ├── ivd_coords.py      # IVD coordinates dataset creation
│   │   ├── classification.py  # Classification dataset (Phenikaa + SPIDER)
│   │   ├── rsna.py            # RSNA dataset utilities (series mapping)
│   │   ├── phenikaa/          # Phenikaa dataset preprocessing
│   │   │   ├── __init__.py    # PreprocessConfig, main, report processors
│   │   │   ├── ocr.py         # DocumentExtractor, TextDetector, TextRecognizer
│   │   │   └── matching.py    # PatientMatcher, fuzzy_value_extract
│   │   └── schemas/           # YAML label definitions
│   │       └── spider.yaml    # SPIDER dataset labels
│   ├── visualization/         # Visualization
│   │   ├── __init__.py
│   │   └── plotly_viewer.py   # Interactive batch-capable viewer
│   ├── training/              # Training infrastructure
│   │   ├── __init__.py
│   │   ├── base.py            # BaseTrainer, BaseModel, TrainingConfig
│   │   ├── heads.py           # Configurable head architectures
│   │   ├── metrics.py         # BaseMetrics, LocalizationMetrics, MTLClassificationMetrics
│   │   ├── registry.py        # ModelRegistry, TrainerRegistry
│   │   ├── visualization.py   # TrainingVisualizer for curves/predictions
│   │   ├── datasets/          # PyTorch datasets for training
│   │   │   ├── __init__.py
│   │   │   ├── ivd_coords.py  # IVDCoordsDataset
│   │   │   └── classification.py # ClassificationDataset, ClassificationCollator
│   │   ├── models/            # Model architectures
│   │   │   ├── __init__.py
│   │   │   ├── backbone.py    # Backbone utilities
│   │   │   └── generic.py     # CoordinateRegressor, MTLClassifier
│   │   └── trainers/          # Task-specific trainers
│   │       ├── __init__.py
│   │       ├── localization.py # LocalizationTrainer, LocalizationConfig
│   │       └── classification.py # ClassificationTrainer, ClassificationConfig
│   └── cli/                   # Unified CLI with subcommands
│       ├── __init__.py        # Main entry point (dataset, train, test, evaluate, analyze)
│       ├── analyze.py         # Dataset analysis command
│       ├── evaluate.py        # Model evaluation command
│       ├── test.py            # Model testing command
│       └── train.py           # Training command entry point
├── data/                      # (gitignored) Data directories
├── weights/                   # (gitignored) Model weights
└── pyproject.toml
```

## Module API

### Core (`spine_vision.core`)
```python
from spine_vision.core import setup_logger, add_file_log, BaseConfig

setup_logger(verbose=True)
add_file_log(log_path=Path("logs"))
```

### I/O (`spine_vision.io`)
```python
from spine_vision.io import (
    read_medical_image, write_medical_image, load_tabular_data,
    normalize_to_uint8, write_records_csv, read_dicom_series,
    pdf_to_images, pdf_first_page_to_array
)

# Auto-detect format (DICOM dir, NIfTI, MHA, NRRD)
image = read_medical_image(Path("input.nii.gz"))
write_medical_image(image, Path("output.nii.gz"))

# Load and preprocess tabular data
df = load_tabular_data(
    table_path=Path("tables/"),
    one_hot_col="Modic",
    corrupted_ids=[25001],
)

# Normalize array to uint8
arr_uint8 = normalize_to_uint8(float_array)
```

### Datasets (`spine_vision.datasets`)
```python
from spine_vision.datasets import (
    # Base classes
    BaseProcessor, DatasetConfig, ProcessingResult,
    # IVD coordinates
    IVDDatasetConfig, IVDCoordsDatasetProcessor,
    # Phenikaa preprocessing
    PreprocessConfig, PhenikkaaProcessor,
    # Classification dataset
    ClassificationDatasetConfig, ClassificationDatasetProcessor,
    # RSNA utilities
    load_series_mapping, get_series_type,
)

# IVD coordinates dataset
config = IVDDatasetConfig(base_path=Path("data"))
processor = IVDCoordsDatasetProcessor(config)
result = processor.process()
print(f"Created {result.num_samples} IVD annotations at {result.output_path}")

# Phenikaa preprocessing (supports both report formats)
# - ID-named reports (250010139.png): extracts name/birthday from OCR
# - Patient-named reports (NGUYEN_VAN_SON_20250718.pdf): extracts ID from OCR
config = PreprocessConfig(data_path=Path("data/raw/Phenikaa"))
processor = PhenikkaaProcessor(config)
result = processor.process()
print(result.summary)  # "Matched X of Y patients"

# Classification dataset (Phenikaa + SPIDER with IVD cropping)
config = ClassificationDatasetConfig(
    base_path=Path("data"),
    localization_model_path=Path("weights/localization/model.pt"),
    crop_size=(64, 64),
    crop_delta_mm=(50.0, 20.0, 30.0, 30.0),  # left, right, top, bottom in mm
)
processor = ClassificationDatasetProcessor(config)
result = processor.process()
print(f"Created {result.num_samples} classification samples")
```

### Phenikaa OCR/Matching (`spine_vision.datasets.phenikaa`)
```python
from spine_vision.datasets.phenikaa.ocr import DocumentExtractor
from spine_vision.datasets.phenikaa.matching import PatientMatcher, fuzzy_value_extract

# Extract text from medical reports (images or PDFs)
extractor = DocumentExtractor(device="cuda:0", pdf_dpi=200)
text_lines = extractor.extract(Path("report.pdf"))

# Extract field value from OCR text
name = fuzzy_value_extract(text_lines, "Ho ten nguoi benh", threshold=80)

# Match patients to image folders
matcher = PatientMatcher(image_path=Path("images/"), threshold=85)
folder = matcher.match(patient_name, patient_birthday)
```

### Visualization (`spine_vision.visualization`)
```python
from spine_vision.visualization import PlotlyViewer

viewer = PlotlyViewer(output_path=Path("results/"), output_mode="html")
viewer.visualize(image, mask, title="Segmentation")
viewer.visualize_batch(images, masks)
```

### Training (`spine_vision.training`)
```python
from spine_vision.training import (
    # Base classes
    BaseModel, BaseTrainer, TrainingConfig, TrainingResult,
    # Datasets
    IVDCoordsDataset, ClassificationDataset, ClassificationCollator,
    # Models
    CoordinateRegressor, MTLClassifier,
    # Trainers
    LocalizationConfig, LocalizationTrainer,
    ClassificationConfig, ClassificationTrainer,
    # Metrics & Visualization
    BaseMetrics, LocalizationMetrics, MTLClassificationMetrics,
    TrainingVisualizer,
)

# Training localization model
config = LocalizationConfig(
    data_path=Path("data/processed/ivd_coords"),
    model_variant="base",
    batch_size=32,
    num_epochs=100,
    use_trackio=True,
)
trainer = LocalizationTrainer(config)
result = trainer.train()

# Training MTL classification model
config = ClassificationConfig(
    data_path=Path("data/processed/classification"),
    output_size=(224, 224),
    dropout=0.3,
    use_trackio=True,
)
trainer = ClassificationTrainer(config)
result = trainer.train()

# Evaluate on test set
metrics = trainer.evaluate(visualize=True)
```

## CLI Options

### spine-vision dataset phenikaa
| Flag | Description | Default |
|------|-------------|---------|
| `--data-path` | Input data directory | `data/raw/Phenikaa` |
| `--output-path` | Output directory | `data/interim/Phenikaa` |
| `-g, --use-gpu` | Enable GPU acceleration | `True` |
| `-v, --verbose` | Debug logging | `False` |
| `--report-fuzzy-threshold` | OCR matching threshold | `80` |
| `--image-fuzzy-threshold` | Folder matching threshold | `85` |
| `--pdf-dpi` | DPI for PDF rendering | `200` |

### spine-vision dataset nnunet
| Flag | Description | Default |
|------|-------------|---------|
| `--input-path` | Source dataset directory | `data/raw/SPIDER` |
| `--output-path` | nnUNet output directory | `data/processed/SPIDER/Dataset501_Spider` |
| `--schema-path` | Label schema YAML (optional) | Built-in `spider` |
| `--channel-name` | Channel name in dataset.json | `MRI` |
| `-v, --verbose` | Debug logging | `False` |

### spine-vision dataset classification
| Flag | Description | Default |
|------|-------------|---------|
| `--base-path` | Base data directory | `data` |
| `--localization-model-path` | Path to trained localization model | None |
| `--crop-size` | Output size of cropped IVD regions (H W) | `128 128` |
| `--crop-delta-mm` | Crop deltas (left right top bottom) in mm | `50.0 20.0 30.0 30.0` |
| `--include-phenikaa` | Include Phenikaa dataset | `True` |
| `--include-spider` | Include SPIDER dataset | `True` |
| `-v, --verbose` | Debug logging | `False` |

### spine-vision train localization
| Flag | Description | Default |
|------|-------------|---------|
| `--data-path` | IVD coordinates dataset path | `data/processed/ivd_coords` |
| `--model-variant` | ConvNext variant | `base` |
| `--batch-size` | Training batch size | `32` |
| `--num-epochs` | Number of training epochs | `100` |
| `--learning-rate` | Learning rate | `1e-4` |
| `--use-trackio` | Enable trackio logging | `False` |
| `-v, --verbose` | Debug logging | `False` |

### spine-vision train classification
| Flag | Description | Default |
|------|-------------|---------|
| `--data-path` | Classification dataset path | `data/processed/classification` |
| `--output-size` | Final input size to model (H W) | `224 224` |
| `--target-labels` | Filter to specific labels | None (all) |
| `--dropout` | Dropout rate | `0.3` |
| `--use-trackio` | Enable trackio logging | `False` |
| `-v, --verbose` | Debug logging | `False` |

## Adding New Components

### New Label Schema
Create `spine_vision/datasets/schemas/my_dataset.yaml`:
```yaml
name: MyDataset
source_labels:
  background: 0
  vertebra: 1
target_labels:
  background: 0
  vertebra: 1
mapping:
  0: 0
  1: 1
```

### New Dataset Pipeline
Add a new dataset module in `spine_vision/datasets/`:
```python
# spine_vision/datasets/my_dataset.py
from spine_vision.datasets.base import BaseProcessor, DatasetConfig, ProcessingResult

class MyDatasetConfig(DatasetConfig):
    """Configuration for my dataset."""
    custom_param: str = "value"

class MyDatasetProcessor(BaseProcessor[MyDatasetConfig]):
    """Processor for creating my dataset."""

    def __init__(self, config: MyDatasetConfig) -> None:
        super().__init__(config)
        setup_logger(verbose=config.verbose)
        if config.enable_file_log:
            add_file_log(config.log_path)

    def process(self) -> ProcessingResult:
        """Execute dataset creation pipeline."""
        self.on_process_begin()

        # Your processing logic here
        num_samples = 100

        result = ProcessingResult(
            num_samples=num_samples,
            output_path=self.config.output_path,
            summary=f"Processed {num_samples} samples",
        )

        self.on_process_end(result)
        return result

def main(config: MyDatasetConfig) -> None:
    """Convenience wrapper for backward compatibility."""
    processor = MyDatasetProcessor(config)
    result = processor.process()
    logger.info(result.summary)
```

Then register it in `spine_vision/cli/__init__.py`:
```python
from spine_vision.datasets.my_dataset import MyDatasetConfig, MyDatasetProcessor

# In cli() function:
case MyDatasetConfig():
    processor = MyDatasetProcessor(cmd)
    result = processor.process()
    logger.info(result.summary)
```

### New Training Model
```python
# spine_vision/training/models/my_model.py
import timm
import torch.nn as nn
from spine_vision.training.base import BaseModel

class MyModel(BaseModel):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, num_classes)
    
    @property
    def name(self) -> str:
        return "MyModel"
    
    def forward(self, x):
        return self.head(self.backbone(x))
    
    def get_loss(self, predictions, targets, **kwargs):
        return F.cross_entropy(predictions, targets)
```

## Gotchas

1. **NumPy version**: Requires `numpy<2` for compatibility with medical imaging libraries
2. **GPU dependencies**: Both `paddlepaddle` and `paddlepaddle-gpu` are listed
3. **Vietnamese OCR**: Uses VietOCR with `vgg_transformer` model
4. **Data paths**: All data and weights are gitignored
5. **Birthday parsing**: Expects `DD/MM/YYYY` format in OCR output
6. **nnUNet source**: Installed from GitHub via uv sources
7. **Training backend**: Uses HuggingFace Accelerate for distributed training
8. **Model backbones**: Uses timm for pretrained models
9. **Trackio logging**: Optional integration for experiment tracking

## Testing

No test suite currently configured.

## Type Checking

```bash
uv run pyright spine_vision
```

## Linting (REQUIRED)

```bash
uv run ruff check --fix spine_vision
```

> Agents MUST run this command after modifying any Python files.

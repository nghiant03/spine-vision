# AGENTS.md - Spine Vision

## Project Overview

Library for lumbar spine MRI dataset creation, model training, and result visualization. The project handles:
- Dataset creation pipelines (localization, classification datasets)
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
  - `torchmetrics` - Metrics computation

## Commands

### Setup
```bash
uv sync                         # Install dependencies
uv sync --group dev             # Include dev dependencies (pandas-stubs, plotly-stubs)
uv sync --group visualization   # Include visualization (seaborn, trackio)
```

### CLI Entry Points (after install)
```bash
spine-vision dataset localization [OPTIONS]    # Create localization dataset from RSNA + Lumbar Coords
spine-vision dataset phenikaa [OPTIONS]        # Preprocess Phenikaa dataset (OCR + matching)
spine-vision dataset classification [OPTIONS]  # Create classification dataset (Phenikaa + SPIDER)
spine-vision train localization [OPTIONS]      # Train localization model (ConvNext)
spine-vision train classification [OPTIONS]    # Train classification model (Multi-task)
spine-vision test [OPTIONS]                    # Test trained models with images/DICOM
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
│   ├── __init__.py            # Version string
│   ├── core/                  # Core utilities
│   │   ├── __init__.py        # Exports: BaseConfig, setup_logger, task system
│   │   ├── config.py          # BaseConfig (Pydantic model)
│   │   ├── logging.py         # Loguru setup with tqdm integration
│   │   └── tasks.py           # TaskConfig, TASK_REGISTRY, strategies
│   ├── io/                    # I/O utilities
│   │   ├── __init__.py        # normalize_to_uint8 + re-exports
│   │   ├── pdf.py             # PDF to image conversion
│   │   ├── readers.py         # DICOM, NIfTI, MHA, NRRD readers
│   │   ├── writers.py         # Format conversion & export
│   │   └── tabular.py         # Excel/CSV handling, write_records_csv
│   ├── datasets/              # Dataset creation pipelines
│   │   ├── __init__.py        # All processor exports
│   │   ├── base.py            # BaseProcessor, ProcessingResult
│   │   ├── levels.py          # IVD level constants (LEVEL_TO_IDX, etc.)
│   │   ├── localization.py    # LocalizationDatasetConfig, LocalizationDatasetProcessor
│   │   ├── classification.py  # ClassificationDatasetConfig, ClassificationDatasetProcessor
│   │   ├── rsna.py            # RSNA series mapping utilities
│   │   └── phenikaa/          # Phenikaa dataset preprocessing
│   │       ├── __init__.py    # PreprocessConfig, PhenikkaaProcessor
│   │       ├── ocr.py         # DocumentExtractor, TextDetector, TextRecognizer
│   │       └── matching.py    # PatientMatcher, fuzzy_value_extract
│   ├── training/              # Training infrastructure
│   │   ├── __init__.py        # All training exports
│   │   ├── base.py            # BaseTrainer, BaseModel, TrainingConfig, TrainingResult
│   │   ├── heads.py           # HeadConfig, HeadFactory, MLPHead, etc.
│   │   ├── losses.py          # FocalLoss
│   │   ├── metrics.py         # LocalizationMetrics, ClassifierMetrics
│   │   ├── registry.py        # ModelRegistry, TrainerRegistry, MetricsRegistry
│   │   ├── models/            # Model architectures
│   │   │   ├── __init__.py    # Model exports
│   │   │   ├── backbone.py    # BackboneFactory, BACKBONES dict (~40 options)
│   │   │   └── generic.py     # Classifier, CoordinateRegressor
│   │   ├── trainers/          # Task-specific trainers
│   │   │   ├── __init__.py
│   │   │   ├── localization.py  # LocalizationTrainer, LocalizationConfig
│   │   │   └── classification.py # ClassificationTrainer, ClassificationConfig
│   │   └── datasets/          # PyTorch datasets for training
│   │       ├── __init__.py
│   │       ├── localization.py  # LocalizationDataset, LocalizationCollator
│   │       └── classification.py # ClassificationDataset, ClassificationCollator, DynamicTargets
│   ├── visualization/         # Visualization utilities
│   │   ├── __init__.py        # All visualization exports
│   │   ├── base.py            # save_figure, load_original_images
│   │   ├── training.py        # plot_training_curves
│   │   ├── localization.py    # plot_localization_predictions, plot_error_distribution
│   │   ├── classification.py  # plot_confusion_matrix_with_samples, etc.
│   │   ├── dataset.py         # plot_dataset_statistics, plot_label_cooccurrence
│   │   └── visualizer.py      # TrainingVisualizer, DatasetVisualizer classes
│   └── cli/                   # Unified CLI with subcommands
│       ├── __init__.py        # Main entry point (cli function)
│       ├── train.py           # Training command entry point
│       └── test.py            # Testing/inference command
├── notebooks/                 # Jupyter notebooks for analysis
├── output/                    # Generated outputs (gitignored)
├── data/                      # Data directories (gitignored)
├── weights/                   # Model weights (gitignored)
└── pyproject.toml
```

## Module API

### Core (`spine_vision.core`)
```python
from spine_vision.core import setup_logger, add_file_log, BaseConfig

setup_logger(verbose=True)
add_file_log(log_path=Path("logs"))
```

### Task System (`spine_vision.core.tasks`)
```python
from spine_vision.core.tasks import (
    # Core types
    TaskConfig, TaskType, TaskStrategy,
    # Registry
    TASK_REGISTRY, AVAILABLE_TASK_NAMES,
    get_task, get_tasks, register_task,
    # Strategies
    get_strategy, BinaryStrategy, MulticlassStrategy, OrdinalStrategy,
    # Helpers
    create_loss_functions, compute_predictions_for_tasks,
    get_task_display_name, get_task_color,
)

# Get a task configuration
task = get_task("pfirrmann")  # TaskConfig
print(task.num_classes)        # 5
print(task.task_type)          # "multiclass"
print(task.display_name)       # "Pfirrmann Grade"

# Override training-specific settings (immutable, returns new copy)
task_with_smoothing = task.with_overrides(
    label_smoothing=0.1,
    loss_weight=2.0,
)

# Use strategy for task-type-specific behavior
strategy = get_strategy(task)
loss_fn = strategy.get_loss_fn(task)
preds = strategy.compute_predictions(logits)
metrics = strategy.get_metrics(task)

# Create loss functions for multiple tasks
tasks = get_tasks(["pfirrmann", "herniation", "modic"])
loss_fns, loss_weights = create_loss_functions(tasks)

# Visualization helpers
display = get_task_display_name("pfirrmann")  # "Pfirrmann Grade"
color = get_task_color("herniation")           # "#2ca02c"
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
    BaseProcessor, ProcessingResult,
    # Localization
    LocalizationDatasetConfig, LocalizationDatasetProcessor,
    # Phenikaa preprocessing
    PreprocessConfig, PhenikkaaProcessor,
    # Classification dataset
    ClassificationDatasetConfig, ClassificationDatasetProcessor,
    # RSNA utilities
    load_series_mapping, get_series_type,
)

# Localization dataset
config = LocalizationDatasetConfig(base_path=Path("data"))
processor = LocalizationDatasetProcessor(config)
result = processor.process()
print(f"Created {result.num_samples} localization annotations at {result.output_path}")

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

### Training (`spine_vision.training`)
```python
from spine_vision.training import (
    # Base classes
    BaseModel, BaseTrainer, TrainingConfig, TrainingResult,
    # Datasets
    LocalizationDataset, ClassificationDataset,
    # Models
    CoordinateRegressor, Classifier,
    # Trainers
    LocalizationConfig, LocalizationTrainer,
    ClassificationConfig, ClassificationTrainer,
    # Metrics & Visualization
    LocalizationMetrics, ClassifierMetrics,
    TrainingVisualizer, DatasetVisualizer,
)

# Training localization model
config = LocalizationConfig(
    data_path=Path("data/processed/localization"),
    backbone="convnext_base",
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

### spine-vision dataset localization
| Flag | Description | Default |
|------|-------------|---------|
| `--base-path` | Base data directory | `data` |
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
| `--data-path` | Localization dataset path | `data/processed/localization` |
| `--backbone` | Backbone model name (see BACKBONES) | `convnext_base` |
| `--image-size` | Input image size | `(512, 512)` |
| `--num-levels` | Number of IVD levels to predict | `5` |
| `--loss-type` | Loss function: mse, smooth_l1, huber | `mse` |
| `--freeze-backbone-epochs` | Epochs to freeze backbone | `0` |
| `--batch-size` | Training batch size | `32` |
| `--num-epochs` | Number of training epochs | `100` |
| `--learning-rate` | Learning rate | `1e-4` |
| `--use-trackio` | Enable trackio logging | `False` |
| `-v, --verbose` | Debug logging | `False` |

### spine-vision train classification
| Flag | Description | Default |
|------|-------------|---------|
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

### TrainingConfig (base for all trainers)
| Flag | Description | Default |
|------|-------------|---------|
| `--batch-size` | Training batch size | `32` |
| `--num-epochs` | Number of training epochs | `15` |
| `--learning-rate` | Learning rate | `1e-4` |
| `--weight-decay` | Weight decay | `1e-5` |
| `--scheduler-type` | LR scheduler: cosine, step, plateau, none | `cosine` |
| `--early-stopping` | Enable early stopping | `True` |
| `--patience` | Early stopping patience | `20` |
| `--mixed-precision` | Enable mixed precision training | `True` |

## Major Classes

### Core Layer
| Class | Purpose |
|-------|---------|
| `BaseConfig` | Pydantic base configuration with verbose, file logging options |
| `setup_logger()` | Configure loguru with tqdm integration |
| `add_file_log()` | Add rotating file log handler |

### Dataset Processing
| Class | Purpose |
|-------|---------|
| `BaseProcessor` | Abstract base for dataset processors |
| `ProcessingResult` | Container for processing statistics |
| `LocalizationDatasetProcessor` | Creates localization dataset from RSNA + Lumbar Coords |
| `ClassificationDatasetProcessor` | Creates cropped IVD images from Phenikaa + SPIDER |
| `PhenikkaaProcessor` | OCR extraction + patient matching |
| `PatientMatcher` | Fuzzy matching of patient data to image folders |
| `DocumentExtractor` | PaddleOCR + VietOCR text extraction |

### Training Infrastructure
| Class | Purpose |
|-------|---------|
| `TrainingConfig` | Base training config (lr, epochs, batch_size, etc.) |
| `TrainingResult` | Container for training results |
| `BaseModel` | Abstract base for models (forward, get_loss, predict, test_inference) |
| `BaseTrainer` | Abstract trainer with Accelerate + hooks |
| `BackboneFactory` | Creates timm backbones by name |
| `HeadFactory` | Creates configurable head architectures |
| `FocalLoss` | Focal loss for imbalanced binary classification |

### Models
| Class | Purpose |
|-------|---------|
| `CoordinateRegressor` | Backbone + regression head for IVD localization (outputs [B, 5, 2]) |
| `Classifier` | Backbone + multi-task heads for classification |

### Trainers
| Class | Purpose |
|-------|---------|
| `LocalizationTrainer` | Trainer for coordinate regression |
| `ClassificationTrainer` | Trainer for multi-task classification |

### Training Datasets
| Class | Purpose |
|-------|---------|
| `LocalizationDataset` | Loads images + all 5 IVD coordinates per image |
| `ClassificationDataset` | Loads IVD crops with T1/T2 pairing |
| `DynamicTargets` | Flexible container for any subset of target labels |
| `create_weighted_sampler()` | Creates WeightedRandomSampler for class imbalance |

### Task System
| Class | Purpose |
|-------|---------|
| `TaskConfig` | Immutable config for classification task (name, num_classes, task_type, etc.) |
| `TaskStrategy` | ABC for task-type-specific behavior (loss, predictions, metrics) |
| `BinaryStrategy` | Strategy for binary tasks (BCEWithLogitsLoss, sigmoid) |
| `MulticlassStrategy` | Strategy for multiclass tasks (CrossEntropyLoss, softmax) |
| `OrdinalStrategy` | Strategy for ordinal tasks (CE + MAE metric) |
| `TASK_REGISTRY` | Dict of 8 predefined tasks (pfirrmann, modic, herniation, etc.) |
| `get_task()` | Get TaskConfig by name |
| `get_strategy()` | Get TaskStrategy for a task/type |

### Metrics
| Class | Purpose |
|-------|---------|
| `LocalizationMetrics` | MED, MAE, PCK metrics with per-level breakdown |
| `ClassifierMetrics` | Multi-task metrics aggregator (overall_accuracy, macro_f1) |

### Visualization
| Class | Purpose |
|-------|---------|
| `TrainingVisualizer` | Training curves, predictions, confusion matrices |
| `DatasetVisualizer` | Dataset statistics, distributions |

## Adding New Components

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
```

Then register it in `spine_vision/cli/__init__.py`.

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
8. **Model backbones**: Uses timm for pretrained models (~40 options in BACKBONES)
9. **Trackio logging**: Optional integration for experiment tracking
10. **IVD Level Conventions**:
    - SPIDER: L5/S1=1, L1/L2=5 (bottom to top)
    - Phenikaa: L1/L2=1, L5/S1=5 (top to bottom)
    - Dataset converts SPIDER to Phenikaa convention
11. **Pfirrmann Grades**: 1-5 in CSV, converted to 0-4 for CrossEntropy
12. **3-Channel Input**: Classification uses [T2, T1, T2] or single-modality triplicate
13. **Crop Coordinates in mm**: `crop_delta_mm` in mm, converted to pixels using image spacing
14. **Crop Modes**: "horizontal" (axis-aligned) or "rotated" (spine-aligned)
15. **Weighted Sampling**: Default for handling class imbalance (not loss weighting)
16. **Backbone Freezing**: Optional via `freeze_backbone_epochs`
17. **Checkpoint Selection**:
    - Localization: MED (lower is better)
    - Classification: F1/macro_F1 (negated for lower-is-better)
18. **Isotropic Resampling**: 0.3mm spacing for consistent cropping

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

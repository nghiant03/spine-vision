# AGENTS.md - Spine Vision

## Project Overview

Medical imaging pipeline for lumbar spine MRI analysis and radiological grading prediction. The project handles:
- OCR-based extraction of patient information from medical reports
- Fuzzy matching of patient data across different data sources
- DICOM/MHA to NIfTI format conversion for nnUNet training
- Tabular data preprocessing from Excel/CSV files
- Interactive visualization of segmentation results with nnUNet inference
- Extensible inference framework for segmentation, localization, and cropping
- Dataset creation pipelines (nnUNet, IVD coordinates, RSNA processing)

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
  - `torch` / `torchvision` - Deep learning backend (GPU support)
  - `plotly` / `kaleido` - Interactive visualization and image export
  - `pyyaml` - Label schema configuration
  - `rich` / `tqdm` - Progress bars and terminal output

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
spine-vision test [OPTIONS]                    # Test trained models with images/DICOM
spine-vision visualize [OPTIONS]               # Visualize segmentation with inference
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
│   │   ├── __init__.py
│   │   ├── logging.py         # Centralized logging setup
│   │   ├── config.py          # Base config classes
│   │   └── pipeline.py        # Unified pipeline orchestration
│   ├── io/                    # I/O utilities
│   │   ├── __init__.py
│   │   ├── readers.py         # DICOM, NIfTI, MHA, NRRD readers
│   │   ├── writers.py         # Format conversion & export
│   │   ├── tabular.py         # Excel/CSV handling
│   │   └── transforms.py      # Image normalization utilities
│   ├── ocr/                   # OCR module
│   │   ├── __init__.py
│   │   ├── detection.py       # Text detection (PaddleOCR)
│   │   ├── recognition.py     # Text recognition (VietOCR)
│   │   └── extraction.py      # Document field extraction
│   ├── matching/              # Fuzzy matching
│   │   ├── __init__.py
│   │   ├── fuzzy.py           # Fuzzy string utilities
│   │   └── patient.py         # Patient folder matching
│   ├── inference/             # Inference models
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract InferenceModel class
│   │   ├── segmentation.py    # nnUNet + other segmentation
│   │   ├── localization.py    # Bounding box / landmark detection
│   │   └── cropping.py        # ROI extraction
│   ├── labels/                # Label management (generic utilities)
│   │   ├── __init__.py
│   │   └── mapping.py         # LabelSchema class, remap_labels, load_label_schema
│   ├── datasets/              # Dataset creation pipelines + dataset-specific info
│   │   ├── __init__.py
│   │   ├── nnunet.py          # nnUNet format conversion
│   │   ├── ivd_coords.py      # IVD coordinates dataset creation
│   │   ├── classification.py  # Classification dataset (Phenikaa + SPIDER)
│   │   ├── phenikaa.py        # Phenikaa preprocessing (OCR + matching)
│   │   ├── rsna.py            # RSNA dataset utilities (series mapping)
│   │   └── schemas/           # YAML label definitions
│   │       └── spider.yaml    # SPIDER dataset labels
│   ├── visualization/         # Visualization
│   │   ├── __init__.py
│   │   └── plotly_viewer.py   # Interactive batch-capable viewer
│   ├── training/              # Training infrastructure
│   │   ├── __init__.py
│   │   ├── base.py            # BaseTrainer, BaseModel, TrainingConfig
│   │   ├── metrics.py         # BaseMetrics, LocalizationMetrics, MTLClassificationMetrics
│   │   ├── visualization.py   # TrainingVisualizer for curves/predictions
│   │   ├── datasets/          # PyTorch datasets for training
│   │   │   ├── __init__.py
│   │   │   ├── ivd_coords.py  # IVDCoordsDataset
│   │   │   └── classification.py # ClassificationDataset, ClassificationCollator
│   │   ├── models/            # Model architectures
│   │   │   ├── __init__.py
│   │   │   ├── convnext.py    # ConvNextLocalization, ConvNextClassifier
│   │   │   ├── resnet_mtl.py  # ResNet50MTL, MTLPredictions, MTLTargets
│   │   │   └── vit.py         # VisionTransformerLocalization
│   │   └── trainers/          # Task-specific trainers
│   │       ├── __init__.py
│   │       ├── localization.py # LocalizationTrainer, LocalizationConfig
│   │       └── classification.py # ClassificationTrainer, ClassificationConfig
│   └── cli/                   # Unified CLI with subcommands
│       ├── __init__.py        # Main entry point (dataset, train, visualize)
│       ├── visualize.py       # Visualization command config and logic
│       └── train.py           # Training command entry point
├── configs/                   # YAML config files (optional)
├── data/                      # (gitignored) Data directories
├── weights/                   # (gitignored) Model weights
├── config.py                  # Legacy config (deprecated)
├── preprocess.py              # Legacy script (deprecated)
├── convert.py                 # Legacy script (deprecated)
├── visualize.py               # Legacy script (deprecated)
└── pyproject.toml
```

## Module API

### Core (`spine_vision.core`)
```python
from spine_vision.core import setup_logger, BaseConfig

setup_logger(verbose=True, enable_file_log=True, log_path=Path("logs"))
```

### I/O (`spine_vision.io`)
```python
from spine_vision.io import (
    read_medical_image, write_medical_image, load_tabular_data,
    normalize_to_uint8, write_records_csv
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

### OCR (`spine_vision.ocr`)
```python
from spine_vision.ocr import DocumentExtractor, crop_polygon

# Supports images (.png, .jpg, etc.) and PDFs
extractor = DocumentExtractor(device="cuda:0", pdf_dpi=200)
text_lines = extractor.extract(Path("report.pdf"))  # or .png
```

### Matching (`spine_vision.matching`)
```python
from spine_vision.matching import fuzzy_value_extract, PatientMatcher

# Extract field value from OCR text
name = fuzzy_value_extract(text_lines, "Ho ten nguoi benh", threshold=80)

# Match patients to image folders
matcher = PatientMatcher(image_path=Path("images/"), threshold=85)
folder = matcher.match(patient_name, patient_birthday)
folder = matcher.match_by_name(patient_name)  # When birthday unavailable
```

### Inference (`spine_vision.inference`)
```python
from spine_vision.inference import NNUNetSegmentation, ROICropper

# nnU-Net segmentation
model = NNUNetSegmentation(
    model_path=Path("weights/"),
    dataset_id=501,
    configuration="3d_fullres",
)
result = model.predict(image)

# ROI cropping from segmentation
cropper = ROICropper(padding=(10, 10, 10))
crops = cropper.crop_from_segmentation(image, segmentation)
```

### Labels (`spine_vision.labels`)
```python
from spine_vision.labels import load_label_schema, remap_labels

# Load from YAML or built-in schema
schema = load_label_schema("spider")  # or Path("custom.yaml")
remapped = remap_labels(mask_array, schema.mapping)
```

### Datasets (`spine_vision.datasets`)
```python
from spine_vision.datasets.nnunet import ConvertConfig, main as convert_main
from spine_vision.datasets.ivd_coords import IVDDatasetConfig, main as ivd_main
from spine_vision.datasets.phenikaa import PreprocessConfig, main as preprocess_main
from spine_vision.datasets.classification import ClassificationDatasetConfig, main as classification_main
from spine_vision.datasets import load_series_mapping, get_series_type

# nnUNet conversion
config = ConvertConfig(input_path=Path("data/raw/SPIDER"))
convert_main(config)

# IVD coordinates dataset
config = IVDDatasetConfig(base_path=Path("data"))
ivd_main(config)

# Phenikaa preprocessing (supports both report formats)
# - ID-named reports (250010139.png): extracts name/birthday from OCR
# - Patient-named reports (NGUYEN_VAN_SON_20250718.pdf): extracts ID from OCR
config = PreprocessConfig(data_path=Path("data/raw/Phenikaa"))
preprocess_main(config)

# Classification dataset (Phenikaa + SPIDER with IVD cropping)
config = ClassificationDatasetConfig(
    base_path=Path("data"),
    localization_model_path=Path("weights/localization/model.pt"),
    crop_size=(64, 64),
)
classification_main(config)

# RSNA series mapping
series_mapping = load_series_mapping(Path("train_series_descriptions.csv"))
series_type = get_series_type(series_id=67890, study_id=12345, series_mapping=series_mapping)
```

### Visualization (`spine_vision.visualization`)
```python
from spine_vision.visualization import PlotlyViewer

viewer = PlotlyViewer(output_path=Path("results/"), output_mode="html")
viewer.visualize(image, mask, title="Segmentation")
viewer.visualize_batch(images, masks)  # Batch processing
```

### Training (`spine_vision.training`)
```python
from spine_vision.training import (
    # Base classes
    BaseModel, BaseTrainer, TrainingConfig, TrainingResult,
    # Datasets
    IVDCoordsDataset, ClassificationDataset, ClassificationCollator,
    LEVEL_TO_IDX, IDX_TO_LEVEL, construct_3channel,
    # Models (via timm) - auto-registered
    ConvNextLocalization, ConvNextClassifier, VisionTransformerLocalization,
    ResNet50MTL, MTLPredictions, MTLTargets,
    # Trainers - auto-registered
    LocalizationConfig, LocalizationTrainer,
    ClassificationConfig, ClassificationTrainer,
    # Registries for extensibility
    ModelRegistry, TrainerRegistry, MetricsRegistry,
    register_model, register_trainer, register_metrics,
    # Configurable heads
    HeadConfig, HeadFactory, create_head, BaseHead, MLPHead, MultiTaskHead,
    # Metrics & Visualization
    BaseMetrics, MetricResult, LocalizationMetrics, MTLClassificationMetrics,
    TrainingVisualizer,
)

# Using registries for dynamic model/trainer discovery
model = ModelRegistry.create("convnext_localization", variant="base")
trainer = TrainerRegistry.create_from_config(config)  # Uses config.task

# List available models/trainers
ModelRegistry.list_models(task="localization")  # ["convnext_localization", "vit_localization"]
TrainerRegistry.list_trainers()  # ["localization", "classification"]

# Using configurable heads
from spine_vision.training.heads import HeadConfig
head_config = HeadConfig(
    head_type="attention",  # or "mlp", "residual", "conv", "linear"
    hidden_dims=[512],
    dropout=0.2,
    output_activation="sigmoid",
)
model = ConvNextLocalization(variant="base", head_config=head_config)

# Training localization model with wandb logging
# Output structure: weights/localization/<run_id>/
#   - best_model.pt, checkpoint_epoch_N.pt
#   - config.yaml (saved automatically)
#   - logs/ (training_curves.html, predictions_epoch_N.html, etc.)
config = LocalizationConfig(
    data_path=Path("data/processed/ivd_coords"),
    # output_path auto-generated: weights/localization/<run_id>
    # run_id auto-generated if not provided
    model_variant="base",  # tiny, small, base, large, xlarge, v2_tiny, v2_small, v2_base, v2_large, v2_huge
    batch_size=32,
    num_epochs=100,
    learning_rate=1e-4,
    use_wandb=True,  # Enable wandb logging (run_name synced with run_id)
    wandb_project="spine-vision",
    # wandb_run_name defaults to run_id for easy mapping
)
trainer = LocalizationTrainer(config)
result = trainer.train()

# Evaluate on test set
test_metrics = trainer.evaluate()

# Training MTL classification model (ResNet50 with 6 heads)
# Predicts 13 clinical labels per disc:
#   - Pfirrmann grade (5 classes)
#   - Modic type (4 classes)  
#   - Herniation + Bulging (2 binary)
#   - Upper/Lower Endplate (2 binary)
#   - Spondylolisthesis (1 binary)
#   - Narrowing (1 binary)
config = ClassificationConfig(
    data_path=Path("data/processed/classification"),
    output_size=(224, 224),  # Final input to model
    dropout=0.3,
    label_smoothing=0.1,
    freeze_backbone_epochs=5,  # Freeze ResNet backbone initially
    batch_size=32,
    num_epochs=100,
    learning_rate=1e-4,
    use_wandb=True,
)
trainer = ClassificationTrainer(config)
result = trainer.train()

# Create custom localization dataset
dataset = IVDCoordsDataset(
    data_path=Path("data/processed/ivd_coords"),
    split="train",
    series_types=["sag_t1", "sag_t2"],
    image_size=(224, 224),
    augment=True,
)

# Create classification dataset from pre-extracted crops
# Automatically pairs T1+T2 images for 3-channel [T2, T1, T2] input
dataset = ClassificationDataset(
    data_path=Path("data/processed/classification"),
    split="train",
    levels=["L4/L5", "L5/S1"],  # Optional level filtering
    output_size=(224, 224),  # Final resize
    augment=True,
)

# Test model inference with images
model = ConvNextLocalization(variant="base", pretrained=True)
model.load_state_dict(torch.load("best_model.pt")["model_state_dict"])
result = model.test_inference(
    images=["image1.png", "image2.png"],
    image_size=(224, 224),
    device="cuda:0",
    level_indices=[0, 1],  # L1/L2, L2/L3
)
print(result["predictions"])  # Predicted coordinates [N, 2]
print(result["coords_pixel"])  # Coordinates in pixel space
print(result["inference_time_ms"])  # Inference time

# Test MTL classification model
mtl_model = ResNet50MTL(pretrained=True)
mtl_model.load_state_dict(torch.load("best_model.pt")["model_state_dict"])
result = mtl_model.test_inference(
    images=["crop1.png", "crop2.png"],
    image_size=(224, 224),
    device="cuda:0",
)
print(result["predictions"]["pfirrmann"])  # Grades 1-5
print(result["predictions"]["modic"])  # Types 0-3
print(result["predictions"]["herniation"])  # [B, 2] binary

# Generate CSV output for competition
predictions = mtl_model.predict(batch)
csv_row = mtl_model.to_csv_row(predictions, patient_id="P001", ivd_level=4)
header = ResNet50MTL.get_csv_header()

# Metrics computation
metrics = LocalizationMetrics(pck_thresholds=[0.02, 0.05, 0.10])
result = metrics.compute(predictions, targets, levels)

# Training visualization with wandb
visualizer = TrainingVisualizer(
    output_path=Path("vis/"),
    output_mode="html",
    use_wandb=True,  # Also log to wandb
)
visualizer.plot_training_curves(history)
visualizer.plot_error_distribution(predictions, targets, levels)
```

## CLI Options

### spine-vision dataset phenikaa
Supports two report formats automatically:
- **ID-named** (e.g., `250010139.png`): ID from filename, name/birthday from OCR
- **Patient-named** (e.g., `NGUYEN_VAN_SON_20250718.pdf`): Name from filename, ID from OCR

| Flag | Description | Default |
|------|-------------|---------|
| `--data-path` | Input data directory | `data/raw/Phenikaa` |
| `--output-path` | Output directory | `data/processed/classification` |
| `-g, --use-gpu` | Enable GPU acceleration | `True` |
| `-v, --verbose` | Debug logging | `False` |
| `--enable-file-log` | Write logs to file | `False` |
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
| `--file-extension` | Input file extension | `.mha` |
| `-v, --verbose` | Debug logging | `False` |

### spine-vision dataset localization
| Flag | Description | Default |
|------|-------------|---------|
| `--base-path` | Base data directory | `data` |
| `--output-name` | Output dataset folder name | `ivd_coords` |
| `--include-neural-foraminal` | Include Neural Foraminal annotations | `True` |
| `--include-spinal-canal` | Include Spinal Canal annotations | `True` |
| `--skip-invalid-instances` | Skip records with invalid instance numbers | `True` |
| `-v, --verbose` | Debug logging | `False` |

### spine-vision dataset classification
| Flag | Description | Default |
|------|-------------|---------|
| `--base-path` | Base data directory | `data` |
| `--output-name` | Output dataset folder name | `classification` |
| `--localization-model-path` | Path to trained localization model (optional) | None |
| `--model-variant` | ConvNext variant for localization | `base` |
| `--crop-size` | Output size of cropped IVD regions in pixels (H W) | `128 128` |
| `--crop-delta` | Crop region deltas (left right top bottom) in pixels | `96 32 64 64` |
| `--image-size` | Input image size for localization model (H W) | `224 224` |
| `--include-phenikaa` | Include Phenikaa dataset | `True` |
| `--include-spider` | Include SPIDER dataset | `True` |
| `--device` | Device for model inference | `cuda:0` |
| `-v, --verbose` | Debug logging | `False` |

### spine-vision visualize
| Flag | Description | Default |
|------|-------------|---------|
| `--input-path` | DICOM input directory | `data/processed/classification/images/...` |
| `--output-path` | Output directory | `results/segmentation` |
| `--model-path` | nnUNet model path | `weights/segmentation/...` |
| `--dataset-id` | nnUNet dataset ID | `501` |
| `--configuration` | nnUNet configuration | `2d` |
| `--fold` | Model fold | `0` |
| `--output-mode` | Output format (`browser`, `html`, `image`) | `html` |
| `--batch` | Enable batch processing | `False` |
| `-v, --verbose` | Debug logging | `False` |

### spine-vision train localization
| Flag | Description | Default |
|------|-------------|---------|
| `--data-path` | IVD coordinates dataset path | `data/processed/ivd_coords` |
| `--output-path` | Training output directory | `weights/<task>/<run_id>` |
| `--model-variant` | ConvNext variant (`tiny`, `small`, `base`, `large`, `xlarge`, `v2_tiny`, `v2_small`, `v2_base`, `v2_large`, `v2_huge`) | `base` |
| `--batch-size` | Training batch size | `32` |
| `--num-epochs` | Number of training epochs | `100` |
| `--learning-rate` | Learning rate | `1e-4` |
| `--scheduler-type` | LR scheduler (`cosine`, `step`, `plateau`, `none`) | `cosine` |
| `--freeze-backbone-epochs` | Epochs to freeze backbone | `0` |
| `--loss-type` | Loss function (`mse`, `smooth_l1`, `huber`) | `smooth_l1` |
| `--use-level-embedding` | Use IVD level embedding | `True` |
| `--series-types` | Filter series types (e.g., `sag_t1 sag_t2`) | None |
| `--levels` | Filter IVD levels (e.g., `L4/L5 L5/S1`) | None |
| `--image-size` | Target image size (H W) | `224 224` |
| `--augment` | Enable data augmentation | `True` |
| `--mixed-precision` | Use mixed precision training (via Accelerate) | `True` |
| `--early-stopping` | Enable early stopping | `True` |
| `--patience` | Early stopping patience | `20` |
| `--visualize-predictions` | Generate prediction visualizations | `True` |
| `--use-wandb` | Enable wandb logging | `False` |
| `--wandb-project` | Wandb project name | `spine-vision` |
| `--run-id` | Unique run identifier | Auto-generated |
| `--wandb-run-name` | Wandb run name | Same as run_id |
| `--wandb-tags` | Wandb tags (e.g., `exp1 baseline`) | None |
| `-v, --verbose` | Debug logging | `False` |

### spine-vision test
| Flag | Description | Default |
|------|-------------|---------|
| `--model-path` | Path to trained model checkpoint (.pt file) | Required |
| `--inputs` | Input image or DICOM file paths | Required |
| `--task` | Task type (`localization`, `classification`, `mtl_classification`) | `localization` |
| `--model-variant` | ConvNext variant (for localization/classification) | `base` |
| `--level-indices` | IVD level indices (0-4) for localization (if not provided, tests all levels) | None |
| `--test-all-levels` | When level_indices is None, test all 5 levels per image | `True` |
| `--num-levels` | Number of IVD levels model was trained on | `5` |
| `--use-level-embedding` | Whether model uses level embedding | `True` |
| `--num-classes` | Number of classes for classification | `4` |
| `--class-names` | Optional class names for output | None |
| `--image-size` | Target image size (H W) | `224 224` |
| `--device` | Inference device | `cuda:0` |
| `--output-path` | Path to save results (JSON) | None |
| `--visualize` | Generate prediction visualizations | `False` |
| `-v, --verbose` | Debug logging | `False` |

### spine-vision train classification
| Flag | Description | Default |
|------|-------------|---------|
| `--data-path` | Classification dataset path | `data/processed/classification` |
| `--output-path` | Training output directory | `weights/classification/<run_id>` |
| `--output-size` | Final input size to model (H W) | `224 224` |
| `--levels` | Filter IVD levels (e.g., `L4/L5 L5/S1`) | None |
| `--pretrained` | Use ImageNet pretrained weights | `True` |
| `--dropout` | Dropout rate | `0.3` |
| `--freeze-backbone-epochs` | Epochs to freeze backbone | `0` |
| `--label-smoothing` | Label smoothing for cross-entropy | `0.1` |
| `--batch-size` | Training batch size | `32` |
| `--num-epochs` | Number of training epochs | `100` |
| `--learning-rate` | Learning rate | `1e-4` |
| `--augment` | Enable data augmentation | `True` |
| `--mixed-precision` | Use mixed precision training | `True` |
| `--early-stopping` | Enable early stopping | `True` |
| `--patience` | Early stopping patience | `20` |
| `--visualize-predictions` | Generate prediction visualizations | `True` |
| `--use-wandb` | Enable wandb logging | `False` |
| `--wandb-project` | Wandb project name | `spine-vision` |
| `-v, --verbose` | Debug logging | `False` |

## Adding New Components

### New Inference Model
```python
from spine_vision.inference.base import InferenceModel, InferenceResult

class MyModel(InferenceModel):
    @property
    def name(self) -> str:
        return "My Custom Model"

    def load(self) -> None:
        self._model = load_my_model(self.model_path)

    def predict(self, image: sitk.Image) -> InferenceResult:
        # Run inference
        return InferenceResult(prediction=output)
```

### New Label Schema
Create `spine_vision/labels/schemas/my_dataset.yaml`:
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
metadata:
  description: My custom dataset
```

### New Dataset Pipeline
Add a new dataset module in `spine_vision/datasets/`:
```python
# spine_vision/datasets/my_dataset.py
from pydantic import BaseModel

class MyDatasetConfig(BaseModel):
    """Configuration for my dataset."""
    input_path: Path
    output_path: Path

def main(config: MyDatasetConfig) -> None:
    """Create my dataset."""
    ...
```

Then register it in `spine_vision/cli/__init__.py`:
```python
Command = Union[
    ...,
    Annotated[MyDatasetConfig, tyro.conf.subcommand("my-dataset", description="...")],
]
```

### New Training Model
Create a new model in `spine_vision/training/models/` using the registry decorator:
```python
# spine_vision/training/models/my_model.py
import timm
import torch.nn as nn
from spine_vision.training.base import BaseModel
from spine_vision.training.registry import register_model
from spine_vision.training.heads import HeadConfig, create_head

@register_model(
    "my_model",
    task="classification",
    description="My custom model",
    aliases=["my_alias"],
)
class MyModel(BaseModel):
    def __init__(
        self,
        num_classes: int = 4,
        head_config: HeadConfig | None = None,  # Configurable head
    ) -> None:
        super().__init__()
        # Use timm for pretrained backbones
        self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
        
        # Use configurable head or default
        if head_config is not None:
            self.head = create_head(head_config, self.backbone.num_features, num_classes)
        else:
            self.head = nn.Linear(self.backbone.num_features, num_classes)
    
    @property
    def name(self) -> str:
        return "MyModel"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
    
    def get_loss(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        return F.cross_entropy(predictions, targets)
    
    # test_inference() is already provided by BaseModel
    # Override for custom behavior if needed

# Models are auto-discovered via registry
# Access: ModelRegistry.create("my_model", num_classes=10)
# List: ModelRegistry.list_models(task="classification")
```

### Configurable Heads
Use `HeadConfig` to experiment with different head architectures:
```python
from spine_vision.training.heads import HeadConfig, create_head, HeadFactory

# Available head types: mlp, linear, attention, residual, conv
config = HeadConfig(
    head_type="mlp",          # or "attention", "residual", etc.
    hidden_dims=[512, 256],   # Hidden layer dimensions
    dropout=0.2,
    activation="gelu",
    use_layer_norm=True,
    output_activation="sigmoid",  # For regression in [0,1]
)
head = create_head(config, in_features=2048, out_features=2)

# Or use factory directly
head = HeadFactory.create("attention", in_features=2048, out_features=4, num_heads=8)

# Register custom head type
@HeadFactory.register("my_head")
class MyHead(BaseHead):
    ...
```

### New Training Task
Create a new trainer using the registry and hooks:
```python
# spine_vision/training/trainers/my_task.py
from spine_vision.training.base import BaseTrainer, TrainingConfig, TrainingResult
from spine_vision.training.registry import register_trainer

class MyTaskConfig(TrainingConfig):
    """Configuration for my task."""
    task: str = "my_task"  # Used for output path: weights/my_task/<run_id>
    my_param: int = 10

@register_trainer("my_task", config_cls=MyTaskConfig)
class MyTaskTrainer(BaseTrainer[MyTaskConfig, MyModel, MyDataset]):
    def _unpack_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        return batch["input"], batch["target"]
    
    def _compute_metrics(self, predictions, targets) -> dict[str, float]:
        return {"accuracy": compute_accuracy(predictions, targets)}
    
    # Use hooks instead of overriding train() - less code duplication!
    def on_train_begin(self) -> None:
        """Called before training starts."""
        logger.info(f"My param: {self.config.my_param}")
    
    def on_epoch_begin(self, epoch: int) -> None:
        """Called at start of each epoch. Good for unfreezing, LR adjustments."""
        if epoch == 5:
            self.accelerator.unwrap_model(self.model).unfreeze_backbone()
    
    def on_train_end(self, result: TrainingResult) -> None:
        """Called after training. Good for final visualizations."""
        self.visualizer.plot_training_curves(self.history)
    
    def get_metric_for_checkpoint(self, val_loss, metrics) -> float:
        """Override to use custom metric for best model selection."""
        return -metrics.get("accuracy", 0)  # Negate for lower-is-better

# Trainers are auto-discovered via registry
# Access: TrainerRegistry.create("my_task", config)
# Or: TrainerRegistry.create_from_config(config)  # Uses config.task
```

### Training Hooks Reference
Available hooks in `BaseTrainer`:
| Hook | When Called | Use Case |
|------|-------------|----------|
| `on_train_begin()` | Before training loop | Initialize state, log info |
| `on_train_end(result)` | After training completes | Final visualizations, cleanup |
| `on_epoch_begin(epoch)` | Start of each epoch | Unfreeze layers, adjust LR |
| `on_epoch_end(epoch, train_loss, val_loss, metrics)` | End of each epoch | Custom logging |
| `on_batch_begin(batch_idx, batch)` | Before each batch | Batch-level adjustments |
| `on_batch_end(batch_idx, batch, loss)` | After each batch | Gradient analysis |
| `on_validation_begin()` | Before validation | Reset metrics accumulators |
| `on_validation_end(val_loss, metrics)` | After validation | Validation visualization |
| `get_metric_for_checkpoint(val_loss, metrics)` | During checkpointing | Custom metric for best model |

Then register in `spine_vision/cli/__init__.py` under `TrainSubcommand`.

## Gotchas

1. **NumPy version**: Requires `numpy<2` for compatibility with medical imaging libraries
2. **GPU dependencies**: Both `paddlepaddle` and `paddlepaddle-gpu` are listed
3. **Vietnamese OCR**: Uses VietOCR with `vgg_transformer` model
4. **Data paths**: All data and weights are gitignored
5. **Birthday parsing**: Expects `DD/MM/YYYY` format in OCR output
6. **Legacy scripts**: Root-level `preprocess.py`, `convert.py`, `visualize.py` are deprecated - use CLI entry points
7. **nnUNet source**: Installed from GitHub via uv sources
8. **Training backend**: Uses HuggingFace Accelerate for distributed training and mixed precision
9. **Model backbones**: Uses timm for pretrained models with extensive architecture support
10. **Wandb logging**: Optional integration for experiment tracking (set `use_wandb=True`)

## Testing

No test suite currently configured. When adding tests:
- Consider using `pytest`
- Mock OCR models for unit tests
- Use sample data fixtures for integration tests

## Type Checking

```bash
uv run pyright spine_vision
```

## Linting (REQUIRED)

**IMPORTANT**: Always run the linter after making code changes.

```bash
uv run ruff check --fix spine_vision
```

> Agents MUST run this command after modifying any Python files. Fix all linting errors before completing tasks.

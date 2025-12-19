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
spine-vision dataset nnunet [OPTIONS]     # Convert datasets to nnU-Net format
spine-vision dataset ivd-coords [OPTIONS] # Create IVD coordinates dataset
spine-vision dataset phenikaa [OPTIONS]   # Preprocess Phenikaa dataset (OCR + matching)
spine-vision train localization [OPTIONS] # Train localization model (ConvNext)
spine-vision visualize [OPTIONS]          # Visualize segmentation with inference
```

### Running Scripts Directly
```bash
uv run python -m spine_vision.cli [OPTIONS]
```

### Type Checking
```bash
uv run pyright spine_vision
```

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
│   │   ├── metrics.py         # LocalizationMetrics, ClassificationMetrics
│   │   ├── visualization.py   # TrainingVisualizer for curves/predictions
│   │   ├── datasets/          # PyTorch datasets for training
│   │   │   ├── __init__.py
│   │   │   └── ivd_coords.py  # IVDCoordsDataset
│   │   ├── models/            # Model architectures
│   │   │   ├── __init__.py
│   │   │   └── convnext.py    # ConvNextLocalization, ConvNextClassifier
│   │   └── trainers/          # Task-specific trainers
│   │       ├── __init__.py
│   │       └── localization.py # LocalizationTrainer, LocalizationConfig
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

extractor = DocumentExtractor(device="cuda:0")
text_lines = extractor.extract(Path("report.png"))
```

### Matching (`spine_vision.matching`)
```python
from spine_vision.matching import fuzzy_value_extract, PatientMatcher

# Extract field value from OCR text
name = fuzzy_value_extract(text_lines, "Ho ten nguoi benh", threshold=80)

# Match patients to image folders
matcher = PatientMatcher(image_path=Path("images/"), threshold=85)
folder = matcher.match(patient_name, patient_birthday)
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
from spine_vision.datasets import load_series_mapping, get_series_type

# nnUNet conversion
config = ConvertConfig(input_path=Path("data/raw/SPIDER"))
convert_main(config)

# IVD coordinates dataset
config = IVDDatasetConfig(base_path=Path("data"))
ivd_main(config)

# Phenikaa preprocessing
config = PreprocessConfig(data_path=Path("data/silver/Phenikaa"))
preprocess_main(config)

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
    # Dataset
    IVDCoordsDataset,
    # Models (via timm)
    ConvNextLocalization, ConvNextClassifier, VisionTransformerLocalization,
    # Trainers
    LocalizationConfig, LocalizationTrainer,
    # Metrics & Visualization
    LocalizationMetrics, TrainingVisualizer,
)

# Training localization model with wandb logging
# Output structure: weights/localization/<run_id>/
#   - best_model.pt, checkpoint_epoch_N.pt
#   - config.yaml (saved automatically)
#   - logs/ (training_curves.html, predictions_epoch_N.html, etc.)
config = LocalizationConfig(
    data_path=Path("data/gold/ivd_coords"),
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

# Create custom dataset
dataset = IVDCoordsDataset(
    data_path=Path("data/gold/ivd_coords"),
    split="train",
    series_types=["sag_t1", "sag_t2"],
    image_size=(224, 224),
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
| Flag | Description | Default |
|------|-------------|---------|
| `--data-path` | Input data directory | `data/silver/Phenikaa` |
| `--output-path` | Output directory | `data/gold/classification` |
| `-g, --use-gpu` | Enable GPU acceleration | `True` |
| `-v, --verbose` | Debug logging | `False` |
| `--enable-file-log` | Write logs to file | `False` |
| `--report-fuzzy-threshold` | OCR matching threshold | `80` |
| `--image-fuzzy-threshold` | Folder matching threshold | `85` |

### spine-vision dataset nnunet
| Flag | Description | Default |
|------|-------------|---------|
| `--input-path` | Source dataset directory | `data/raw/SPIDER` |
| `--output-path` | nnUNet output directory | `data/silver/SPIDER/Dataset501_Spider` |
| `--schema-path` | Label schema YAML (optional) | Built-in `spider` |
| `--channel-name` | Channel name in dataset.json | `MRI` |
| `--file-extension` | Input file extension | `.mha` |
| `-v, --verbose` | Debug logging | `False` |

### spine-vision dataset ivd-coords
| Flag | Description | Default |
|------|-------------|---------|
| `--base-path` | Base data directory | `data` |
| `--output-name` | Output dataset folder name | `ivd_coords` |
| `--include-neural-foraminal` | Include Neural Foraminal annotations | `True` |
| `--include-spinal-canal` | Include Spinal Canal annotations | `True` |
| `--skip-invalid-instances` | Skip records with invalid instance numbers | `True` |
| `-v, --verbose` | Debug logging | `False` |

### spine-vision visualize
| Flag | Description | Default |
|------|-------------|---------|
| `--input-path` | DICOM input directory | `data/gold/classification/images/...` |
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
| `--data-path` | IVD coordinates dataset path | `data/gold/ivd_coords` |
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
Create a new model in `spine_vision/training/models/`:
```python
# spine_vision/training/models/my_model.py
import timm
from spine_vision.training.base import BaseModel

class MyModel(BaseModel):
    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        # Use timm for pretrained backbones
        self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
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
```

### New Training Task
Create a new trainer in `spine_vision/training/trainers/`:
```python
# spine_vision/training/trainers/my_task.py
from spine_vision.training.base import BaseTrainer, TrainingConfig

class MyTaskConfig(TrainingConfig):
    """Configuration for my task."""
    my_param: int = 10
    # Wandb options are inherited from TrainingConfig:
    # use_wandb, wandb_project, wandb_run_name, wandb_tags

class MyTaskTrainer(BaseTrainer[MyTaskConfig, MyModel, MyDataset]):
    def _unpack_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        return batch["input"], batch["target"]
    
    def _compute_metrics(self, predictions, targets) -> dict[str, float]:
        return {"accuracy": compute_accuracy(predictions, targets)}
```

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

## Linting

```bash
uv run ruff check --fix spine_vision
```

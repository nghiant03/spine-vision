# AGENTS.md - Spine Vision

## Project Overview

Medical imaging data processing pipeline for spine analysis. The project handles:
- OCR-based extraction of patient information from medical reports
- Fuzzy matching of patient data across different data sources
- DICOM/MHA to NIfTI format conversion for nnUNet training
- Tabular data preprocessing from Excel/CSV files
- Interactive visualization of segmentation results with nnUNet inference

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
  - `plotly` - Interactive visualization for segmentation results

## Commands

### Setup
```bash
uv sync                    # Install dependencies
uv sync --group dev        # Include dev dependencies (pandas-stubs)
```

### Running Scripts
```bash
# Preprocess medical data (main pipeline)
uv run python preprocess.py [OPTIONS]

# Convert SPIDER dataset to nnUNet format
uv run python convert.py [OPTIONS]

# Visualize segmentation with nnUNet inference
uv run python visualize.py [OPTIONS]
```

### Preprocess CLI Options
| Flag | Description | Default |
|------|-------------|---------|
| `--data-path` | Input data directory | `data/silver/Phenikaa` |
| `--output-path` | Output directory | `data/gold/classfication` |
| `--model-path` | OCR model weights | `weights/ocr/` |
| `-g, --use-gpu` | Enable GPU acceleration | `True` |
| `-v, --verbose` | Debug logging | `False` |
| `--enable-file-log` | Write logs to file | `False` |
| `--report-fuzzy-threshold` | OCR matching threshold | `80` |
| `--image-fuzzy-threshold` | Folder matching threshold | `85` |

### Convert CLI Options
| Flag | Description | Default |
|------|-------------|---------|
| `--input-path` | SPIDER dataset directory | `data/silver/SPIDER/` |
| `--output-path` | nnUNet output directory | `data/gold/segmentation/Dataset501_Spider` |
| `--dataset-name` | Dataset name | `Spider` |
| `--channel-name` | Channel name in dataset.json | `MRI` |
| `--file-extension` | Input file extension | `.mha` |
| `-v, --verbose` | Debug logging | `False` |

### Visualize CLI Options
| Flag | Description | Default |
|------|-------------|---------|
| `--input-path` | DICOM input directory | `data/gold/classification/images` |
| `--output-path` | Inference output directory | `results/segmentation` |
| `--model-path` | nnUNet model weights | `weights/nnunet` |
| `--dataset-id` | nnUNet dataset ID | `501` |
| `--configuration` | nnUNet configuration | `3d_fullres` |
| `--fold` | Model fold to use | `0` |
| `--save-probabilities` | Save prediction probabilities | `True` |
| `-v, --verbose` | Debug logging | `False` |

## Project Structure

```
spine-vision/
├── config.py          # Pydantic configs (PreprocessConfig, ConvertConfig, VisualizeConfig)
├── preprocess.py      # Main OCR + fuzzy matching pipeline
├── convert.py         # SPIDER → nnUNet converter
├── visualize.py       # Interactive segmentation visualization with nnUNet inference
├── data/              # (gitignored) Input/output data
│   ├── raw/           # Raw input data (DICOM, SPIDER)
│   ├── silver/        # Intermediate processed data
│   │   ├── images/    # Patient image folders
│   │   └── labels/
│   │       ├── reports/  # Medical report PNGs
│   │       └── tables/   # Excel/CSV label files
│   ├── gold/          # Final processed output
│   │   ├── images/    # Matched patient images
│   │   └── radiological_labels.csv
│   └── inference/     # nnUNet inference output
└── weights/           # (gitignored) Model weights
    ├── ocr/           # VietOCR weights
    └── nnunet/        # nnUNet trained models
```

## Code Patterns

### Configuration
All scripts use Pydantic `BaseModel` with `computed_field` for derived paths:
```python
class PreprocessConfig(BaseModel):
    data_path: Path = Path.cwd() / "data/silver/"
    
    @computed_field
    @property
    def image_path(self) -> Path:
        return self.data_path / "images"

class ConvertConfig(BaseModel):
    input_path: Path = Path.cwd() / "SPIDER"
    
    @computed_field
    @property
    def input_images_path(self) -> Path:
        return self.input_path / "images"
```

CLI arguments parsed with `tyro`:
```python
config = tyro.cli(PreprocessConfig)  # or ConvertConfig, VisualizeConfig
```

### Logging
Uses `loguru` with `tqdm` integration for progress bars:
```python
from loguru import logger
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, ...)
```

### OCR Pipeline Pattern
1. Detect text regions with `TextDetection` (PaddleOCR)
2. Crop polygons with perspective transform (`crop_polygon`)
3. Recognize Vietnamese text with `vietocr.Predictor`
4. Extract fields with fuzzy matching (`fuzzy_value_extract`)

### Fuzzy Matching
Uses `rapidfuzz.fuzz.partial_ratio` for flexible string matching:
- Report field extraction: threshold ~80
- Image folder matching: threshold ~85

### Data Processing
- Patient IDs link reports → tabular data → image folders
- One-hot encoding for `Modic` column with `&` separator
- Corrupted IDs filtered via config

## Data Conventions

### Image Folder Naming
Pattern: `PATIENT_NAME_YYYY_YYYYMMDD( (N))?`
- `YYYY` (optional): Birth year
- `YYYYMMDD`: Study date
- `( (N))` (optional): Duplicate counter

Regex: `^[A-Z_]+(_\d{4})?_\d{8}( \(\d+\))?$`

### Label Mapping (convert.py)
SPIDER dataset labels remapped to contiguous integers:
- `1-25` → Vertebrae (kept as-is)
- `100` → Spinal Canal (→ 26)
- `201-225` → Discs (→ 27-51)

## Gotchas

1. **NumPy version**: Requires `numpy<2` for compatibility with medical imaging libraries
2. **GPU dependencies**: Both `paddlepaddle` and `paddlepaddle-gpu` are listed - may need adjustment for specific CUDA versions
3. **Vietnamese OCR**: Uses VietOCR with `vgg_transformer` model - requires model weights in `weights/ocr/`
4. **Data paths**: All data and weights are gitignored - must be provided separately
5. **SPIDER conversion**: Default paths can be overridden via CLI flags (`--input-path`, `--output-path`)
6. **Birthday parsing**: Expects `DD/MM/YYYY` format in OCR output
7. **Text normalization**: Uses `unidecode` to convert Vietnamese to ASCII for matching

## Testing

No test suite currently configured. When adding tests:
- Consider using `pytest`
- Mock OCR models for unit tests
- Use sample data fixtures for integration tests

## Type Checking

Dev dependencies include `pandas-stubs` for pandas type hints. Run type checking with:
```bash
uv run pyright preprocess.py convert.py visualize.py config.py
```

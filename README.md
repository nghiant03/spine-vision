# Spine Vision

Medical imaging pipeline for lumbar spine MRI analysis and radiological grading prediction.

## Features

- **OCR Extraction** - Extract patient information from Vietnamese medical reports using PaddleOCR + VietOCR
- **Fuzzy Matching** - Match patient data across different data sources with configurable thresholds
- **Format Conversion** - Convert DICOM/MHA to NIfTI format for nnU-Net training
- **Dataset Pipelines** - Create nnU-Net, IVD coordinates, and RSNA datasets
- **Segmentation Inference** - Run nnU-Net inference with interactive visualization
- **Interactive Visualization** - Batch-capable Plotly viewer with HTML/image export

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
```

## Quick Start

### CLI Commands

```bash
# Convert dataset to nnU-Net format
spine-vision dataset nnunet --input-path data/raw/SPIDER --output-path data/nnunet

# Create IVD coordinates dataset
spine-vision dataset ivd-coords --base-path data

# Preprocess Phenikaa dataset (OCR + patient matching)
spine-vision dataset phenikaa --data-path data/raw/Phenikaa

# Visualize segmentation results
spine-vision visualize --input-path data/images --output-mode html
```

### Python API

```python
from pathlib import Path
from spine_vision.io import read_medical_image, write_medical_image
from spine_vision.inference import NNUNetSegmentation
from spine_vision.visualization import PlotlyViewer

# Load and process medical image
image = read_medical_image(Path("input.nii.gz"))

# Run nnU-Net segmentation
model = NNUNetSegmentation(
    model_path=Path("weights/"),
    dataset_id=501,
    configuration="3d_fullres",
)
result = model.predict(image)

# Visualize results
viewer = PlotlyViewer(output_path=Path("results/"), output_mode="html")
viewer.visualize(image, result.prediction, title="Segmentation")
```

## CLI Reference

### `spine-vision dataset nnunet`

Convert datasets to nnU-Net format.

| Option | Description | Default |
|--------|-------------|---------|
| `--input-path` | Source dataset directory | `data/raw/SPIDER` |
| `--output-path` | nnU-Net output directory | `data/processed/SPIDER/Dataset501_Spider` |
| `--schema-path` | Label schema YAML | Built-in `spider` |
| `--channel-name` | Channel name in dataset.json | `MRI` |
| `--file-extension` | Input file extension | `.mha` |
| `-v, --verbose` | Debug logging | `False` |

### `spine-vision dataset ivd-coords`

Create IVD coordinates dataset.

| Option | Description | Default |
|--------|-------------|---------|
| `--base-path` | Base data directory | `data` |
| `--output-name` | Output dataset folder name | `ivd_coords` |
| `--include-neural-foraminal` | Include Neural Foraminal annotations | `True` |
| `--include-spinal-canal` | Include Spinal Canal annotations | `True` |
| `--skip-invalid-instances` | Skip records with invalid instance numbers | `True` |
| `-v, --verbose` | Debug logging | `False` |

### `spine-vision dataset phenikaa`

Preprocess Phenikaa dataset with OCR and patient matching.

| Option | Description | Default |
|--------|-------------|---------|
| `--data-path` | Input data directory | `data/raw/Phenikaa` |
| `--output-path` | Output directory | `data/processed/classification` |
| `-g, --use-gpu` | Enable GPU acceleration | `True` |
| `--report-fuzzy-threshold` | OCR matching threshold | `80` |
| `--image-fuzzy-threshold` | Folder matching threshold | `85` |
| `-v, --verbose` | Debug logging | `False` |

### `spine-vision visualize`

Visualize segmentation results with nnU-Net inference.

| Option | Description | Default |
|--------|-------------|---------|
| `--input-path` | DICOM input directory | - |
| `--output-path` | Output directory | `results/segmentation` |
| `--model-path` | nnU-Net model path | - |
| `--dataset-id` | nnU-Net dataset ID | `501` |
| `--configuration` | nnU-Net configuration | `2d` |
| `--fold` | Model fold | `0` |
| `--output-mode` | Output format (`browser`, `html`, `image`) | `html` |
| `--batch` | Enable batch processing | `False` |
| `-v, --verbose` | Debug logging | `False` |

## Project Structure

```
spine-vision/
├── spine_vision/
│   ├── core/           # Logging, config, pipeline utilities
│   ├── io/             # Medical image I/O, tabular data, RSNA utils
│   ├── ocr/            # PaddleOCR detection + VietOCR recognition
│   ├── matching/       # Fuzzy string matching, patient matching
│   ├── inference/      # nnU-Net segmentation, localization, cropping
│   ├── labels/         # Label schemas and remapping
│   ├── datasets/       # Dataset creation pipelines
│   ├── visualization/  # Plotly interactive viewer
│   └── cli/            # Unified CLI entry point
├── configs/            # YAML config files
├── data/               # Data directories (gitignored)
├── weights/            # Model weights (gitignored)
└── pyproject.toml
```

## Requirements

- Python 3.11+
- NumPy < 2 (compatibility with medical imaging libraries)
- CUDA-capable GPU recommended for OCR and inference

## Type Checking

```bash
uv run pyright spine_vision
```

## License

MIT

"""Label constants and metadata for spine classification datasets.

Centralizes all label-related constants used across the codebase:
- IVD level mappings
- Available classification labels
- Label type information (binary vs multiclass)
- Display names and colors for visualization
"""

from typing import Any

# IVD level mapping (L1/L2 to L5/S1)
LEVEL_TO_IDX: dict[str, int] = {
    "L1/L2": 0,
    "L2/L3": 1,
    "L3/L4": 2,
    "L4/L5": 3,
    "L5/S1": 4,
}
IDX_TO_LEVEL: dict[int, str] = {v: k for k, v in LEVEL_TO_IDX.items()}

# All available classification labels
AVAILABLE_LABELS: tuple[str, ...] = (
    "pfirrmann",
    "modic",
    "herniation",
    "bulging",
    "upper_endplate",
    "lower_endplate",
    "spondy",
    "narrowing",
)

# Mapping from label names to their types and classes
LABEL_INFO: dict[str, dict[str, Any]] = {
    "pfirrmann": {"type": "multiclass", "num_classes": 5},
    "modic": {"type": "multiclass", "num_classes": 4},
    "herniation": {"type": "binary", "num_classes": 1},
    "bulging": {"type": "binary", "num_classes": 1},
    "upper_endplate": {"type": "binary", "num_classes": 1},
    "lower_endplate": {"type": "binary", "num_classes": 1},
    "spondy": {"type": "binary", "num_classes": 1},
    "narrowing": {"type": "binary", "num_classes": 1},
}

# Label display names for visualization
LABEL_DISPLAY_NAMES: dict[str, str] = {
    "pfirrmann": "Pfirrmann",
    "modic": "Modic",
    "herniation": "Herniation",
    "bulging": "Bulging",
    "upper_endplate": "Upper Endplate",
    "lower_endplate": "Lower Endplate",
    "spondy": "Spondylolisthesis",
    "narrowing": "Narrowing",
}

# Color palette for labels
LABEL_COLORS: dict[str, str] = {
    "pfirrmann": "#1f77b4",  # Blue
    "modic": "#ff7f0e",  # Orange
    "herniation": "#2ca02c",  # Green
    "bulging": "#d62728",  # Red
    "upper_endplate": "#9467bd",  # Purple
    "lower_endplate": "#8c564b",  # Brown
    "spondy": "#e377c2",  # Pink
    "narrowing": "#7f7f7f",  # Gray
}

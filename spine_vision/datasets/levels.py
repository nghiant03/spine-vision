"""IVD level constants for spine datasets.

Provides mappings between IVD level names and indices.
For task-related constants (labels, types, etc.), use spine_vision.core.tasks.
"""

# IVD level mapping (L1/L2 to L5/S1)
LEVEL_TO_IDX: dict[str, int] = {
    "L1/L2": 0,
    "L2/L3": 1,
    "L3/L4": 2,
    "L4/L5": 3,
    "L5/S1": 4,
}
IDX_TO_LEVEL: dict[int, str] = {v: k for k, v in LEVEL_TO_IDX.items()}

LEVEL_NAMES: tuple[str, ...] = tuple(LEVEL_TO_IDX.keys())
NUM_LEVELS: int = len(LEVEL_NAMES)

__all__ = [
    "LEVEL_TO_IDX",
    "IDX_TO_LEVEL",
    "LEVEL_NAMES",
    "NUM_LEVELS",
]

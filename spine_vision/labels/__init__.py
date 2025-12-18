"""Label mapping and schema management."""

from spine_vision.labels.mapping import (
    LabelSchema,
    generate_nnunet_labels,
    load_label_schema,
    remap_labels,
)

__all__ = [
    "LabelSchema",
    "load_label_schema",
    "remap_labels",
    "generate_nnunet_labels",
]

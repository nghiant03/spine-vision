"""Label mapping and schema management."""

from spine_vision.labels.mapping import (
    LabelSchema,
    load_label_schema,
    remap_labels,
    generate_nnunet_labels,
)

__all__ = [
    "LabelSchema",
    "load_label_schema",
    "remap_labels",
    "generate_nnunet_labels",
]

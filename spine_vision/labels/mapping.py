"""Label mapping and remapping utilities."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger


@dataclass
class LabelSchema:
    """Schema defining label mappings for a dataset.

    Attributes:
        name: Dataset name.
        source_labels: Mapping of source label names to IDs.
        target_labels: Mapping of target label names to IDs.
        mapping: Mapping from source IDs to target IDs.
        metadata: Additional dataset-specific information.
    """

    name: str
    source_labels: dict[str, int]
    target_labels: dict[str, int]
    mapping: dict[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "LabelSchema":
        """Load schema from a YAML file.

        Args:
            yaml_path: Path to YAML schema file.

        Returns:
            LabelSchema instance.
        """
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        source_labels = data.get("source_labels", {})
        target_labels = data.get("target_labels", {})

        mapping = {}
        for src_name, src_id in source_labels.items():
            if src_name in target_labels:
                mapping[src_id] = target_labels[src_name]
            elif src_id in data.get("mapping", {}):
                mapping[src_id] = data["mapping"][src_id]

        if "mapping" in data:
            for src_id, tgt_id in data["mapping"].items():
                mapping[int(src_id)] = int(tgt_id)

        return cls(
            name=data.get("name", yaml_path.stem),
            source_labels=source_labels,
            target_labels=target_labels,
            mapping=mapping,
            metadata=data.get("metadata", {}),
        )

    def to_yaml(self, yaml_path: Path) -> None:
        """Save schema to a YAML file.

        Args:
            yaml_path: Output path for YAML file.
        """
        data = {
            "name": self.name,
            "source_labels": self.source_labels,
            "target_labels": self.target_labels,
            "mapping": self.mapping,
            "metadata": self.metadata,
        }

        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    def get_target_name(self, target_id: int) -> str | None:
        """Get the name of a target label by ID."""
        for name, id_ in self.target_labels.items():
            if id_ == target_id:
                return name
        return None


def load_label_schema(schema_path: Path | str) -> LabelSchema:
    """Load a label schema from file.

    Args:
        schema_path: Path to YAML schema file, or name of built-in schema.

    Returns:
        LabelSchema instance.
    """
    schema_path = Path(schema_path)

    if not schema_path.exists():
        # Check datasets/schemas for built-in schemas
        datasets_schema_path = (
            Path(__file__).parent.parent
            / "datasets"
            / "schemas"
            / f"{schema_path.stem}.yaml"
        )
        if datasets_schema_path.exists():
            schema_path = datasets_schema_path
        else:
            raise FileNotFoundError(f"Schema not found: {schema_path}")

    logger.debug(f"Loading label schema from {schema_path}")
    return LabelSchema.from_yaml(schema_path)


def remap_labels(
    mask_array: np.ndarray,
    mapping: dict[int, int],
    default: int = 0,
) -> np.ndarray:
    """Remap label values in a segmentation mask.

    Args:
        mask_array: Input mask as numpy array.
        mapping: Dictionary mapping source IDs to target IDs.
        default: Default value for unmapped labels.

    Returns:
        Remapped mask array.
    """
    remapped = np.full_like(mask_array, default)

    for old_label, new_label in mapping.items():
        remapped[mask_array == old_label] = new_label

    return remapped


def generate_nnunet_labels(schema: LabelSchema) -> dict[str, int]:
    """Generate nnU-Net compatible label dictionary.

    Args:
        schema: LabelSchema instance.

    Returns:
        Dictionary in nnU-Net dataset.json format.
    """
    return {"background": 0, **schema.target_labels}

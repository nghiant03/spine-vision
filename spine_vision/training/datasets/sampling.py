"""Weighted sampling utilities for handling class imbalance.

Provides functions for creating weighted samplers that oversample
minority classes during training.
"""

from collections import Counter
from typing import TYPE_CHECKING

from torch.utils.data import WeightedRandomSampler

if TYPE_CHECKING:
    from spine_vision.training.datasets.classification import ClassificationDataset


def create_weighted_sampler(
    dataset: "ClassificationDataset",
    target_label: str,
) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler for handling class imbalance.

    Computes sample weights based on inverse class frequency (1 / N_c) where N_c
    is the count of samples in class c. This ensures that samples from minority
    classes are sampled more frequently during training.

    Args:
        dataset: The ClassificationDataset to sample from.
        target_label: The label key to use for computing class weights.
            For multiclass labels (pfirrmann, modic), uses the class index.
            For binary labels, uses 0/1 values.

    Returns:
        WeightedRandomSampler with replacement=True for balanced sampling.

    Raises:
        ValueError: If target_label is not valid.

    Example:
        >>> dataset = ClassificationDataset(data_path, split="train")
        >>> sampler = create_weighted_sampler(dataset, "pfirrmann")
        >>> loader = DataLoader(dataset, sampler=sampler, batch_size=32)
    """
    # Map label name to record key
    label_to_record_key = {
        "pfirrmann": "pfirrmann",
        "modic": "modic",
        "herniation": "herniation",
        "bulging": "bulging",
        "upper_endplate": "upper_endplate",
        "lower_endplate": "lower_endplate",
        "spondy": "spondylolisthesis",
        "narrowing": "narrowing",
    }

    if target_label not in label_to_record_key:
        raise ValueError(
            f"Invalid target_label: {target_label}. "
            f"Valid labels: {list(label_to_record_key.keys())}"
        )

    record_key = label_to_record_key[target_label]

    # Extract target values for every sample
    # For pfirrmann, values are 1-5 in records, convert to 0-4 for consistency
    if target_label == "pfirrmann":
        target_values = [r[record_key] - 1 for r in dataset.records]
    else:
        target_values = [r[record_key] for r in dataset.records]

    # Compute class counts
    class_counts = Counter(target_values)

    # Compute weight per class: 1 / N_c
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}

    # Map weights to every sample
    sample_weights = [class_weights[val] for val in target_values]

    # Create and return the sampler
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

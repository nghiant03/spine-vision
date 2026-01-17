"""Evaluate CLI for trained classification models.

Evaluates a trained model on the test dataset with optional visualization
and trackio logging. Generates test sample visualizations with labels overlaid.
"""

from pathlib import Path

from loguru import logger

from spine_vision.core import BaseConfig, setup_logger


class EvaluateConfig(BaseConfig):
    """Configuration for model evaluation on test set.

    Supports classification task with visualization and trackio logging.
    """

    # Required parameters
    model_path: Path
    """Path to the trained model checkpoint (.pt file)."""

    data_path: Path = Path("data/processed/classification")
    """Path to the classification dataset."""

    # Model configuration
    backbone: str = "resnet50"
    """Backbone architecture (see BackboneFactory for options)."""

    # Dataset configuration
    output_size: tuple[int, int] = (128, 128)
    """Input size to model (H, W)."""

    levels: list[str] | None = None
    """Filter to specific IVD levels (e.g., ["L4/L5", "L5/S1"])."""

    series_types: list[str] | None = None
    """Filter to specific series types (e.g., ["sag_t2"] for T2 only).

    If None, requires both T1 and T2 images.
    Options: "sag_t1", "sag_t2".
    """

    target_labels: list[str] | None = None
    """Filter to specific labels. If None, evaluates all labels."""

    # Visualization options
    visualize: bool = True
    """Generate confusion matrix with samples visualizations."""

    max_samples_per_cell: int = 4
    """Maximum samples to display per confusion matrix cell."""

    output_path: Path | None = None
    """Output directory for visualizations. Defaults to model directory."""

    # Trackio options
    use_trackio: bool = False
    """Log evaluation results and visualizations to trackio."""

    trackio_project: str = "spine-vision"
    """Trackio project name."""

    trackio_run_name: str | None = None
    """Trackio run name. Defaults to 'eval-<model_name>'."""

    # Inference parameters
    device: str = "cuda:0"
    """Device to run evaluation on."""

    batch_size: int = 32
    """Batch size for evaluation."""


def main(config: EvaluateConfig) -> dict[str, float]:
    """Run model evaluation on test set.

    Args:
        config: Evaluation configuration.

    Returns:
        Dictionary of evaluation metrics.
    """
    import torch

    from spine_vision.datasets.labels import AVAILABLE_LABELS
    from spine_vision.training.datasets.classification import ClassificationDataset
    from spine_vision.training.metrics import MTLClassificationMetrics
    from spine_vision.training.models import MultiTaskClassifier, TaskConfig
    from spine_vision.visualization import (
        TrainingVisualizer,
        load_classification_original_images,
    )

    setup_logger(verbose=config.verbose)

    logger.info(f"Evaluating model: {config.model_path}")
    logger.info(f"Data path: {config.data_path}")

    # Determine output path
    output_path = config.output_path or config.model_path.parent / "evaluation"
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize trackio if enabled
    trackio_run = None
    if config.use_trackio:
        try:
            import trackio

            run_name = config.trackio_run_name or f"eval-{config.model_path.stem}"
            trackio_run = trackio.init(
                project=config.trackio_project,
                name=run_name,
                config={
                    "model_path": str(config.model_path),
                    "data_path": str(config.data_path),
                    "backbone": config.backbone,
                    "output_size": config.output_size,
                    "levels": config.levels,
                    "target_labels": config.target_labels,
                }
            )
            logger.info(f"Trackio run initialized: {run_name}")
        except ImportError:
            logger.warning("trackio not installed. Disabling trackio logging.")
            config.use_trackio = False

    # Create test dataset
    target_labels = config.target_labels or list(AVAILABLE_LABELS)

    test_dataset = ClassificationDataset(
        data_path=config.data_path,
        split="test",
        levels=config.levels,
        series_types=config.series_types,
        target_labels=target_labels,
        output_size=config.output_size,
        augment=False,
    )

    logger.info(f"Test dataset: {len(test_dataset)} samples")
    logger.info(f"Target labels: {target_labels}")

    # Build task configs
    from spine_vision.training.trainers.classification import _ALL_TASK_CONFIGS

    tasks = [
        TaskConfig(
            name=label,
            num_classes=_ALL_TASK_CONFIGS[label].num_classes,
            task_type=_ALL_TASK_CONFIGS[label].task_type,
        )
        for label in target_labels
    ]

    # Create model
    model = MultiTaskClassifier(
        backbone=config.backbone,
        tasks=tasks,
        pretrained=False,
    )

    # Load checkpoint
    checkpoint = torch.load(config.model_path, map_location=config.device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "unknown")
        logger.info(f"Loaded model from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)

    model.to(config.device)
    model.eval()
    logger.info(f"Model loaded: {model.name}")

    # Create dataloader
    from torch.utils.data import DataLoader

    from spine_vision.training.datasets.classification import ClassificationCollator

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=ClassificationCollator(),
    )

    # Create metrics and visualizer
    metrics = MTLClassificationMetrics(target_labels=target_labels)
    visualizer = TrainingVisualizer(
        output_path=output_path,
        output_mode="html",
        use_trackio=config.use_trackio,
    )

    # Evaluate
    import numpy as np

    from spine_vision.training.datasets.classification import DynamicTargets

    all_predictions: dict[str, list[np.ndarray]] = {label: [] for label in target_labels}
    all_targets: dict[str, list[np.ndarray]] = {label: [] for label in target_labels}
    all_metadata: list[dict] = []

    metrics.reset()

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["image"].to(config.device)
            targets: DynamicTargets = batch["targets"].to(config.device)
            metadata_list = batch["metadata"]

            predictions = model(inputs)
            metrics.update(predictions, targets)

            # Collect metadata and predictions for visualization
            if config.visualize:
                all_metadata.extend(metadata_list)

                for label in target_labels:
                    if label in predictions:
                        pred_tensor = predictions[label]
                        if pred_tensor.dim() == 1:
                            pred_np = pred_tensor.cpu().numpy()
                        else:
                            pred_np = torch.softmax(pred_tensor, dim=-1).cpu().numpy()
                        all_predictions[label].extend(pred_np)

                    if label in targets:
                        target_tensor = getattr(targets, label)
                        all_targets[label].extend(target_tensor.cpu().numpy())

    # Compute final metrics
    eval_metrics = metrics.compute()

    logger.info("Evaluation Results:")
    for key, value in sorted(eval_metrics.items()):
        logger.info(f"  {key}: {value:.4f}")

    # Generate visualizations
    if config.visualize and all_metadata:
        # Load all original images for confusion analysis
        all_original_images = load_classification_original_images(
            data_path=config.data_path,
            metadata_list=all_metadata,
            output_size=config.output_size,
        )
        all_pred_arrays = {k: np.array(v) for k, v in all_predictions.items()}
        all_target_arrays = {k: np.array(v) for k, v in all_targets.items()}

        # Plot per-label metrics
        visualizer.plot_classification_metrics(
            metrics=eval_metrics,
            target_labels=target_labels,
            filename="evaluation_metrics",
        )

        # Plot confusion matrices with samples for each label
        visualizer.plot_confusion_matrices_with_samples(
            images=all_original_images,
            predictions=all_pred_arrays,
            targets=all_target_arrays,
            target_labels=target_labels,
            metadata=all_metadata,
            max_samples_per_cell=config.max_samples_per_cell,
            filename_prefix="confusion_matrix_samples",
        )

        # Plot confusion summary bar chart
        visualizer.plot_confusion_summary(
            predictions=all_pred_arrays,
            targets=all_target_arrays,
            target_labels=target_labels,
            filename="confusion_summary",
        )

        logger.info(f"Visualizations saved to: {output_path}")

    # Log to trackio
    if config.use_trackio and trackio_run is not None:
        import trackio

        trackio.log({f"eval/{k}": v for k, v in eval_metrics.items()})
        trackio.finish()

    return eval_metrics

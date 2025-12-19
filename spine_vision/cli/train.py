"""Training CLI command configuration and entry point."""


import tyro
from loguru import logger

from spine_vision.core.logging import setup_logger
from spine_vision.training.trainers.localization import (
    LocalizationConfig,
    LocalizationTrainer,
)


def main(config: LocalizationConfig) -> None:
    """Run localization training.

    Args:
        config: Training configuration.
    """
    setup_logger(verbose=config.verbose, enable_file_log=config.enable_file_log)

    logger.info("=" * 50)
    logger.info("Starting Localization Training")
    logger.info("=" * 50)
    logger.info(f"Data path: {config.data_path}")
    logger.info(f"Output path: {config.output_path}")
    logger.info(f"Model variant: {config.model_variant}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"Device: {config.device}")
    if config.use_wandb:
        logger.info(f"Wandb project: {config.wandb_project}")
        if config.wandb_run_name:
            logger.info(f"Wandb run name: {config.wandb_run_name}")

    # Create trainer and run
    trainer = LocalizationTrainer(config)
    result = trainer.train()

    logger.info("=" * 50)
    logger.info("Training Complete")
    logger.info("=" * 50)
    logger.info(f"Best epoch: {result.best_epoch + 1}")
    logger.info(f"Best MED: {result.best_metric:.4f}")
    logger.info(f"Final train loss: {result.final_train_loss:.6f}")
    logger.info(f"Final val loss: {result.final_val_loss:.6f}")
    logger.info(f"Checkpoint: {result.checkpoint_path}")

    # Run evaluation on test set
    logger.info("Running test evaluation...")
    test_metrics = trainer.evaluate()
    logger.info(f"Test MED: {test_metrics.get('med', 0):.4f}")


if __name__ == "__main__":
    config = tyro.cli(LocalizationConfig)
    main(config)

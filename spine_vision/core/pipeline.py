"""Unified pipeline orchestration for end-to-end processing."""

from pathlib import Path
from typing import Any

from loguru import logger

from spine_vision.core.logging import add_file_log, setup_logger
from spine_vision.inference.base import InferenceModel
from spine_vision.io.readers import read_medical_image
from spine_vision.io.writers import write_medical_image


class Pipeline:
    """Orchestrates the full spine analysis pipeline.

    Stages:
        1. Load DICOM/NIfTI input
        2. Run preprocessing (optional)
        3. Run inference (segmentation, localization, etc.)
        4. Generate radiological gradings
        5. Visualize results
    """

    def __init__(
        self,
        inference_model: InferenceModel | None = None,
        verbose: bool = False,
        enable_file_log: bool = False,
        log_path: Path | None = None,
    ) -> None:
        setup_logger(verbose)
        if enable_file_log:
            add_file_log(log_path)
        self.inference_model = inference_model
        self._results: dict[str, Any] = {}

    def load_image(self, input_path: Path) -> "Pipeline":
        """Load medical image from path."""
        logger.info(f"Loading image from {input_path}")
        self._results["image"] = read_medical_image(input_path)
        self._results["input_path"] = input_path
        return self

    def run_inference(self) -> "Pipeline":
        """Run inference model on loaded image."""
        if self.inference_model is None:
            raise ValueError("No inference model configured")
        if "image" not in self._results:
            raise ValueError("No image loaded. Call load_image() first.")

        logger.info(f"Running inference with {self.inference_model.name}")
        self._results["prediction"] = self.inference_model.predict(
            self._results["image"]
        )
        return self

    def save_prediction(self, output_path: Path) -> "Pipeline":
        """Save prediction to file."""
        if "prediction" not in self._results:
            raise ValueError("No prediction available. Call run_inference() first.")

        logger.info(f"Saving prediction to {output_path}")
        write_medical_image(self._results["prediction"], output_path)
        return self

    def get_results(self) -> dict[str, Any]:
        """Get all pipeline results."""
        return self._results

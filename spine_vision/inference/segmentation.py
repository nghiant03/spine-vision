"""Segmentation models including nnU-Net."""

import subprocess
import tempfile
from pathlib import Path

import SimpleITK as sitk
from loguru import logger

from spine_vision.inference.base import InferenceModel, InferenceResult


class NNUNetSegmentation(InferenceModel):
    """nnU-Net segmentation model wrapper.

    Uses nnUNetv2_predict CLI for inference.
    """

    def __init__(
        self,
        model_path: Path,
        dataset_id: int = 501,
        configuration: str = "3d_fullres",
        fold: int | str = 0,
        save_probabilities: bool = False,
        enable_tta: bool = False,
        device: str = "cuda:0",
    ) -> None:
        """Initialize nnU-Net model.

        Args:
            model_path: Path to nnU-Net model directory.
            dataset_id: nnU-Net dataset ID.
            configuration: Model configuration (2d, 3d_fullres, etc.).
            fold: Fold number or 'all'.
            save_probabilities: Whether to save probability maps.
            enable_tta: Whether to enable test-time augmentation.
            device: Device for inference.
        """
        super().__init__(model_path, device)
        self.dataset_id = dataset_id
        self.configuration = configuration
        self.fold = fold
        self.save_probabilities = save_probabilities
        self.enable_tta = enable_tta

    @property
    def name(self) -> str:
        return f"nnU-Net (Dataset {self.dataset_id}, {self.configuration})"

    def load(self) -> None:
        """nnU-Net loads on-demand via CLI, so this is a no-op."""
        self._model = True
        logger.info(f"Configured {self.name}")

    def predict(self, image: sitk.Image) -> InferenceResult:
        """Run nnU-Net prediction on a single image.

        Args:
            image: Input SimpleITK Image.

        Returns:
            InferenceResult with segmentation mask.
        """
        self._ensure_loaded()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "input"
            output_dir = tmpdir / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            input_path = input_dir / "case_0000.nii.gz"
            sitk.WriteImage(image, str(input_path))

            self._run_prediction(input_dir, output_dir)

            output_path = output_dir / "case.nii.gz"
            prediction = sitk.ReadImage(str(output_path))

            probabilities = None
            prob_path = output_dir / "case.npz"
            if self.save_probabilities and prob_path.exists():
                import numpy as np
                probabilities = np.load(prob_path)["probabilities"]

        return InferenceResult(
            prediction=prediction,
            probabilities=probabilities,
            metadata={
                "dataset_id": self.dataset_id,
                "configuration": self.configuration,
                "fold": self.fold,
            },
        )

    def predict_from_path(
        self,
        input_path: Path,
        output_path: Path,
    ) -> InferenceResult:
        """Run prediction using file paths directly.

        More efficient for batch processing as it avoids temp files.

        Args:
            input_path: Directory containing input NIfTI files.
            output_path: Directory for output predictions.

        Returns:
            InferenceResult with path to prediction.
        """
        self._ensure_loaded()

        output_path.mkdir(parents=True, exist_ok=True)
        self._run_prediction(input_path, output_path)

        pred_files = list(output_path.glob("*.nii.gz"))
        if pred_files:
            prediction = sitk.ReadImage(str(pred_files[0]))
        else:
            raise RuntimeError(f"No prediction files found in {output_path}")

        return InferenceResult(
            prediction=prediction,
            metadata={"output_path": output_path},
        )

    def _run_prediction(self, input_dir: Path, output_dir: Path) -> None:
        """Execute nnUNetv2_predict CLI command."""
        cmd = [
            "nnUNetv2_predict",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-d", str(self.dataset_id),
            "-c", self.configuration,
            "-f", str(self.fold),
        ]

        if self.save_probabilities:
            cmd.append("--save_probabilities")

        if not self.enable_tta:
            cmd.append("--disable_tta")

        logger.debug(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)


class SegmentationModel(InferenceModel):
    """Generic segmentation model base class.

    Subclass this for non-nnUNet segmentation models.
    """

    @property
    def name(self) -> str:
        return "Generic Segmentation Model"

    def load(self) -> None:
        raise NotImplementedError("Subclasses must implement load()")

    def predict(self, image: sitk.Image) -> InferenceResult:
        raise NotImplementedError("Subclasses must implement predict()")

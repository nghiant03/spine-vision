import subprocess
import warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import SimpleITK as sitk
import tyro
from loguru import logger
from tqdm.rich import tqdm
from tqdm.std import TqdmExperimentalWarning

from config import VisualizeConfig


def read_dicom_series(folder_path: Path) -> sitk.Image:
    logger.debug(f"Reading DICOM from: {folder_path}")
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(folder_path))

    if not series_ids:
        raise ValueError("No DICOM series found in directory.")

    dicom_names = reader.GetGDCMSeriesFileNames(str(folder_path), series_ids[0])
    reader.SetFileNames(dicom_names)
    return reader.Execute()


def run_nnunet_prediction(config: VisualizeConfig) -> None:
    logger.info("Running nnU-Net prediction...")
    cmd = [
        "nnUNetv2_predict",
        "-i", str(config.temp_input_path),
        "-o", str(config.temp_output_path),
        "-d", str(config.dataset_id),
        "-c", config.configuration,
        "-f", str(config.fold),
    ]

    if config.save_probabilities:
        cmd.append("--save_probabilities")

    if not config.enable_tta:
        cmd.append("--disable_tta")

    subprocess.check_call(cmd)


def normalize_image(image_array: np.ndarray, percentile: float = 99) -> np.ndarray:
    upper_bound = np.percentile(image_array, percentile)
    return np.clip(image_array, 0, upper_bound)


def create_visualization(image_sitk: sitk.Image, pred_sitk: sitk.Image) -> go.Figure:
    logger.info("Generating interactive Plotly visualization...")

    image_array = sitk.GetArrayFromImage(image_sitk)
    pred_array = sitk.GetArrayFromImage(pred_sitk)

    image_array = normalize_image(image_array)

    pred_array_float = pred_array.astype(float)
    pred_array_float[pred_array_float == 0] = np.nan

    z_dim, _, _ = image_array.shape
    mid_slice = z_dim // 2

    trace_mri = go.Heatmap(
        z=image_array[mid_slice],
        colorscale="Gray",
        showscale=False,
        name="MRI",
    )

    trace_mask = go.Heatmap(
        z=pred_array_float[mid_slice],
        colorscale="Jet",
        opacity=0.5,
        showscale=False,
        name="Mask",
        hoverinfo="z",
    )

    fig = go.Figure(data=[trace_mri, trace_mask])

    frames = [
        go.Frame(
            data=[
                go.Heatmap(z=image_array[k]),
                go.Heatmap(z=pred_array_float[k]),
            ],
            name=str(k),
        )
        for k in range(z_dim)
    ]
    fig.frames = frames

    sliders = [
        dict(
            steps=[
                dict(
                    method="animate",
                    args=[
                        [str(k)],
                        dict(
                            mode="immediate",
                            frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0),
                        ),
                    ],
                    label=str(k),
                )
                for k in range(z_dim)
            ],
            active=mid_slice,
            transition=dict(duration=0),
            x=0,
            y=0,
            currentvalue=dict(font=dict(size=12), prefix="Slice: ", xanchor="center"),
            len=1.0,
        )
    ]

    fig.update_layout(
        title=f"Spine Segmentation (Interactive) - {z_dim} Slices",
        width=700,
        height=700,
        sliders=sliders,
        yaxis=dict(scaleanchor="x", scaleratio=1, autorange="reversed"),
        xaxis=dict(constrain="domain"),
        margin=dict(l=50, r=50, t=50, b=100),
    )

    return fig


def main(config: VisualizeConfig) -> None:
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

    log_level = "DEBUG" if config.verbose else "INFO"
    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
    )

    config.temp_input_path.mkdir(parents=True, exist_ok=True)
    config.temp_output_path.mkdir(parents=True, exist_ok=True)

    image_sitk = read_dicom_series(config.input_path)
    nifti_input_path = config.temp_input_path / "test_case_0000.nii.gz"
    sitk.WriteImage(image_sitk, str(nifti_input_path))
    logger.info(f"Converted DICOM to NIfTI: {nifti_input_path}")

    run_nnunet_prediction(config)

    prediction_path = config.temp_output_path / "test_case.nii.gz"
    pred_sitk = sitk.ReadImage(str(prediction_path))

    fig = create_visualization(image_sitk, pred_sitk)
    if config.output_mode == "browser":
        fig.show()
    elif config.output_mode == "html":
        html_path = config.output_path / "visualization.html"
        fig.write_html(html_path)
        logger.info(f"Saved HTML visualization to {html_path}")
    elif config.output_mode == "image":
        img_path = config.output_path / "visualization.png"
        try:
            fig.write_image(img_path)
            logger.info(f"Saved image visualization to {img_path}")
        except Exception as e:
            logger.error(f"Failed to save image visualization: {e}")
            logger.warning("Falling back to HTML output due to image export failure")
            html_path = config.output_path / "visualization.html"
            fig.write_html(html_path)
            logger.info(f"Saved HTML visualization to {html_path}")
    else:
        logger.warning(f"Unknown output_mode {config.output_mode}, defaulting to browser")
        fig.show()

    logger.info("Visualization complete")


if __name__ == "__main__":
    visualize_config = tyro.cli(VisualizeConfig)
    main(visualize_config)

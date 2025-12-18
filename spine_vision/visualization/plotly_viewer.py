"""Interactive visualization using Plotly."""

from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go
import SimpleITK as sitk
from loguru import logger


def normalize_image(
    image_array: np.ndarray,
    percentile: float = 99,
    min_val: float = 0,
) -> np.ndarray:
    """Normalize image intensity for visualization.

    Args:
        image_array: Input image array.
        percentile: Upper percentile for clipping.
        min_val: Minimum value for clipping.

    Returns:
        Normalized image array.
    """
    upper_bound = np.percentile(image_array, percentile)
    return np.clip(image_array, min_val, upper_bound)


def create_slice_figure(
    image_array: np.ndarray,
    mask_array: np.ndarray | None = None,
    initial_slice: int | None = None,
    title: str = "Medical Image Viewer",
    image_colorscale: str = "Gray",
    mask_colorscale: str = "Jet",
    mask_opacity: float = 0.5,
    width: int = 700,
    height: int = 700,
) -> go.Figure:
    """Create an interactive slice viewer figure.

    Args:
        image_array: 3D image array (z, y, x).
        mask_array: Optional 3D segmentation mask.
        initial_slice: Initial slice index to display. Defaults to middle slice.
        title: Figure title.
        image_colorscale: Colorscale for image.
        mask_colorscale: Colorscale for mask overlay.
        mask_opacity: Opacity of mask overlay.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with slider for slice navigation.
    """
    z_dim, _, _ = image_array.shape
    mid_slice = initial_slice if initial_slice is not None else z_dim // 2

    image_array = normalize_image(image_array)

    traces = [
        go.Heatmap(
            z=image_array[mid_slice],
            colorscale=image_colorscale,
            showscale=False,
            name="Image",
        )
    ]

    if mask_array is not None:
        mask_float = mask_array.astype(float)
        mask_float[mask_float == 0] = np.nan

        traces.append(
            go.Heatmap(
                z=mask_float[mid_slice],
                colorscale=mask_colorscale,
                opacity=mask_opacity,
                showscale=False,
                name="Mask",
                hoverinfo="z",
            )
        )

    fig = go.Figure(data=traces)

    frames = []
    for k in range(z_dim):
        frame_data = [go.Heatmap(z=image_array[k])]
        if mask_array is not None:
            mask_slice = mask_array[k].astype(float)
            mask_slice[mask_slice == 0] = np.nan
            frame_data.append(go.Heatmap(z=mask_slice))
        frames.append(go.Frame(data=frame_data, name=str(k)))

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
            currentvalue=dict(
                font=dict(size=12),
                prefix="Slice: ",
                xanchor="center",
            ),
            len=1.0,
        )
    ]

    fig.update_layout(
        title=f"{title} - {z_dim} Slices",
        width=width,
        height=height,
        sliders=sliders,
        yaxis=dict(scaleanchor="x", scaleratio=1, autorange="reversed"),
        xaxis=dict(constrain="domain"),
        margin=dict(l=50, r=50, t=50, b=100),
    )

    return fig


class PlotlyViewer:
    """Interactive viewer for medical images with segmentation overlays.

    Supports single and batch visualization with multiple output formats.
    """

    def __init__(
        self,
        output_path: Path | None = None,
        output_mode: Literal["browser", "html", "image"] = "html",
        image_colorscale: str = "Gray",
        mask_colorscale: str = "Jet",
        mask_opacity: float = 0.5,
        width: int = 700,
        height: int = 700,
    ) -> None:
        """Initialize viewer.

        Args:
            output_path: Directory for saved outputs.
            output_mode: Output mode ('browser', 'html', 'image').
            image_colorscale: Colorscale for images.
            mask_colorscale: Colorscale for mask overlays.
            mask_opacity: Opacity for mask overlays.
            width: Figure width.
            height: Figure height.
        """
        self.output_path = output_path
        self.output_mode = output_mode
        self.image_colorscale = image_colorscale
        self.mask_colorscale = mask_colorscale
        self.mask_opacity = mask_opacity
        self.width = width
        self.height = height

    def visualize(
        self,
        image: sitk.Image,
        mask: sitk.Image | None = None,
        title: str = "Spine Segmentation",
        filename: str | None = None,
    ) -> go.Figure:
        """Visualize a single image with optional mask overlay.

        Args:
            image: SimpleITK image to visualize.
            mask: Optional segmentation mask.
            title: Figure title.
            filename: Output filename (without extension).

        Returns:
            Plotly Figure object.
        """
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask) if mask else None

        fig = create_slice_figure(
            image_array=image_array,
            mask_array=mask_array,
            title=title,
            image_colorscale=self.image_colorscale,
            mask_colorscale=self.mask_colorscale,
            mask_opacity=self.mask_opacity,
            width=self.width,
            height=self.height,
        )

        self._output_figure(fig, filename or "visualization")
        return fig

    def visualize_batch(
        self,
        images: list[sitk.Image],
        masks: list[sitk.Image] | None = None,
        titles: list[str] | None = None,
        filenames: list[str] | None = None,
    ) -> list[go.Figure]:
        """Visualize multiple images.

        Args:
            images: List of SimpleITK images.
            masks: Optional list of segmentation masks.
            titles: Optional list of titles.
            filenames: Optional list of filenames.

        Returns:
            List of Plotly Figure objects.
        """
        masks_list: list[sitk.Image | None] = list(masks) if masks else [None] * len(images)
        titles_list = titles or [f"Image {i}" for i in range(len(images))]
        filenames_list = filenames or [f"visualization_{i}" for i in range(len(images))]

        figures = []
        for img, mask, title, fname in zip(images, masks_list, titles_list, filenames_list):
            fig = self.visualize(img, mask, title, fname)
            figures.append(fig)

        logger.info(f"Generated {len(figures)} visualizations")
        return figures

    def _output_figure(self, fig: go.Figure, filename: str) -> None:
        """Output figure according to configured mode."""
        if self.output_mode == "browser":
            fig.show()

        elif self.output_mode == "html":
            if self.output_path:
                self.output_path.mkdir(parents=True, exist_ok=True)
                html_path = self.output_path / f"{filename}.html"
                fig.write_html(html_path)
                logger.info(f"Saved HTML: {html_path}")

        elif self.output_mode == "image":
            if self.output_path:
                self.output_path.mkdir(parents=True, exist_ok=True)
                img_path = self.output_path / f"{filename}.png"
                try:
                    fig.write_image(img_path)
                    logger.info(f"Saved image: {img_path}")
                except Exception as e:
                    logger.error(f"Image export failed: {e}")
                    html_path = self.output_path / f"{filename}.html"
                    fig.write_html(html_path)
                    logger.warning(f"Fallback to HTML: {html_path}")

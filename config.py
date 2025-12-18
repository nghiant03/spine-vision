from pathlib import Path
from typing import Annotated

import tyro
from pydantic import BaseModel, computed_field


class PreprocessConfig(BaseModel):
    data_path: Path = Path.cwd() / "data/silver/Phenikaa"
    exclude_files: list[str] = []
    id_col: str = "Patient ID"
    corrupted_ids: list[int] = [
        25001,
        250027783,
        250026093,
        250026925,
        250026665,
        250010269,
    ]
    output_path: Path = Path.cwd() / "data/gold/classification"
    output_table: str = "radiological_labels.csv"
    model_path: Path = Path.cwd() / "weights/ocr"
    detection_model: str = "PP-OCRv5_server_det"
    recognition_model: str = "vgg_transformer"
    report_fuzzy_threshold: float = 80
    image_fuzzy_threshold: float = 85
    use_gpu: Annotated[bool, tyro.conf.arg(aliases=["-g"])] = True
    verbose: Annotated[bool, tyro.conf.arg(aliases=["-v"])] = False
    enable_file_log: bool = False
    log_path: Path = Path.cwd() / "logs"

    @computed_field
    @property
    def image_path(self) -> Path:
        return self.data_path / "images"

    @computed_field
    @property
    def label_path(self) -> Path:
        return self.data_path / "labels"

    @computed_field
    @property
    def report_path(self) -> Path:
        return self.label_path / "reports"

    @computed_field
    @property
    def table_path(self) -> Path:
        return self.label_path / "tables"

    @computed_field
    @property
    def output_table_path(self) -> Path:
        return self.output_path / self.output_table

    @computed_field
    @property
    def output_image_path(self) -> Path:
        return self.output_path / "images"

    @computed_field
    @property
    def detection_model_path(self) -> Path:
        return self.model_path / self.detection_model


class ConvertConfig(BaseModel):
    input_path: Path = Path.cwd() / "data/raw/SPIDER"
    output_path: Path = Path.cwd() / "data/silver/SPIDER/Dataset501_Spider"
    dataset_name: str = "Spider"
    channel_name: str = "MRI"
    file_extension: str = ".mha"
    verbose: Annotated[bool, tyro.conf.arg(aliases=["-v"])] = False

    @computed_field
    @property
    def input_images_path(self) -> Path:
        return self.input_path / "images"

    @computed_field
    @property
    def input_masks_path(self) -> Path:
        return self.input_path / "masks"

    @computed_field
    @property
    def output_images_path(self) -> Path:
        return self.output_path / "imagesTr"

    @computed_field
    @property
    def output_labels_path(self) -> Path:
        return self.output_path / "labelsTr"


class VisualizeConfig(BaseModel):
    input_path: Path = Path.cwd() / "data/gold/classification/images/250029976/SAG T1"
    output_path: Path = Path.cwd() / "results/segmentation"
    model_path: Path = (
        Path.cwd()
        / "weights/segmentation/Dataset501_Spider/nnUNetTrainer__nnUNetPlans__2d"
    )
    dataset_id: int = 501
    configuration: str = "2d"
    fold: int = 0
    save_probabilities: bool = False
    enable_tta: bool = False
    verbose: Annotated[bool, tyro.conf.arg(aliases=["-v"])] = False
    output_mode: str = "html"

    @computed_field
    @property
    def temp_input_path(self) -> Path:
        return self.output_path / "temp_input"

    @computed_field
    @property
    def temp_output_path(self) -> Path:
        return self.output_path / "temp_prediction"

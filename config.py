from pathlib import Path

import tyro
from pydantic import BaseModel, computed_field
from typing import Annotated


class PreprocessConfig(BaseModel):
    data_path: Path = Path.cwd() / "data/silver/"
    exclude_files: list[str] = []
    id_col: str = "Patient ID"
    corrupted_ids: list[int] = [25001, 250027783, 250026093, 250026925, 250026665, 250010269]
    output_path: Path = Path("data/gold")
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

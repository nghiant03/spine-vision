from pathlib import Path
from typing import Annotated

import tyro
from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Base configuration with common fields."""

    verbose: Annotated[bool, tyro.conf.arg(aliases=["-v"])] = False
    enable_file_log: bool = False
    log_path: Path = Path.cwd() / "logs"

    model_config = ConfigDict(arbitrary_types_allowed=True)

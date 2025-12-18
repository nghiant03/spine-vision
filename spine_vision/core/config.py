"""Base configuration classes."""

from pathlib import Path
from typing import Annotated

import tyro
from pydantic import BaseModel


class BaseConfig(BaseModel):
    """Base configuration with common fields."""

    verbose: Annotated[bool, tyro.conf.arg(aliases=["-v"])] = False
    enable_file_log: bool = False
    log_path: Path = Path.cwd() / "logs"

    class Config:
        arbitrary_types_allowed = True

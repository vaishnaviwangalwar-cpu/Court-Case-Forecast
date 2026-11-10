from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException

from src import config
from src.bundle import ModelBundle


@lru_cache(maxsize=1)
def _load(directory: str) -> ModelBundle:
    return ModelBundle.load(Path(directory))


def get_bundle() -> ModelBundle:
    directory = os.getenv("CCF_ARTIFACT_DIR", str(config.ARTIFACT_DIR))
    try:
        return _load(directory)
    except FileNotFoundError:
        raise HTTPException(503, f"No trained model in {directory}. Run `python -m src.pipeline` first.")

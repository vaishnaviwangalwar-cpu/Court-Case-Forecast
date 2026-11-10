from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os  # noqa: E402

import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.bundle import DISCLAIMER, ModelBundle  # noqa: E402


def disclaimer() -> None:
    st.warning("**Responsible use.** " + DISCLAIMER)


@st.cache_resource
def load_bundle(directory: str) -> ModelBundle | None:
    try:
        return ModelBundle.load(Path(directory))
    except FileNotFoundError:
        return None


def get_bundle_or_stop() -> ModelBundle:
    b = load_bundle(os.getenv("CCF_ARTIFACT_DIR", str(config.ARTIFACT_DIR)))
    if b is None:
        st.error("No trained model found. Run `python -m src.pipeline --synthetic` first.")
        st.stop()
    if b.meta.get("synthetic_data"):
        st.info("Model trained on SYNTHETIC demo data - numbers are illustrative only.")
    return b

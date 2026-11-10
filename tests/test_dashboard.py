from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

SRC = Path(__file__).resolve().parents[1] / "src" / "dashboard"


@pytest.mark.parametrize("page", ["app.py", "pages/case_view.py", "pages/simulator.py"])
def test_pages_render_with_disclaimer(page, trained, monkeypatch):
    root, _ = trained
    monkeypatch.setenv("CCF_ARTIFACT_DIR", str(root / "artifacts"))
    at = AppTest.from_file(str(SRC / page), default_timeout=60).run()
    assert not at.exception
    assert any("Responsible use" in w.value for w in at.warning)          # disclaimer on every screen


def test_court_view(trained, monkeypatch):
    root, _ = trained
    monkeypatch.setenv("CCF_ARTIFACT_DIR", str(root / "artifacts"))
    monkeypatch.setenv("CCF_DATA_DIR", str(root / "data"))
    # config reads env at import time, so point the module-level path explicitly
    from src import config
    monkeypatch.setattr(config, "PROCESSED_DIR", root / "data" / "processed")
    at = AppTest.from_file(str(SRC / "pages/court_view.py"), default_timeout=90).run()
    assert not at.exception

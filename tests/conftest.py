import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Run the whole pipeline once on a small synthetic dataset."""
    from src import pipeline
    root = tmp_path_factory.mktemp("ccf")
    res = pipeline.run(n=4000, synthetic=True, fast=True, with_deep=False, out_dir=root / "artifacts",
                       data_dir=root / "data", docs_dir=root / "docs")
    return root, res

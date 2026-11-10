import numpy as np
import pytest

from src.evaluation import validate
from src.models.deephit import deep_available
from src.uncertainty.conformal import ConformalDelayInterval


def test_pipeline_outputs(trained):
    _, res = trained
    assert set(res["schemes"]) == {"random", "temporal", "spatial"}
    for scheme in res["schemes"].values():
        for m in scheme["models"].values():
            assert 0.5 < m["c_index"] < 1.0                      # better than chance
    assert res["baseline_comparison"] and res["landmarks"]


def test_conformal_coverage_close_to_target(trained):
    _, res = trained
    cov = res["schemes"]["random"]["models"]["cox"]["conformal"]["empirical_coverage"]
    assert cov >= 0.85


def test_conformal_unit():
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)
    c = ConformalDelayInterval(0.1).calibrate(p[:2500], y[:2500])
    assert c.report(p[2500:], y[2500:])["empirical_coverage"] >= 0.87


def test_temporal_split_has_no_future_info(trained):
    import datetime as dt
    from src.data import clean
    root, _ = trained
    df = clean.load_modelling_table(root / "data" / "processed")
    train, test = validate.temporal_split(df, int(df["filing_year"].max()))
    assert train["filing_year"].max() < test["filing_year"].min()
    cutoff = dt.date(int(df["filing_year"].max()), 1, 1)
    assert (train["decision_date"].drop_nulls() <= cutoff).all()      # nothing decided after the cutoff is visible


def test_spatial_split_disjoint(trained):
    from src.data import clean
    root, _ = trained
    df = clean.load_modelling_table(root / "data" / "processed")
    tr, te, held = validate.spatial_split(df)
    d = lambda s: {validate.district_of(c) for c in s["court_id"].unique().to_list()}
    assert not (d(tr) & d(te))


@pytest.mark.skipif(not deep_available(), reason="torch/pycox not installed")
def test_deephit():
    from src.data import clean
    from src.data.features import FeatureEncoder
    from src.data.synthetic import generate_cases
    from src.models.deephit import DeepHitModel
    df, _ = clean.clean(generate_cases(n=2000, seed=5))
    pdf = df.to_pandas()
    X = FeatureEncoder().fit(pdf).transform(pdf)
    m = DeepHitModel(epochs=3).fit(X, pdf["duration_months"], pdf["event"])
    S = m.survival(X.head(5), np.array([12.0, 24.0]))
    assert S.shape == (5, 2) and ((S >= 0) & (S <= 1)).all()

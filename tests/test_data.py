import datetime as dt

import numpy as np
import polars as pl

from src import config
from src.data import clean
from src.data.features import FeatureEncoder, delay_label, hearing_features
from src.data.synthetic import generate_cases


def test_clean_rules():
    raw = generate_cases(n=3000, seed=1)
    df, counts = clean.clean(raw)
    assert counts["final"] < counts["input"]                          # dirty rows removed
    assert df["case_id"].n_unique() == df.height                      # deduped
    assert (df["duration_months"] > 0).all()
    ev = df["event"].to_numpy()
    assert ((ev == 1) == df["decision_date"].is_not_null().to_numpy()).all()
    assert (df.filter(pl.col("decision_date").is_not_null())["decision_date"] < config.PANDEMIC_START).all()
    assert (df["filing_date"] <= config.EXTRACTION_DATE).all()
    assert df["case_type"].is_in(list(config.CASE_TYPE_SYNONYMS)).all()  # labels normalised
    assert (df["congestion_count"] >= 0).all()


def test_censoring_duration_matches_extraction():
    df, _ = clean.clean(generate_cases(n=2000, seed=2))
    cens = df.filter(pl.col("event") == 0).head(20)
    for r in cens.iter_rows(named=True):
        expected = (config.EXTRACTION_DATE - r["filing_date"]).days / config.DAYS_PER_MONTH
        assert abs(r["duration_months"] - expected) < 1e-6


def test_versioned_table_roundtrip(tmp_path):
    df, counts = clean.clean(generate_cases(n=1500, seed=3))
    clean.write_versioned(df, counts, tmp_path)
    assert clean.load_modelling_table(tmp_path).height == df.height


def test_delay_label():
    y = delay_label(np.array([10, 30, 12, 40.0]), np.array([1, 1, 0, 0]))
    assert y.tolist() == [0, 1, -1, 1]


def test_hearing_features_do_not_peek_past_origin():
    off = np.array([[1.0, 5.0, 20.0]])
    f = hearing_features(off, 6.0)
    assert f["n_hearings"][0] == 2 and f["months_since_last_hearing"][0] == 1.0
    assert hearing_features(off, 0.5)["n_hearings"][0] == 0


def test_encoder_handles_unseen_court():
    df, _ = clean.clean(generate_cases(n=2000, seed=4))
    pdf = df.to_pandas()
    enc = FeatureEncoder().fit(pdf)
    novel = pdf.head(3).copy()
    novel["court_id"] = "99_99_99"
    X = enc.transform(novel)
    assert list(X.columns) == enc.columns_ and (X["court_OTHER"] == 1).all()

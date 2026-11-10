"""Serialisable model bundle + single-case / court-level prediction helpers shared by API and dashboard."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data.clean import normalise_case_type
from src.data.features import FeatureEncoder
from src.models.base import SurvivalModel
from src.uncertainty.conformal import ConformalDelayInterval

CURVE_TIMES = np.arange(0, 121, 3.0)
DISCLAIMER = ("Statistical forecast of case DURATION only, based on historical court data. It does not predict "
              "outcomes, must not be used to judge individuals, rank judges, or influence any case, and carries "
              "uncertainty shown by the interval. Intended for research and capacity planning.")


@dataclass
class ModelBundle:
    version: str
    model: SurvivalModel
    encoder: FeatureEncoder
    conformal: ConformalDelayInterval
    meta: dict

    def save(self, directory: Path = config.ARTIFACT_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "bundle.joblib"
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(directory: Path = config.ARTIFACT_DIR) -> "ModelBundle":
        return joblib.load(directory / "bundle.joblib")

    # ------------------------------------------------------------ prediction
    def case_frame(self, cases: list[dict]) -> pd.DataFrame:
        rows = []
        for c in cases:
            fd = c["filing_date"]
            fd = dt.date.fromisoformat(fd) if isinstance(fd, str) else fd
            court_default = self.encoder.court_capacity_.get(c["court_id"], self.encoder.global_capacity_)
            rows.append({"court_id": c["court_id"], "case_type": normalise_case_type(c["case_type"]),
                         "act": str(c.get("act", "")).strip().upper(), "filing_month": fd.month, "filing_year": fd.year,
                         "congestion_count": c.get("congestion_count") if c.get("congestion_count") is not None else court_default})
            if c.get("sanctioned_strength"):
                rows[-1]["sanctioned_strength"] = c["sanctioned_strength"]
        return pd.DataFrame(rows)

    def predict(self, cases: list[dict], times=CURVE_TIMES) -> list[dict]:
        X = self.encoder.transform(self.case_frame(cases))
        S = self.model.survival(X, np.asarray(times, float))
        p = self.model.p_delay(X, config.HORIZON_MONTHS)
        lo, hi = self.conformal.interval(p)
        return [{"model_version": self.version, "times_months": list(map(float, times)), "survival": S[i].round(4).tolist(),
                 "p_delay_24m": float(p[i]), "interval": {"lower": float(lo[i]), "upper": float(hi[i]),
                                                          "alpha": self.conformal.alpha},
                 "disclaimer": DISCLAIMER} for i in range(len(cases))]

    def court_summary(self, df_pd: pd.DataFrame, sample_per_court: int = 500) -> pd.DataFrame:
        d = df_pd.sample(frac=1, random_state=config.SEED).groupby("court_id").head(sample_per_court)
        p = self.model.p_delay(self.encoder.transform(d), config.HORIZON_MONTHS)
        d = d.assign(p=p)
        out = d.groupby("court_id").agg(cases_sampled=("p", "size"), mean_p_delay=("p", "mean"),
                                        share_long_delay=("p", lambda s: float((s > 0.5).mean())))
        return out.sort_values("share_long_delay", ascending=False).reset_index()

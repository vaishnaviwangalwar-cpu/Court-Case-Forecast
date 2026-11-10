"""Landmark models: re-forecast with hearing history known at filing, first hearing, 6m, 12m."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from src.data.features import FeatureEncoder, hearing_features, hearing_offsets
from src.models.base import SurvivalModel

REMAIN_GRID = np.arange(0, 121, 1.0)


@dataclass(frozen=True)
class Landmark:
    name: str
    months: float | None  # None -> per-case time of the first hearing


LANDMARKS = (Landmark("filing", 0.0), Landmark("first_hearing", None), Landmark("6m", 6.0), Landmark("12m", 12.0))


def landmark_cohort(df: pl.DataFrame, lm: Landmark):
    """Cases still open at the landmark. Returns (pandas frame, origin months, remaining time)."""
    off = hearing_offsets(df)
    dur = df["duration_months"].to_numpy()
    if lm.months is None:
        origin = off[:, 0] if off.shape[1] else np.full(df.height, np.nan)
        keep = ~np.isnan(origin) & (dur > origin)
    else:
        origin = np.full(df.height, lm.months)
        keep = dur > origin
    sub = df.filter(pl.Series(keep))
    o, offs = origin[keep], off[keep]
    pdf = sub.to_pandas()
    pdf["_origin"] = o
    pdf["_remaining"] = sub["duration_months"].to_numpy() - o
    return pdf, offs


class LandmarkModel:
    """One survival model per landmark, trained on the at-risk cohort with remaining-time targets."""

    def __init__(self, factory, landmarks=LANDMARKS):
        self.factory, self.landmarks = factory, landmarks

    def fit(self, train: pl.DataFrame):
        self.parts_: dict[str, tuple[FeatureEncoder, SurvivalModel]] = {}
        for lm in self.landmarks:
            pdf, offs = landmark_cohort(train, lm)
            if len(pdf) < 200 or pdf["event"].sum() < 50:
                continue
            extra = hearing_features(offs, pdf["_origin"].to_numpy())
            enc = FeatureEncoder().fit(pdf, extra)
            self.parts_[lm.name] = (enc, self.factory().fit(enc.transform(pdf, extra), pdf["_remaining"], pdf["event"]))
        return self

    def p_delay(self, df: pl.DataFrame, lm: Landmark, horizon: float = 24.0):
        """P(T > horizon | still open at landmark) for each cohort case. Returns (cohort frame, p)."""
        pdf, offs = landmark_cohort(df, lm)
        enc, model = self.parts_[lm.name]
        extra = hearing_features(offs, pdf["_origin"].to_numpy())
        S = model.survival(enc.transform(pdf, extra), REMAIN_GRID)
        h = np.clip(horizon - pdf["_origin"].to_numpy(), 0, REMAIN_GRID[-1])
        idx = np.searchsorted(REMAIN_GRID, h, side="right") - 1
        return pdf, S[np.arange(len(pdf)), idx]

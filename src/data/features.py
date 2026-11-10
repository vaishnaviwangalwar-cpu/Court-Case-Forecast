"""At-filing features (leakage-safe) and hearing-history features (for landmark models)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from src import config


def delay_label(duration: np.ndarray, event: np.ndarray, horizon: float = config.HORIZON_MONTHS) -> np.ndarray:
    """1 = delayed (still open past horizon), 0 = on time, -1 = unknown (censored before horizon)."""
    y = np.full(len(duration), -1, dtype=int)
    y[(event == 1) & (duration <= horizon)] = 0
    y[duration > horizon] = 1
    return y


class FeatureEncoder:
    """Fit on training data only; unseen courts/types map to zero columns (courts -> OTHER)."""

    def __init__(self, min_court_cases: int = 100, min_category_cases: int = 200):
        self.min_court_cases, self.min_category_cases = min_court_cases, min_category_cases

    def fit(self, df: pd.DataFrame, extra: pd.DataFrame | None = None) -> "FeatureEncoder":
        counts = df["court_id"].value_counts()
        self.courts_ = sorted(counts[counts >= self.min_court_cases].index)
        ct, ac = df["case_type"].value_counts(), df["act"].value_counts()   # rare labels -> *_OTHER
        self.case_types_ = sorted(ct[ct >= self.min_category_cases].index)
        self.acts_ = sorted(ac[ac >= self.min_category_cases].index)
        self.extra_cols_ = list(extra.columns) if extra is not None else []
        # capacity proxy when sanctioned strength is absent: per-court 90th percentile congestion (train only)
        self.global_capacity_ = float(max(df["congestion_count"].quantile(0.9), 1.0))
        self.court_capacity_ = df.groupby("court_id")["congestion_count"].quantile(0.9).clip(lower=1.0).to_dict()
        self.columns_ = list(self._build(df.head(1), extra.head(1) if extra is not None else None).columns)
        return self

    def _build(self, df: pd.DataFrame, extra: pd.DataFrame | None) -> pd.DataFrame:
        n = len(df)
        out: dict[str, np.ndarray] = {}
        cong = df["congestion_count"].to_numpy(float)
        if "sanctioned_strength" in df.columns:
            cap = df["sanctioned_strength"].to_numpy(float)
        else:
            cap = df["court_id"].map(self.court_capacity_).fillna(self.global_capacity_).to_numpy(float)
        out["congestion_count"] = np.log1p(cong)
        out["court_congestion_ratio"] = cong / np.maximum(cap, 1.0)
        m = df["filing_month"].to_numpy(float)
        out["month_sin"], out["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
        q = ((m - 1) // 3 + 1).astype(int)
        for k in range(1, 5):
            out[f"filing_quarter_Q{k}"] = (q == k).astype(float)
        for c in self.case_types_:
            out[f"case_type_{c}"] = (df["case_type"].to_numpy() == c).astype(float)
        out["case_type_OTHER"] = (~np.isin(df["case_type"].to_numpy(), self.case_types_)).astype(float)
        for a in self.acts_:
            out[f"act_{a}"] = (df["act"].to_numpy() == a).astype(float)
        out["act_OTHER"] = (~np.isin(df["act"].to_numpy(), self.acts_)).astype(float)
        court = df["court_id"].to_numpy()
        for c in self.courts_:
            out[f"court_{c}"] = (court == c).astype(float)
        out["court_OTHER"] = (~np.isin(court, self.courts_)).astype(float)
        X = pd.DataFrame(out, index=df.index)
        if extra is not None:
            X = pd.concat([X, extra.reset_index(drop=True).set_index(df.index)[self.extra_cols_]], axis=1)
        return X

    def transform(self, df: pd.DataFrame, extra: pd.DataFrame | None = None) -> pd.DataFrame:
        X = self._build(df, extra)
        return X.reindex(columns=self.columns_, fill_value=0.0)

    @staticmethod
    def group_of(col: str) -> str:
        for p in ("court_", "case_type_", "act_", "filing_quarter_", "month_"):
            if col.startswith(p):
                return {"court_": "court", "case_type_": "case_type", "act_": "act",
                        "filing_quarter_": "season", "month_": "season"}[p]
        if "congestion" in col:
            return "congestion"
        return "hearing_history"


def hearing_offsets(df: pl.DataFrame) -> np.ndarray:
    """(n, K) months from filing to each hearing; NaN when absent."""
    cols = [f"hearing_{k}_date" for k in range(1, config.MAX_HEARINGS + 1) if f"hearing_{k}_date" in df.columns]
    if not cols:
        return np.full((df.height, 0), np.nan)
    arr = df.select([((pl.col(c) - pl.col("filing_date")).dt.total_days() / config.DAYS_PER_MONTH).alias(c) for c in cols])
    return arr.to_numpy().astype(float)


def hearing_features(offsets: np.ndarray, origin: np.ndarray | float) -> pd.DataFrame:
    """Information available at `origin` months after filing (never looks past it)."""
    origin = np.broadcast_to(np.asarray(origin, float), (offsets.shape[0],))[:, None]
    seen = np.where(offsets <= origin, offsets, np.nan)
    n = np.sum(~np.isnan(seen), axis=1)
    first = np.where(n > 0, np.nanmin(np.where(np.isnan(seen), np.inf, seen), axis=1), -1.0)
    last = np.where(n > 0, np.nanmax(np.where(np.isnan(seen), -np.inf, seen), axis=1), -1.0)
    gap = np.where(n > 1, (last - first) / np.maximum(n - 1, 1), -1.0)
    since_last = np.where(n > 0, origin[:, 0] - last, -1.0)
    return pd.DataFrame({"n_hearings": n.astype(float), "months_to_first_hearing": first,
                         "mean_hearing_gap": gap, "months_since_last_hearing": since_last})

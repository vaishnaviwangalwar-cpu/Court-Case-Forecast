"""Splits and metrics: C-index, time-dependent AUC, IBS, calibration, coverage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from lifelines import KaplanMeierFitter
from sklearn.metrics import brier_score_loss, roc_auc_score
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score
from sksurv.util import Surv

from src import config
from src.data.features import delay_label
from src.models.base import SurvivalModel


# ---------------------------------------------------------------- splits
def district_of(court_id: str) -> str:
    return "_".join(court_id.split("_")[:2])


def random_split(df: pl.DataFrame, test_frac: float = 0.25, seed: int = config.SEED):
    rng = np.random.default_rng(seed)
    mask = rng.random(df.height) < test_frac
    return df.filter(pl.Series(~mask)), df.filter(pl.Series(mask))


def censor_at(df: pl.DataFrame, cutoff) -> pl.DataFrame:
    """Pretend the data was extracted at `cutoff`: later decisions become censored (no future leakage)."""
    cut_months = ((pl.lit(cutoff) - pl.col("filing_date")).dt.total_days() / config.DAYS_PER_MONTH)
    late = pl.col("decision_date").is_not_null() & (pl.col("decision_date") > cutoff)
    return df.filter(pl.col("filing_date") < cutoff).with_columns(
        pl.when(late | pl.col("decision_date").is_null()).then(cut_months).otherwise(pl.col("duration_months")).alias("duration_months"),
        pl.when(late).then(0).otherwise(pl.col("event")).cast(pl.Int8).alias("event"),
    ).with_columns(
        pl.when(late).then(None).otherwise(pl.col("decision_date")).alias("decision_date"))


def temporal_split(df: pl.DataFrame, test_from_year: int):
    """Train on earlier filing years (re-censored at the test start); test on later years."""
    import datetime as dt
    cutoff = dt.date(test_from_year, 1, 1)
    train = censor_at(df.filter(pl.col("filing_year") < test_from_year), cutoff)
    return train, df.filter(pl.col("filing_year") >= test_from_year)


def spatial_split(df: pl.DataFrame, holdout_frac: float = 0.25, seed: int = config.SEED):
    districts = sorted({district_of(c) for c in df["court_id"].unique().to_list()})
    rng = np.random.default_rng(seed)
    held = set(rng.choice(districts, max(1, int(len(districts) * holdout_frac)), replace=False))
    is_held = df["court_id"].map_elements(lambda c: district_of(c) in held, return_dtype=pl.Boolean)
    return df.filter(~is_held), df.filter(is_held), sorted(held)


# ---------------------------------------------------------------- metrics
def calibration_table(p: np.ndarray, duration, event, horizon: float = 24.0, n_bins: int = 10) -> pd.DataFrame:
    """Predicted P(T>h) vs Kaplan-Meier observed (censoring-aware) in prediction deciles."""
    df = pd.DataFrame({"p": p, "t": duration, "e": event})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    rows = []
    for b, g in df.groupby("bin"):
        km = KaplanMeierFitter().fit(g["t"], g["e"])
        rows.append({"bin": int(b), "n": len(g), "predicted": g["p"].mean(),
                     "observed": float(km.survival_function_at_times(horizon).iloc[0])})
    return pd.DataFrame(rows)


def evaluate_survival(model: SurvivalModel, Xtr, tr_time, tr_event, Xte, te_time, te_event) -> dict:
    tr_time, tr_event, te_time, te_event = map(np.asarray, (tr_time, tr_event, te_time, te_event))
    # IPCW metrics need test follow-up inside the training follow-up range (matters for temporal splits,
    # where training data is re-censored at the test start). Truncate the test set at that cap.
    tau = float(tr_time.max()) - 1e-6
    te_event = np.where(te_time > tau, 0, te_event)
    te_time = np.minimum(te_time, tau)
    y_tr, y_te = Surv.from_arrays(tr_event.astype(bool), tr_time), Surv.from_arrays(te_event.astype(bool), te_time)
    res: dict = {"n_test": int(len(te_time)), "event_rate_test": float(te_event.mean()), "eval_horizon_cap_months": tau}
    res["c_index"] = float(concordance_index_censored(te_event.astype(bool), te_time, model.risk(Xte))[0])

    lo, hi = te_time[te_event == 1].min(), te_time.max()
    auc = {}
    for t in config.EVAL_TIMES:
        if lo < t < hi:
            try:
                auc[t] = float(cumulative_dynamic_auc(y_tr, y_te, 1 - model.survival(Xte, np.array([float(t)]))[:, 0], [t])[0][0])
            except ValueError:
                auc[t] = float("nan")
        else:
            auc[t] = float("nan")
    res["td_auc"] = auc

    # IPCW weights blow up where the training censoring curve is near zero (heavy administrative censoring),
    # so the IBS time range is capped at the 75th percentile of training follow-up.
    grid_hi = min(np.percentile(te_time, 90), np.percentile(tr_time, 75), hi - 1e-6)
    grid = np.linspace(max(lo, np.percentile(te_time[te_event == 1], 5)), grid_hi, 20)
    try:
        ibs = float(integrated_brier_score(y_tr, y_te, model.survival(Xte, grid), grid))
        res["ibs"] = ibs if 0 <= ibs <= 1 else float("nan")
        res["ibs_max_time_months"] = float(grid_hi)
    except ValueError:
        res["ibs"] = float("nan")

    p24 = model.p_delay(Xte, config.HORIZON_MONTHS)
    res.update(binary_delay_metrics(p24, te_time, te_event))
    return res


def binary_delay_metrics(p24: np.ndarray, duration, event) -> dict:
    y = delay_label(np.asarray(duration), np.asarray(event))
    k = y >= 0
    out = {"n_known_label": int(k.sum())}
    if k.sum() > 10 and len(set(y[k])) > 1:
        out["auc_24m"] = float(roc_auc_score(y[k], p24[k]))
        out["brier_24m"] = float(brier_score_loss(y[k], np.clip(p24[k], 0, 1)))
    return out


def performance_by_court_size(model: SurvivalModel, X: pd.DataFrame, df_pd: pd.DataFrame, n_groups: int = 3) -> pd.DataFrame:
    """C-index by court size tercile (is the model worse in small / low-data courts?)."""
    size = df_pd.groupby("court_id")["case_id"].transform("count")
    bucket = pd.qcut(size.rank(method="first"), n_groups, labels=["small", "medium", "large"][:n_groups])
    risk = model.risk(X)
    rows = []
    for b in bucket.cat.categories:
        m = (bucket == b).to_numpy()
        if m.sum() > 50 and df_pd.loc[m, "event"].sum() > 10:
            c = concordance_index_censored(df_pd.loc[m, "event"].to_numpy().astype(bool), df_pd.loc[m, "duration_months"].to_numpy(), risk[m])[0]
            rows.append({"court_size": b, "n": int(m.sum()), "c_index": float(c)})
    return pd.DataFrame(rows)

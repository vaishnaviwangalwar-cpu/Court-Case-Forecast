"""Permutation importance and feature-group ablation (SHAP optional, not required)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

from src import config
from src.data.features import FeatureEncoder


def _c(model, X, t, e):
    return concordance_index_censored(np.asarray(e).astype(bool), np.asarray(t), model.risk(X))[0]


def permutation_importance(model, X: pd.DataFrame, t, e, n_repeats: int = 3, seed: int = config.SEED) -> pd.DataFrame:
    """Drop in C-index when a feature *group* is shuffled (groups keep one-hot columns together)."""
    rng = np.random.default_rng(seed)
    base = _c(model, X, t, e)
    groups: dict[str, list[str]] = {}
    for c in X.columns:
        groups.setdefault(FeatureEncoder.group_of(c), []).append(c)
    rows = []
    for g, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            drops.append(base - _c(model, Xp, t, e))
        rows.append({"feature_group": g, "c_index_drop": float(np.mean(drops))})
    return pd.DataFrame(rows).sort_values("c_index_drop", ascending=False).reset_index(drop=True)


def group_ablation(factory, Xtr: pd.DataFrame, ttr, etr, Xte: pd.DataFrame, tte, ete) -> pd.DataFrame:
    """Retrain without each feature group. Judge features are not in the data -> documented exclusion."""
    full = _c(factory().fit(Xtr, ttr, etr), Xte, tte, ete)
    rows = [{"removed_group": "none (full model)", "c_index": float(full)}]
    groups = sorted({FeatureEncoder.group_of(c) for c in Xtr.columns})
    for g in groups:
        keep = [c for c in Xtr.columns if FeatureEncoder.group_of(c) != g]
        m = factory().fit(Xtr[keep], ttr, etr)
        rows.append({"removed_group": g, "c_index": float(_c(m, Xte[keep], tte, ete))})
    rows.append({"removed_group": "judge features", "c_index": float("nan")})  # excluded: not available in e-Courts data
    return pd.DataFrame(rows)

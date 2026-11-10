"""Replicate the published baseline: 24-month delay classifier + two duration regressors."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

from src import config
from src.data.features import delay_label


class Baseline:
    def fit(self, X, duration, event):
        X, duration, event = np.asarray(X, float), np.asarray(duration, float), np.asarray(event)
        y = delay_label(duration, event)
        known = y >= 0
        self.clf_ = HistGradientBoostingClassifier(random_state=config.SEED).fit(X[known], y[known])
        ontime = (event == 1) & (duration <= config.HORIZON_MONTHS)
        late = (event == 1) & (duration > config.HORIZON_MONTHS)
        self.reg_ontime_ = self._reg(X[ontime], duration[ontime])
        self.reg_late_ = self._reg(X[late], duration[late])
        return self

    @staticmethod
    def _reg(X, y):
        # too few resolved cases in this bucket (e.g. temporal split with short follow-up) -> no regressor
        return HistGradientBoostingRegressor(random_state=config.SEED).fit(X, y) if len(y) >= 20 else None

    def p_delay(self, X):
        return self.clf_.predict_proba(np.asarray(X, float))[:, 1]

    def evaluate(self, X, duration, event) -> dict:
        X, duration, event = np.asarray(X, float), np.asarray(duration, float), np.asarray(event)
        y = delay_label(duration, event)
        known = y >= 0
        out = {"auc_24m": float(roc_auc_score(y[known], self.p_delay(X[known]))) if len(set(y[known])) > 1 else np.nan}
        for tag, mask in (("ontime", (event == 1) & (duration <= config.HORIZON_MONTHS)),
                          ("delayed", (event == 1) & (duration > config.HORIZON_MONTHS))):
            reg = self.reg_ontime_ if tag == "ontime" else self.reg_late_
            if mask.sum() and reg is not None:
                pred = reg.predict(X[mask])
                out[f"{tag}_rmse"] = float(np.sqrt(mean_squared_error(duration[mask], pred)))
                out[f"{tag}_mae"] = float(mean_absolute_error(duration[mask], pred))
        return out


def compare_to_published(ours: dict) -> pd.DataFrame:
    rows = [{"metric": k, "published": v, "replicated": ours.get(k, np.nan), "diff": ours.get(k, np.nan) - v}
            for k, v in config.PUBLISHED_BASELINE.items()]
    return pd.DataFrame(rows)

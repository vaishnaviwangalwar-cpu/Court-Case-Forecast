from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from src.models.base import SurvivalModel, eval_step


class CoxModel(SurvivalModel):
    name = "cox"

    def __init__(self, alpha: float = 1.0, n_iter: int = 50, clip: float = 5.0):
        self.alpha, self.n_iter, self.clip = alpha, n_iter, clip

    def fit(self, X, time, event):
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = CoxPHSurvivalAnalysis(alpha=self.alpha, n_iter=self.n_iter).fit(
            self._z(X), Surv.from_arrays(np.asarray(event, bool), np.asarray(time, float)))
        return self

    def _z(self, X):
        # rare one-hot columns standardise to huge values (-> exp overflow); clip them
        return np.clip(self.scaler_.transform(X), -self.clip, self.clip)

    def survival(self, X, times):
        times = np.asarray(times, float)
        Xs = self._z(X)
        out = []
        for i in range(0, len(Xs), 5000):   # chunked: step-function objects for 100k+ cases exhaust RAM
            fns = self.model_.predict_survival_function(Xs[i:i + 5000])
            out.append(np.vstack([eval_step(f.x, f.y, times) for f in fns]))
        return np.vstack(out)

    def risk(self, X):
        return self.model_.predict(self._z(X))

from __future__ import annotations

import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from src import config
from src.models.base import SurvivalModel, eval_step


class RSFModel(SurvivalModel):
    name = "rsf"

    def __init__(self, n_estimators: int = 100, min_samples_leaf: int = 30, max_train: int = 20000):
        self.n_estimators, self.min_samples_leaf, self.max_train = n_estimators, min_samples_leaf, max_train

    def fit(self, X, time, event):
        X, time, event = np.asarray(X, float), np.asarray(time, float), np.asarray(event, bool)
        if len(X) > self.max_train:  # scalability: subsample (stratified by event) for very large tables
            rng = np.random.default_rng(config.SEED)
            idx = np.concatenate([rng.choice(np.where(event == e)[0], int(self.max_train * (event == e).mean()), replace=False)
                                  for e in (True, False) if (event == e).any()])
            X, time, event = X[idx], time[idx], event[idx]
        self.model_ = RandomSurvivalForest(n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
                                           max_features="sqrt", n_jobs=-1, random_state=config.SEED)
        self.model_.fit(X, Surv.from_arrays(event, time))
        return self

    def survival(self, X, times):
        times = np.asarray(times, float)
        X = np.asarray(X, float)
        out = []
        for i in range(0, len(X), 5000):   # chunked to keep memory flat
            fns = self.model_.predict_survival_function(X[i:i + 5000])
            out.append(np.vstack([eval_step(f.x, f.y, times) for f in fns]))
        return np.vstack(out)

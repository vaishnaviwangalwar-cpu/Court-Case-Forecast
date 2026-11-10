"""DeepHit (pycox). Optional: needs `pip install torch pycox` (extras: `.[deep]`).

NOTE: written against pycox's documented API but NOT executed in the build environment
(torch was not installed there). Run tests/test_models.py::test_deephit locally to confirm.
"""
from __future__ import annotations

import numpy as np

from src import config
from src.models.base import SurvivalModel, eval_step


def deep_available() -> bool:
    try:
        import torch  # noqa: F401
        import pycox  # noqa: F401
        return True
    except Exception:
        return False


class DeepHitModel(SurvivalModel):
    name = "deephit"

    def __init__(self, num_durations: int = 40, hidden=(64, 64), dropout: float = 0.1,
                 epochs: int = 60, lr: float = 1e-3, batch_size: int = 256):
        self.num_durations, self.hidden, self.dropout = num_durations, tuple(hidden), dropout
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size

    def fit(self, X, time, event):
        import torch
        import torchtuples as tt
        from pycox.models import DeepHitSingle
        from sklearn.preprocessing import StandardScaler

        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X).astype("float32")
        time, event = np.asarray(time, "float32"), np.asarray(event).astype("int64")
        self.labtrans_ = DeepHitSingle.label_transform(self.num_durations)
        y = self.labtrans_.fit_transform(time, event)
        rng = np.random.default_rng(config.SEED)
        val = rng.random(len(Xs)) < 0.15
        net = tt.practical.MLPVanilla(Xs.shape[1], list(self.hidden), self.labtrans_.out_features,
                                      batch_norm=True, dropout=self.dropout)
        self.model_ = DeepHitSingle(net, tt.optim.Adam(self.lr), alpha=0.2, sigma=0.1,
                                    duration_index=self.labtrans_.cuts)
        self.model_.fit(Xs[~val], (y[0][~val], y[1][~val]), self.batch_size, self.epochs,
                        callbacks=[tt.callbacks.EarlyStopping()], val_data=(Xs[val], (y[0][val], y[1][val])),
                        verbose=False)
        return self

    def survival(self, X, times):
        surv = self.model_.predict_surv_df(self.scaler_.transform(X).astype("float32"))
        cuts, arr = surv.index.to_numpy(float), surv.to_numpy().T
        times = np.asarray(times, float)
        return np.vstack([eval_step(cuts, row, times) for row in arr])

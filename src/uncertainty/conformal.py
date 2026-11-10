"""Split-conformal intervals for the 24-month delay probability.

Score = |y - p_hat| on a held-out calibration set, where y in {0,1} is the realised 24-month
delay label. The interval [p_hat - q, p_hat + q] (clipped to [0,1]) covers the realised label with
probability >= 1 - alpha, marginally, under exchangeability. Cases censored before 24 months have
no label and are excluded from calibration/coverage (a known selection effect: say so in reports).
Because y is binary the intervals are wide; that is the honest price of a guaranteed coverage.
"""
from __future__ import annotations

import numpy as np

from src import config


class ConformalDelayInterval:
    def __init__(self, alpha: float = config.CONFORMAL_ALPHA):
        self.alpha = alpha

    def calibrate(self, p_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalDelayInterval":
        scores = np.abs(np.asarray(y_cal, float) - np.asarray(p_cal, float))
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_ = float(np.quantile(scores, level, method="higher"))
        self.n_cal_ = n
        return self

    def interval(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(p, float)
        return np.clip(p - self.q_, 0, 1), np.clip(p + self.q_, 0, 1)

    def report(self, p: np.ndarray, y: np.ndarray) -> dict:
        lo, hi = self.interval(p)
        y = np.asarray(y, float)
        return {"alpha": self.alpha, "target_coverage": 1 - self.alpha,
                "empirical_coverage": float(np.mean((y >= lo) & (y <= hi))),
                "mean_interval_width": float(np.mean(hi - lo)), "n_eval": int(len(y))}

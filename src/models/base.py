"""Common interface: every survival model returns S(t|x) on an arbitrary time grid."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

RISK_GRID = np.arange(1, 97, 3.0)


def eval_step(xs: np.ndarray, ys: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Evaluate a right-continuous step function (S=1 before first knot, last value afterwards)."""
    idx = np.searchsorted(xs, times, side="right") - 1
    out = np.where(idx < 0, 1.0, ys[np.clip(idx, 0, len(ys) - 1)])
    return out


class SurvivalModel(ABC):
    name = "base"

    @abstractmethod
    def fit(self, X, time, event) -> "SurvivalModel": ...

    @abstractmethod
    def survival(self, X, times) -> np.ndarray:
        """(n_samples, len(times)) survival probabilities."""

    def p_delay(self, X, horizon: float = 24.0) -> np.ndarray:
        return self.survival(X, np.array([horizon]))[:, 0]

    def risk(self, X) -> np.ndarray:
        """Higher = expected to end sooner (negative restricted mean survival time)."""
        return -self.survival(X, RISK_GRID).mean(axis=1)

"""Baseline forecasters for comparison with Prophet.

All share the interface ``fit(series) -> self`` and ``predict(steps) -> np.ndarray``,
where ``series`` is a ``pd.Series`` of daily values in date order.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


class NaiveForecaster:
    """Last observed value carried forward."""
    name = "Naive (last value)"

    def fit(self, y: pd.Series):
        self.last_value = float(y.iloc[-1])
        return self

    def predict(self, steps: int) -> np.ndarray:
        return np.full(steps, self.last_value)


class MovingAverageForecaster:
    """Mean of the last ``window`` observations carried forward."""

    def __init__(self, window: int = 7):
        self.window = window
        self.name = f"Moving average ({window}-day)"

    def fit(self, y: pd.Series):
        self.level = float(y.iloc[-self.window:].mean())
        return self

    def predict(self, steps: int) -> np.ndarray:
        return np.full(steps, self.level)


class SeasonalNaiveForecaster:
    """Value from one season ago (``season=7`` weekly, ``season=365`` yearly)."""

    def __init__(self, season: int = 365):
        self.season = season
        self.name = f"Seasonal naive ({season}-day)"

    def fit(self, y: pd.Series):
        if len(y) < self.season:
            raise ValueError(f"Need at least {self.season} observations")
        self.history = y.to_numpy(dtype=float)
        return self

    def predict(self, steps: int) -> np.ndarray:
        last_season = self.history[-self.season:]
        reps = int(np.ceil(steps / self.season))
        return np.tile(last_season, reps)[:steps]


class ARIMAForecaster:
    """statsmodels ARIMA(p, d, q)."""

    def __init__(self, order=(2, 1, 2)):
        self.order = order
        self.name = f"ARIMA{order}"

    def fit(self, y: pd.Series):
        from statsmodels.tsa.arima.model import ARIMA

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result = ARIMA(y.to_numpy(dtype=float), order=self.order).fit()
        return self

    def predict(self, steps: int) -> np.ndarray:
        return np.asarray(self.result.forecast(steps=steps), dtype=float)

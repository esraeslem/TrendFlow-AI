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


class LightGBMForecaster:
    """Gradient-boosted trees on calendar + lag features (direct multi-step).

    Every lag is at least ``horizon`` days old, so all features for the next
    ``horizon`` days are known at forecast time: no recursive feeding of
    predictions. Uses quantile-free point forecasts (objective L2).
    """

    def __init__(self, horizon: int = 30, seed: int = 42, **params):
        self.horizon = horizon
        self.name = "LightGBM (calendar + lags)"
        self.params = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
                           min_child_samples=10, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=seed, verbose=-1)
        self.params.update(params)

    def _features(self, dates: pd.DatetimeIndex, values: pd.Series) -> pd.DataFrame:
        h = self.horizon
        s = values.reindex(dates)
        f = pd.DataFrame(index=dates)
        f["dow"] = dates.dayofweek
        f["month"] = dates.month
        f["doy"] = dates.dayofyear
        f["t"] = (dates - self.origin).days
        f["lag_h"] = s.shift(h).to_numpy()
        f["lag_h7"] = s.shift(h + 7).to_numpy()
        f["lag_364"] = s.shift(364).to_numpy()
        f["roll7_h"] = s.shift(h).rolling(7).mean().to_numpy()
        f["roll28_h"] = s.shift(h).rolling(28).mean().to_numpy()
        return f

    def fit(self, y: pd.Series):
        """``y`` must be indexed by a daily DatetimeIndex."""
        from lightgbm import LGBMRegressor

        if not isinstance(y.index, pd.DatetimeIndex):
            raise TypeError("LightGBMForecaster.fit needs a Series indexed by dates")
        self.history = y.astype(float)
        self.origin = y.index.min()
        X = self._features(y.index, self.history)
        mask = X[["lag_h", "roll28_h"]].notna().all(axis=1)
        self.model = LGBMRegressor(**self.params).fit(X[mask], y[mask])
        return self

    def predict(self, steps: int) -> np.ndarray:
        if steps > self.horizon:
            raise ValueError(f"Trained for horizon {self.horizon}, asked for {steps}")
        last = self.history.index.max()
        full_idx = pd.date_range(self.origin, last + pd.Timedelta(days=steps))
        X = self._features(full_idx, self.history).iloc[-steps:]
        return np.clip(self.model.predict(X), 0, None)

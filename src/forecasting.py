"""Prophet wrapper used by the dashboard, API, benchmark and README example."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

# Prophet/cmdstanpy are very chatty at INFO level
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def _to_prophet_frame(df: pd.DataFrame, date_col: str | None, value_col: str | None) -> pd.DataFrame:
    if {"ds", "y"}.issubset(df.columns):
        out = df[["ds", "y"]].copy()
    else:
        date_col = date_col or ("Date" if "Date" in df.columns else "date")
        value_col = value_col or ("Sales" if "Sales" in df.columns else "sales")
        missing = {date_col, value_col} - set(df.columns)
        if missing:
            raise KeyError(f"Missing columns {missing}; pass date_col/value_col explicitly")
        out = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
    out["ds"] = pd.to_datetime(out["ds"])
    if out["ds"].duplicated().any():
        raise ValueError("Duplicate dates found - filter to a single product/series first "
                         "(e.g. df[df['Product'] == 'Floral Dress']) or aggregate by date.")
    return out.sort_values("ds").reset_index(drop=True)


class ProphetForecaster:
    """Thin, reproducible wrapper around ``prophet.Prophet`` for one daily series."""

    def __init__(self, yearly_seasonality=True, weekly_seasonality=True,
                 changepoint_prior_scale: float = 0.05, interval_width: float = 0.95,
                 uncertainty_samples: int = 1000, seed: int = 42):
        self.params = dict(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=False,  # data is daily; intra-day seasonality is meaningless
            changepoint_prior_scale=changepoint_prior_scale,
            interval_width=interval_width,
            uncertainty_samples=uncertainty_samples,
        )
        self.seed = seed
        self.model = None
        self.history: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame, date_col: str | None = None, value_col: str | None = None):
        from prophet import Prophet  # imported lazily: heavy import

        # cmdstanpy resets its level on import, so silence it here
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        self.history = _to_prophet_frame(df, date_col, value_col)
        self.model = Prophet(**self.params)
        self.model.fit(self.history)
        return self

    def _future(self, periods: int) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Call fit() first")
        if periods < 1:
            raise ValueError("periods must be >= 1")
        future = self.model.make_future_dataframe(periods=periods, include_history=False)
        return future

    def predict(self, periods: int = 30) -> pd.DataFrame:
        """Daily forecast for the next ``periods`` days (future rows only)."""
        future = self._future(periods)
        np.random.seed(self.seed)
        fc = self.model.predict(future)
        out = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        out[["yhat", "yhat_lower", "yhat_upper"]] = out[["yhat", "yhat_lower", "yhat_upper"]].clip(lower=0)
        return out.reset_index(drop=True)

    def predict_in_sample(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Call fit() first")
        np.random.seed(self.seed)
        return self.model.predict(self.history[["ds"]])

    def daily_demand_samples(self, periods: int = 30) -> np.ndarray:
        """Posterior predictive draws of daily demand, shape (periods, uncertainty_samples)."""
        future = self._future(periods)
        np.random.seed(self.seed)
        return np.clip(self.model.predictive_samples(future)["yhat"], 0, None)

    def total_demand_samples(self, periods: int = 30) -> np.ndarray:
        """Draws of *total* demand over the next ``periods`` days.

        Uses Prophet's posterior predictive samples (trend + observation noise),
        summed across days within each draw, so day-to-day correlation of the
        trend uncertainty is kept. Summing daily interval bounds instead would
        overstate the uncertainty of the total.
        """
        return self.daily_demand_samples(periods).sum(axis=0)

    def forecast_total_demand(self, periods: int = 30) -> dict:
        """Summary of total demand over the horizon: mean, std, interval and raw samples."""
        s = self.total_demand_samples(periods)
        alpha = (1 - self.params["interval_width"]) / 2
        return {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)),
            "lower": float(np.quantile(s, alpha)),
            "upper": float(np.quantile(s, 1 - alpha)),
            "samples": s,
        }

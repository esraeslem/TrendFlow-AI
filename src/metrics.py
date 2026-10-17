"""Forecast-accuracy and inventory-outcome metrics."""

from __future__ import annotations

import numpy as np


def _arrays(y_true, y_pred):
    a, b = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("Empty input")
    return a, b


def mae(y_true, y_pred) -> float:
    a, b = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def rmse(y_true, y_pred) -> float:
    a, b = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error in %, ignoring periods with zero actuals."""
    a, b = _arrays(y_true, y_pred)
    mask = a != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100)


def r2(y_true, y_pred) -> float:
    a, b = _arrays(y_true, y_pred)
    ss_tot = np.sum((a - a.mean()) ** 2)
    return float(1 - np.sum((a - b) ** 2) / ss_tot) if ss_tot > 0 else float("nan")


def accuracy_report(y_true, y_pred) -> dict:
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred),
            "MAPE": mape(y_true, y_pred), "R2": r2(y_true, y_pred)}


def inventory_report(order_qty, demand) -> dict:
    """Outcome rates for a set of period decisions (arrays of equal length)."""
    q, d = _arrays(order_qty, demand)
    sold = np.minimum(q, d)
    return {
        "stockout_rate": float(np.mean(d > q) * 100),          # % of periods that ran out
        "service_level": float(np.mean(d <= q) * 100),         # % of periods with no stockout
        "fill_rate": float(sold.sum() / d.sum() * 100),        # % of demand served
        "leftover_pct": float(np.maximum(q - d, 0).sum() / q.sum() * 100),  # % of ordered units unsold
    }

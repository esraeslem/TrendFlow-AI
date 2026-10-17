"""
Newsvendor optimisation for TrendFlow.

Single-period inventory decision under uncertain demand D:

    c_u = price - cost          (profit lost per unit of unmet demand)
    c_o = cost  - salvage       (loss per unsold unit)
    CR  = c_u / (c_u + c_o)     (critical ratio)
    Q*  = F^-1(CR)              (F = CDF of demand over the planning period)

Two ways of specifying F are supported:

* ``optimize_samples`` - an empirical distribution, e.g. Prophet's predictive
  samples of total demand over the horizon (preferred: no shape assumption and
  correct aggregation of daily uncertainty).
* ``optimize_normal`` - a Normal(mean, std) approximation.

``calculate_optimal_quantity`` keeps the original README interface
(mean + 95% interval bounds of the *period total*) and maps the interval to a
normal standard deviation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def realized_profit(order_qty, demand, unit_price: float, unit_cost: float,
                    salvage_value: float = 0.0):
    """Profit actually earned when ``order_qty`` units were bought and ``demand`` occurred.

    Works element-wise on arrays.
    """
    q = np.asarray(order_qty, dtype=float)
    d = np.asarray(demand, dtype=float)
    sold = np.minimum(q, d)
    leftover = np.maximum(q - d, 0.0)
    return unit_price * sold - unit_cost * q + salvage_value * leftover


@dataclass
class NewsvendorOptimizer:
    """Profit-maximising order quantity for one planning period."""

    unit_cost: float
    unit_price: float
    salvage_value: float = 0.0

    def __post_init__(self):
        if self.salvage_value < 0:
            raise ValueError("salvage_value must be >= 0")
        if not self.unit_cost > self.salvage_value:
            raise ValueError("unit_cost must be greater than salvage_value "
                             "(otherwise ordering unlimited stock is optimal)")
        if not self.unit_price > self.unit_cost:
            raise ValueError("unit_price must be greater than unit_cost "
                             "(otherwise ordering nothing is optimal)")

    # ------------------------------------------------------------------ costs
    @property
    def understock_cost(self) -> float:
        return self.unit_price - self.unit_cost

    @property
    def overstock_cost(self) -> float:
        return self.unit_cost - self.salvage_value

    @property
    def critical_ratio(self) -> float:
        cu, co = self.understock_cost, self.overstock_cost
        return cu / (cu + co)

    # -------------------------------------------------------- normal demand
    def expected_outcome_normal(self, order_qty: float, mean: float, std: float) -> dict:
        """Closed-form expected sales / leftovers / profit for D ~ Normal(mean, std)."""
        if std <= 0:
            sold = min(order_qty, mean)
            leftover = max(order_qty - mean, 0.0)
            shortage = max(mean - order_qty, 0.0)
            in_stock_prob = 1.0 if order_qty >= mean else 0.0
        else:
            z = (order_qty - mean) / std
            loss = norm.pdf(z) - z * norm.sf(z)        # standard normal loss L(z)
            shortage = std * loss                       # E[(D - Q)+]
            sold = mean - shortage                      # E[min(Q, D)]
            leftover = order_qty - sold                 # E[(Q - D)+]
            in_stock_prob = float(norm.cdf(z))
        profit = (self.unit_price * sold - self.unit_cost * order_qty
                  + self.salvage_value * leftover)
        return self._pack(order_qty, profit, sold, leftover, shortage, in_stock_prob, mean)

    def optimize_normal(self, mean: float, std: float) -> dict:
        """Optimal order for D ~ Normal(mean, std)."""
        if std < 0:
            raise ValueError("std must be >= 0")
        q = mean + norm.ppf(self.critical_ratio) * std if std > 0 else mean
        q = max(q, 0.0)
        return self.expected_outcome_normal(q, mean, std)

    # ------------------------------------------------------ sampled demand
    def expected_outcome_samples(self, order_qty: float, samples) -> dict:
        d = np.clip(np.asarray(samples, dtype=float), 0.0, None)
        sold = np.minimum(order_qty, d)
        profit = realized_profit(order_qty, d, self.unit_price, self.unit_cost, self.salvage_value)
        return self._pack(order_qty, profit.mean(), sold.mean(),
                          np.maximum(order_qty - d, 0).mean(),
                          np.maximum(d - order_qty, 0).mean(),
                          float((d <= order_qty).mean()), d.mean())

    def optimize_samples(self, samples) -> dict:
        """Optimal order = CR-quantile of the empirical demand distribution."""
        d = np.clip(np.asarray(samples, dtype=float), 0.0, None)
        if d.size == 0:
            raise ValueError("samples is empty")
        q = float(np.quantile(d, self.critical_ratio))
        return self.expected_outcome_samples(q, d)

    # ------------------------------------------------- README-compatible API
    def calculate_optimal_quantity(self, forecast_mean: float, forecast_lower: float,
                                   forecast_upper: float, interval_width: float = 0.95) -> dict:
        """Optimal order from a mean and a central prediction interval of total demand.

        The interval is converted to a normal standard deviation:
        ``std = (upper - lower) / (2 * z_{(1 + interval_width) / 2})``.
        Pass bounds of the *period total* (e.g. from
        ``ProphetForecaster.forecast_total_demand``), not sums of daily bounds.
        """
        if forecast_upper < forecast_lower:
            raise ValueError("forecast_upper must be >= forecast_lower")
        z = norm.ppf(0.5 + interval_width / 2)
        std = (forecast_upper - forecast_lower) / (2 * z)
        return self.optimize_normal(forecast_mean, std)

    # ----------------------------------------------------------- internals
    def _pack(self, q, profit, sold, leftover, shortage, in_stock_prob, mean_demand) -> dict:
        fill_rate = sold / mean_demand if mean_demand > 0 else 1.0
        return {
            "optimal_quantity": int(round(q)),
            "order_quantity_exact": float(q),
            "critical_ratio": round(self.critical_ratio, 4),
            "expected_profit": round(float(profit), 2),
            "expected_sales": float(sold),
            "expected_leftover": float(leftover),
            "expected_shortage": float(shortage),
            # probability that the period ends without a stockout (cycle service level)
            "service_level": round(100 * in_stock_prob, 1),
            "fill_rate": round(100 * float(fill_rate), 1),
        }

"""
Real-world validation on the Kaggle "Store Sales - Time Series Forecasting"
dataset (Corporación Favorita).

The data is not redistributed in this repo. Download ``train.csv`` from
https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
and run:

    python validate_real_world.py --data path/to/train.csv
    python validate_real_world.py --data train.csv --store 44 --family BEVERAGES --test-days 60

Protocol: Prophet is trained on all days before the last ``--test-days``.
For every test day an order is placed; profit is computed on actual sales.

Strategies compared:
  * Training-period mean (naive "order the average" policy)
  * Prophet mean forecast
  * TrendFlow: per-day newsvendor quantile of Prophet's predictive samples
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from src.forecasting import ProphetForecaster
from src.metrics import accuracy_report, inventory_report
from src.optimization import NewsvendorOptimizer, realized_profit


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="train.csv")
    p.add_argument("--store", type=int, default=44)
    p.add_argument("--family", default="BEVERAGES")
    p.add_argument("--test-days", type=int, default=60)
    p.add_argument("--price", type=float, default=15.0)
    p.add_argument("--cost", type=float, default=5.0)
    p.add_argument("--salvage", type=float, default=2.0)
    a = p.parse_args()

    if not os.path.exists(a.data):
        sys.exit(f"'{a.data}' not found. Download train.csv from Kaggle (see this script's docstring) "
                 "and pass its path with --data.")

    raw = pd.read_csv(a.data, usecols=["date", "store_nbr", "family", "sales"])
    series = (raw[(raw.store_nbr == a.store) & (raw.family == a.family)]
              .rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]])
    series["ds"] = pd.to_datetime(series["ds"])
    series = series.sort_values("ds").reset_index(drop=True)
    if len(series) <= a.test_days + 60:
        sys.exit("Not enough history for this store/family.")
    train, test = series.iloc[:-a.test_days], series.iloc[-a.test_days:]

    opt = NewsvendorOptimizer(unit_cost=a.cost, unit_price=a.price, salvage_value=a.salvage)
    f = ProphetForecaster().fit(train)
    # The Favorita series has gaps (stores closed on 25 Dec), so forecast every calendar day up to
    # the last test date and pick out the days that actually appear in the test set
    horizon = (test["ds"].max() - train["ds"].max()).days
    fc = f.predict(horizon)
    samples = f.daily_demand_samples(horizon)
    idx = pd.Index(fc["ds"]).get_indexer(test["ds"])
    if (idx < 0).any():
        sys.exit("Test dates fall outside the forecast horizon; check for gaps in the series.")
    actual = test["y"].to_numpy()

    orders = {
        "Training-period mean": np.full(len(test), train["y"].mean()),
        "Prophet mean forecast": fc["yhat"].to_numpy()[idx],
        "TrendFlow (newsvendor)": np.quantile(samples[idx], opt.critical_ratio, axis=1),
    }
    print(f"\nStore {a.store} / {a.family}: {len(train)} training days, {len(test)} test days, "
          f"critical ratio {opt.critical_ratio:.3f}")
    acc = accuracy_report(actual, fc["yhat"].to_numpy()[idx])
    print(f"Prophet daily accuracy: MAE {acc['MAE']:.1f}, MAPE {acc['MAPE']:.1f}%\n")

    base = None
    for name, q in orders.items():
        profit = realized_profit(q, actual, a.price, a.cost, a.salvage).sum()
        base = profit if base is None else base
        inv = inventory_report(q, actual)
        print(f"{name:<24} profit ${profit:>12,.0f}  ({(profit / base - 1) * 100:+6.2f}% vs mean)  "
              f"stockout days {inv['stockout_rate']:4.0f}%  fill rate {inv['fill_rate']:5.1f}%")


if __name__ == "__main__":
    main()

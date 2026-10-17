"""
Rolling-origin backtest for TrendFlow.

For every product and every forecast origin (default: day 365, then every 30
days) each model is trained only on data before the origin, then:

1. Forecast accuracy - daily forecasts for the next ``horizon`` days are scored
   against actual sales (MAE, RMSE, MAPE, R2).
2. Ordering decision - each strategy places one order for the whole
   ``horizon``-day period. Profit is computed against the *realised* demand
   (sum of actual sales), never against the forecast.

Usage:
    python run_benchmark.py                       # defaults: 30-day periods, price 120 / cost 50 / salvage 10
    python run_benchmark.py --horizon 14 --price 80

Outputs go to experiments/results/: accuracy.csv, strategies.csv,
decisions.csv (every individual decision) and summary.md.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from src.baselines import (ARIMAForecaster, MovingAverageForecaster, NaiveForecaster,
                           SeasonalNaiveForecaster)
from src.data_processing import load_sales_data, product_series
from src.forecasting import ProphetForecaster
from src.metrics import accuracy_report, inventory_report
from src.optimization import NewsvendorOptimizer, realized_profit

BASELINE_STRATEGY = "Order = Prophet mean forecast"
TRENDFLOW_STRATEGY = "TrendFlow: newsvendor on Prophet predictive samples"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.path.join("data", "raw", "fashion_sales_data.csv"))
    p.add_argument("--horizon", type=int, default=30, help="planning period length in days")
    p.add_argument("--first-origin", type=int, default=365, help="days of history before the first decision")
    p.add_argument("--step", type=int, default=30, help="days between forecast origins")
    p.add_argument("--price", type=float, default=120.0)
    p.add_argument("--cost", type=float, default=50.0)
    p.add_argument("--salvage", type=float, default=10.0)
    p.add_argument("--out", default=os.path.join("experiments", "results"))
    return p.parse_args()


def run(args) -> dict:
    df = load_sales_data(args.data)
    opt = NewsvendorOptimizer(unit_cost=args.cost, unit_price=args.price, salvage_value=args.salvage)
    cr = opt.critical_ratio
    baseline_models = [NaiveForecaster(), MovingAverageForecaster(7), SeasonalNaiveForecaster(7),
                       SeasonalNaiveForecaster(365), ARIMAForecaster((2, 1, 2))]

    accuracy_rows, decision_rows = [], []
    for product in sorted(df["Product"].unique()):
        series = product_series(df, product)
        n = len(series)
        origins = range(args.first_origin, n - args.horizon + 1, args.step)
        for origin in origins:
            train = series.iloc[:origin]
            test = series.iloc[origin:origin + args.horizon]
            actual = test["Sales"].to_numpy(dtype=float)
            demand = actual.sum()

            # --- Prophet
            pf = ProphetForecaster().fit(train)
            daily = pf.predict(args.horizon)
            total = pf.forecast_total_demand(args.horizon)
            preds = {"Prophet": daily["yhat"].to_numpy()}

            # --- Baselines
            for m in baseline_models:
                if isinstance(m, SeasonalNaiveForecaster) and len(train) < m.season:
                    continue
                preds[m.name] = m.fit(train["Sales"]).predict(args.horizon)

            for model, yhat in preds.items():
                accuracy_rows.append({"product": product, "origin": test["Date"].iloc[0].date(),
                                      "model": model, "y_true": actual, "y_pred": yhat})

            # --- Ordering strategies (one order for the whole period)
            orders = {
                "Order = moving-average (7-day) forecast": preds["Moving average (7-day)"].sum(),
                "Order = same period last year": preds.get("Seasonal naive (365-day)", np.array([np.nan])).sum(),
                BASELINE_STRATEGY: total["mean"],
                "Order = Prophet upper 97.5% bound": total["upper"],
                # rule used in the original dashboard: linear interpolation across
                # the sum of daily 95% bounds
                "Original dashboard rule (summed daily bounds)":
                    daily["yhat_lower"].sum() + cr * (daily["yhat_upper"].sum() - daily["yhat_lower"].sum()),
                TRENDFLOW_STRATEGY: opt.optimize_samples(total["samples"])["order_quantity_exact"],
                "Perfect foresight (upper limit)": demand,
            }
            for strategy, q in orders.items():
                if np.isnan(q):
                    continue
                q = max(float(q), 0.0)
                decision_rows.append({
                    "product": product, "origin": test["Date"].iloc[0].date(), "strategy": strategy,
                    "order_qty": q, "demand": demand,
                    "profit": float(realized_profit(q, demand, args.price, args.cost, args.salvage)),
                })
        print(f"  {product}: {len(origins)} decisions")

    # ---------------------------------------------------------- accuracy table
    acc = pd.DataFrame(accuracy_rows)
    acc_table = []
    for model, g in acc.groupby("model"):
        y_true = np.concatenate(g["y_true"].to_list())
        y_pred = np.concatenate(g["y_pred"].to_list())
        acc_table.append({"model": model, **accuracy_report(y_true, y_pred), "n_days": len(y_true)})
    acc_table = pd.DataFrame(acc_table).sort_values("MAE").reset_index(drop=True)

    # ---------------------------------------------------------- strategy table
    dec = pd.DataFrame(decision_rows)
    base = dec[dec.strategy == BASELINE_STRATEGY].set_index(["product", "origin"])["profit"]
    oracle = dec[dec.strategy == "Perfect foresight (upper limit)"]["profit"].sum()
    strat_table = []
    for strategy, g in dec.groupby("strategy"):
        paired = g.set_index(["product", "origin"])["profit"]
        common = paired.index.intersection(base.index)
        strat_table.append({
            "strategy": strategy,
            "n_decisions": len(g),
            "avg_profit_per_period": g["profit"].mean(),
            **inventory_report(g["order_qty"], g["demand"]),
            "profit_vs_mean_forecast_pct": (paired[common].sum() / base[common].sum() - 1) * 100,
            "beats_mean_forecast_pct_of_periods": float((paired[common] > base[common]).mean() * 100),
            "pct_of_perfect_foresight_profit": g["profit"].sum() / oracle * 100
            if len(g) == len(dec[dec.strategy == "Perfect foresight (upper limit)"]) else np.nan,
        })
    strat_table = (pd.DataFrame(strat_table)
                   .sort_values("avg_profit_per_period", ascending=False).reset_index(drop=True))

    return {"accuracy": acc_table, "strategies": strat_table, "decisions": dec, "critical_ratio": cr}


def write_outputs(res: dict, args):
    os.makedirs(args.out, exist_ok=True)
    res["accuracy"].round(3).to_csv(os.path.join(args.out, "accuracy.csv"), index=False)
    res["strategies"].round(3).to_csv(os.path.join(args.out, "strategies.csv"), index=False)
    res["decisions"].round(2).to_csv(os.path.join(args.out, "decisions.csv"), index=False)

    acc = res["accuracy"].copy()
    acc_md = acc.assign(MAE=acc.MAE.map("{:.1f}".format), RMSE=acc.RMSE.map("{:.1f}".format),
                        MAPE=acc.MAPE.map("{:.1f}%".format), R2=acc.R2.map("{:.2f}".format))
    st = res["strategies"].copy()
    st_md = pd.DataFrame({
        "Strategy": st.strategy,
        "Avg profit / period": st.avg_profit_per_period.map("${:,.0f}".format),
        "vs. mean forecast": st.profit_vs_mean_forecast_pct.map("{:+.2f}%".format),
        "Stockout rate": st.stockout_rate.map("{:.0f}%".format),
        "Fill rate": st.fill_rate.map("{:.1f}%".format),
        "Unsold stock": st.leftover_pct.map("{:.1f}%".format),
        "% of perfect-foresight profit": st.pct_of_perfect_foresight_profit.map(
            lambda v: "n/a" if np.isnan(v) else f"{v:.1f}%"),
    })
    n_dec = int(st.n_decisions.max())
    md = [
        "# Benchmark results",
        "",
        f"Generated by `python run_benchmark.py` on `{args.data}`. Rolling-origin backtest: "
        f"first decision after {args.first_origin} days of history, then every {args.step} days; "
        f"{args.horizon}-day planning periods; {n_dec} decisions in total.",
        f"Unit economics: price ${args.price:g}, cost ${args.cost:g}, salvage ${args.salvage:g} "
        f"→ critical ratio {res['critical_ratio']:.3f}.",
        "",
        "## Daily forecast accuracy (out of sample)",
        "",
        acc_md[["model", "MAE", "RMSE", "MAPE", "R2", "n_days"]].to_markdown(index=False),
        "",
        "## Ordering strategies (profit measured on realised demand)",
        "",
        st_md.to_markdown(index=False),
        "",
    ]
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(md))
    return "\n".join(md)


if __name__ == "__main__":
    a = parse_args()
    t0 = time.time()
    print("Running rolling-origin backtest...")
    results = run(a)
    print(write_outputs(results, a))
    print(f"Done in {time.time() - t0:.0f}s. Files written to {a.out}/")

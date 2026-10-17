"""
Rolling-origin backtest for TrendFlow.

For every product and every forecast origin (default: day 365, then every 30
days) each model is trained only on data before the origin, then:

1. Forecast accuracy - daily forecasts for the next ``horizon`` days are scored
   against actual sales (MAE, RMSE, MAPE, R2).
2. Interval calibration - how often actuals fall inside Prophet's 95% bands
   (daily and for the period total). Well-calibrated bands should hit ~95%.
3. Ordering decision - each strategy places one order for the whole
   ``horizon``-day period. Profit is computed against *realised* demand
   (true demand if the data has a ``Demand`` column, otherwise sales).

Usage:
    python run_benchmark.py                                  # synthetic data, uniform economics
    python run_benchmark.py --economics experiments/economics.json   # per-product margins
    python run_benchmark.py --stockout-prob 0.15 --censoring mask    # censored demand
    python run_benchmark.py --data data/processed/hm_daily.csv --max-products 6   # H&M

Outputs go to ``--out`` (default experiments/results/): accuracy.csv,
strategies.csv, calibration.csv, decisions.csv and summary.md.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from src.baselines import (ARIMAForecaster, LightGBMForecaster, MovingAverageForecaster,
                           NaiveForecaster, SeasonalNaiveForecaster)
from src.data_processing import (fill_stockouts, generate_fashion_data, load_sales_data,
                                 mask_stockouts, product_series)
from src.forecasting import ProphetForecaster
from src.metrics import accuracy_report, inventory_report
from src.optimization import NewsvendorOptimizer, realized_profit

BASELINE_STRATEGY = "Order = Prophet mean forecast"
TRENDFLOW_STRATEGY = "TrendFlow: newsvendor on Prophet predictive samples"
ORACLE_STRATEGY = "Perfect foresight (upper limit)"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.path.join("data", "raw", "fashion_sales_data.csv"),
                   help="long-format CSV with Date, Product, Sales (optional Demand, InStock)")
    p.add_argument("--horizon", type=int, default=30, help="planning period length in days")
    p.add_argument("--first-origin", type=int, default=365, help="days of history before the first decision")
    p.add_argument("--step", type=int, default=30, help="days between forecast origins")
    p.add_argument("--price", type=float, default=120.0)
    p.add_argument("--cost", type=float, default=50.0)
    p.add_argument("--salvage", type=float, default=10.0)
    p.add_argument("--economics", help="JSON {product: {price, cost, salvage}, 'default': {...}} "
                                       "overriding --price/--cost/--salvage per product")
    p.add_argument("--products", nargs="*", help="only these products")
    p.add_argument("--max-products", type=int, help="only the N highest-volume products")
    p.add_argument("--stockout-prob", type=float, default=0.0,
                   help="regenerate the synthetic data with simulated stockouts (ignores --data)")
    p.add_argument("--censoring", choices=["ignore", "mask"], default="ignore",
                   help="ignore: train on raw sales; mask: treat stockout-day sales as missing")
    p.add_argument("--no-arima", action="store_true", help="skip ARIMA (slow on long or many series)")
    p.add_argument("--out", default=os.path.join("experiments", "results"))
    return p.parse_args(argv)


def economics_for(args) -> dict:
    default = {"price": args.price, "cost": args.cost, "salvage": args.salvage}
    table = {}
    if args.economics:
        with open(args.economics) as f:
            table = json.load(f)
        default = {**default, **table.pop("default", {})}
        table = {k: v for k, v in table.items() if not k.startswith("_")}
    return {"default": default, **table}


def run(args) -> dict:
    if args.stockout_prob > 0:
        df = generate_fashion_data(output_path=None, stockout_prob=args.stockout_prob)
    else:
        df = load_sales_data(args.data, generate_if_missing=args.data.endswith("fashion_sales_data.csv"))
    products = sorted(df["Product"].unique())
    if args.products:
        products = [p for p in products if p in set(args.products)]
    if args.max_products:
        top = df.groupby("Product")["Sales"].sum().sort_values(ascending=False).index[:args.max_products]
        products = [p for p in products if p in set(top)]
    econ = economics_for(args)

    accuracy_rows, decision_rows, calib_rows = [], [], []
    for product in products:
        e = econ.get(product, econ["default"])
        opt = NewsvendorOptimizer(unit_cost=e["cost"], unit_price=e["price"], salvage_value=e["salvage"])
        cr = opt.critical_ratio

        series = product_series(df, product)
        has_demand = "Demand" in series.columns
        n = len(series)
        origins = list(range(args.first_origin, n - args.horizon + 1, args.step))
        if not origins:
            print(f"  {product}: skipped (only {n} days of data)")
            continue
        models = [NaiveForecaster(), MovingAverageForecaster(7), SeasonalNaiveForecaster(7),
                  SeasonalNaiveForecaster(365), LightGBMForecaster(args.horizon)]
        if not args.no_arima:
            models.append(ARIMAForecaster((2, 1, 2)))

        for origin in origins:
            raw_train = series.iloc[:origin]
            test = series.iloc[origin:origin + args.horizon]
            # what we observed (sales) vs. what customers wanted (demand)
            actual_sales = test["Sales"].to_numpy(dtype=float)
            actual_demand = (test["Demand"] if has_demand else test["Sales"]).to_numpy(dtype=float)
            demand_total = actual_demand.sum()

            if args.censoring == "mask":
                prophet_train, base_train = mask_stockouts(raw_train), fill_stockouts(raw_train)
            else:
                prophet_train = base_train = raw_train

            # --- Prophet
            pf = ProphetForecaster().fit(prophet_train[["Date", "Sales"]])
            daily = pf.predict(args.horizon)
            total = pf.forecast_total_demand(args.horizon)
            preds = {"Prophet": daily["yhat"].to_numpy()}

            # --- Baselines
            y = base_train.set_index("Date")["Sales"].astype(float)
            for m in models:
                if isinstance(m, SeasonalNaiveForecaster) and len(y) < m.season:
                    continue
                preds[m.name] = m.fit(y).predict(args.horizon)

            origin_date = test["Date"].iloc[0].date()
            for model, yhat in preds.items():
                accuracy_rows.append({"product": product, "origin": origin_date, "model": model,
                                      "y_true": actual_demand, "y_pred": yhat})

            # --- Calibration of Prophet's uncertainty
            calib_rows.append({
                "product": product, "origin": origin_date,
                "daily_in_95": float(((actual_demand >= daily["yhat_lower"]) &
                                      (actual_demand <= daily["yhat_upper"])).mean()),
                "total_in_95": float(total["lower"] <= demand_total <= total["upper"]),
                "pit": float((total["samples"] <= demand_total).mean()),  # ~Uniform(0,1) if calibrated
            })

            # --- Ordering strategies (one order for the whole period)
            orders = {
                "Order = moving-average (7-day) forecast": preds["Moving average (7-day)"].sum(),
                "Order = LightGBM forecast": preds["LightGBM (calendar + lags)"].sum(),
                "Order = same period last year": preds.get("Seasonal naive (365-day)", np.array([np.nan])).sum(),
                BASELINE_STRATEGY: total["mean"],
                "Order = Prophet upper 97.5% bound": total["upper"],
                "Original dashboard rule (summed daily bounds)":
                    daily["yhat_lower"].sum() + cr * (daily["yhat_upper"].sum() - daily["yhat_lower"].sum()),
                TRENDFLOW_STRATEGY: opt.optimize_samples(total["samples"])["order_quantity_exact"],
                ORACLE_STRATEGY: demand_total,
            }
            for strategy, q in orders.items():
                if np.isnan(q):
                    continue
                q = max(float(q), 0.0)
                decision_rows.append({
                    "product": product, "origin": origin_date, "strategy": strategy,
                    "critical_ratio": cr, "order_qty": q, "demand": demand_total,
                    "observed_sales": actual_sales.sum(),
                    "profit": float(realized_profit(q, demand_total, e["price"], e["cost"], e["salvage"])),
                })
        print(f"  {product}: {len(origins)} decisions (CR {cr:.2f})")

    if not decision_rows:
        raise SystemExit("No product had enough history for the requested --first-origin/--horizon.")

    # ---------------------------------------------------------- accuracy table
    acc = pd.DataFrame(accuracy_rows)
    acc_table = []
    for model, g in acc.groupby("model"):
        y_true = np.concatenate(g["y_true"].to_list())
        y_pred = np.concatenate(g["y_pred"].to_list())
        acc_table.append({"model": model, **accuracy_report(y_true, y_pred), "n_days": len(y_true)})
    acc_table = pd.DataFrame(acc_table).sort_values("MAE").reset_index(drop=True)

    # ------------------------------------------------------- calibration table
    cal = pd.DataFrame(calib_rows)
    cal_summary = {
        "daily_coverage_95": cal["daily_in_95"].mean() * 100,
        "total_coverage_95": cal["total_in_95"].mean() * 100,
        "pit_below_10pct": (cal["pit"] < 0.1).mean() * 100,   # demand in Prophet's lower tail
        "pit_above_90pct": (cal["pit"] > 0.9).mean() * 100,   # demand in Prophet's upper tail
        "n_periods": len(cal),
    }

    # ---------------------------------------------------------- strategy table
    dec = pd.DataFrame(decision_rows)
    base = dec[dec.strategy == BASELINE_STRATEGY].set_index(["product", "origin"])["profit"]
    oracle = dec[dec.strategy == ORACLE_STRATEGY].set_index(["product", "origin"])["profit"]
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
            "pct_of_perfect_foresight_profit": paired.sum() / oracle[paired.index].sum() * 100,
        })
    strat_table = (pd.DataFrame(strat_table)
                   .sort_values("avg_profit_per_period", ascending=False).reset_index(drop=True))

    # ------------------------------------------------- per-product TrendFlow gain
    piv = dec.pivot_table(index="product", columns="strategy", values="profit", aggfunc="sum")
    per_product = pd.DataFrame({
        "critical_ratio": dec.groupby("product")["critical_ratio"].first(),
        "trendflow_vs_mean_pct": (piv[TRENDFLOW_STRATEGY] / piv[BASELINE_STRATEGY] - 1) * 100,
        "trendflow_vs_moving_avg_pct":
            (piv[TRENDFLOW_STRATEGY] / piv["Order = moving-average (7-day) forecast"] - 1) * 100,
    }).reset_index()

    return {"accuracy": acc_table, "strategies": strat_table, "decisions": dec,
            "calibration": cal, "calibration_summary": cal_summary, "per_product": per_product,
            "economics": econ, "products": products}


def write_outputs(res: dict, args) -> str:
    os.makedirs(args.out, exist_ok=True)
    res["accuracy"].round(3).to_csv(os.path.join(args.out, "accuracy.csv"), index=False)
    res["strategies"].round(3).to_csv(os.path.join(args.out, "strategies.csv"), index=False)
    res["calibration"].round(4).to_csv(os.path.join(args.out, "calibration.csv"), index=False)
    res["per_product"].round(3).to_csv(os.path.join(args.out, "per_product.csv"), index=False)
    res["decisions"].round(2).to_csv(os.path.join(args.out, "decisions.csv"), index=False)

    acc = res["accuracy"]
    acc_md = acc.assign(MAE=acc.MAE.map("{:.1f}".format), RMSE=acc.RMSE.map("{:.1f}".format),
                        MAPE=acc.MAPE.map("{:.1f}%".format), R2=acc.R2.map("{:.2f}".format))
    st = res["strategies"]
    st_md = pd.DataFrame({
        "Strategy": st.strategy,
        "Avg profit / period": st.avg_profit_per_period.map("${:,.0f}".format),
        "vs. mean forecast": st.profit_vs_mean_forecast_pct.map("{:+.2f}%".format),
        "Stockout rate": st.stockout_rate.map("{:.0f}%".format),
        "Fill rate": st.fill_rate.map("{:.1f}%".format),
        "Unsold stock": st.leftover_pct.map("{:.1f}%".format),
        "% of perfect-foresight profit": st.pct_of_perfect_foresight_profit.map("{:.1f}%".format),
    })
    pp = res["per_product"]
    pp_md = pd.DataFrame({
        "Product": pp["product"], "Critical ratio": pp.critical_ratio.map("{:.2f}".format),
        "TrendFlow vs. ordering the mean": pp.trendflow_vs_mean_pct.map("{:+.2f}%".format),
        "TrendFlow vs. moving average": pp.trendflow_vs_moving_avg_pct.map("{:+.2f}%".format),
    })
    c = res["calibration_summary"]
    econ = res["economics"]
    econ_line = (f"price ${econ['default']['price']:g}, cost ${econ['default']['cost']:g}, "
                 f"salvage ${econ['default']['salvage']:g} for all products"
                 if len(econ) == 1 else f"per product from `{args.economics}`")
    data_line = (f"synthetic data regenerated with {args.stockout_prob:.0%} stockout days"
                 if args.stockout_prob > 0 else f"`{args.data}`")
    n_dec = int(st.n_decisions.max())
    md = [
        "# Benchmark results",
        "",
        f"Command: `python run_benchmark.py {' '.join(getattr(args, 'argv', []))}`".rstrip(),
        "",
        f"Data: {data_line}; products: {', '.join(res['products'])}. Rolling-origin backtest: first decision "
        f"after {args.first_origin} days, then every {args.step} days; {args.horizon}-day planning periods; "
        f"{n_dec} decisions. Censoring: `{args.censoring}`. Economics: {econ_line}.",
        "",
        "## Daily forecast accuracy (out of sample)",
        "",
        acc_md[["model", "MAE", "RMSE", "MAPE", "R2", "n_days"]].to_markdown(index=False),
        "",
        "## Calibration of Prophet's 95% intervals",
        "",
        "| Check | Result | Ideal |",
        "|---|---:|---:|",
        f"| Daily actuals inside the 95% band | {c['daily_coverage_95']:.1f}% | 95% |",
        f"| Period totals inside the 95% interval | {c['total_coverage_95']:.1f}% | 95% |",
        f"| Periods with demand in the lowest 10% of the forecast distribution | {c['pit_below_10pct']:.1f}% | 10% |",
        f"| Periods with demand in the highest 10% of the forecast distribution | {c['pit_above_90pct']:.1f}% | 10% |",
        "",
        "## Ordering strategies (profit measured on realised demand)",
        "",
        st_md.to_markdown(index=False),
        "",
        "## Newsvendor gain by product",
        "",
        pp_md.to_markdown(index=False),
        "",
    ]
    text = "\n".join(md)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(text)
    return text


if __name__ == "__main__":
    import sys

    a = parse_args()
    a.argv = sys.argv[1:]
    t0 = time.time()
    print("Running rolling-origin backtest...")
    results = run(a)
    print(write_outputs(results, a))
    print(f"Done in {time.time() - t0:.0f}s. Files written to {a.out}/")

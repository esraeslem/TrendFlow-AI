"""
Turn the H&M Kaggle dataset into TrendFlow's daily long format.

Data: "H&M Personalized Fashion Recommendations"
https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data
Download ``transactions_train.csv`` (~3.5 GB, 31M rows, 2018-09-20 to 2020-09-22)
and ``articles.csv`` into a folder, e.g. ``data/hm/``. They are not redistributed here.

    python prepare_hm.py --hm-dir data/hm                         # product types, top 12
    python prepare_hm.py --hm-dir data/hm --level product_group_name --top 8
    python run_benchmark.py --data data/processed/hm_daily.csv --no-arima

Each transaction row is one unit sold, so daily units = row count per
(day, product group). Days with no sales for a group are filled with 0.

Caveats (also printed):
* ``price`` is normalised by H&M, so real margins are unknown: unit economics
  must be assumed (``--economics`` in run_benchmark.py).
* There is no inventory data, so stockout days cannot be flagged; sales may
  understate demand for popular items.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

LEVELS = ["product_type_name", "product_group_name", "garment_group_name", "index_group_name"]


def aggregate(transactions_path: str, articles_path: str, level: str = "product_type_name",
              chunksize: int = 2_000_000) -> pd.DataFrame:
    """Daily units sold per ``level`` group, read in chunks to keep memory low."""
    articles = pd.read_csv(articles_path, usecols=["article_id", level], dtype={"article_id": "int64"})
    group_of = articles.set_index("article_id")[level]

    parts = []
    reader = pd.read_csv(transactions_path, usecols=["t_dat", "article_id"],
                         dtype={"article_id": "int64"}, chunksize=chunksize)
    for i, chunk in enumerate(reader, 1):
        chunk["Product"] = chunk["article_id"].map(group_of)
        parts.append(chunk.groupby(["t_dat", "Product"]).size())
        print(f"  chunk {i}: {len(chunk):,} rows", file=sys.stderr)
    daily = pd.concat(parts).groupby(level=[0, 1]).sum().rename("Sales").reset_index()
    daily = daily.rename(columns={"t_dat": "Date"})
    daily["Date"] = pd.to_datetime(daily["Date"])

    # complete calendar: a group with no sales on a day sold 0 units
    full = pd.MultiIndex.from_product(
        [pd.date_range(daily["Date"].min(), daily["Date"].max()), daily["Product"].unique()],
        names=["Date", "Product"])
    daily = daily.set_index(["Date", "Product"]).reindex(full, fill_value=0).reset_index()
    return daily[["Date", "Product", "Sales"]].sort_values(["Product", "Date"]).reset_index(drop=True)


def select_top(daily: pd.DataFrame, top: int, min_daily_mean: float = 20.0) -> pd.DataFrame:
    """Keep the ``top`` highest-volume groups with enough volume to forecast."""
    vol = daily.groupby("Product")["Sales"].mean()
    keep = vol[vol >= min_daily_mean].sort_values(ascending=False).index[:top]
    return daily[daily["Product"].isin(keep)].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hm-dir", default=os.path.join("data", "hm"))
    p.add_argument("--level", choices=LEVELS, default="product_type_name")
    p.add_argument("--top", type=int, default=12, help="number of product groups to keep")
    p.add_argument("--min-daily-mean", type=float, default=20.0)
    p.add_argument("--out", default=os.path.join("data", "processed", "hm_daily.csv"))
    a = p.parse_args()

    tx = os.path.join(a.hm_dir, "transactions_train.csv")
    art = os.path.join(a.hm_dir, "articles.csv")
    missing = [f for f in (tx, art) if not os.path.exists(f)]
    if missing:
        sys.exit(f"Missing {missing}. Download them from Kaggle (see this script's docstring).")

    daily = select_top(aggregate(tx, art, a.level), a.top, a.min_daily_mean)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    daily.to_csv(a.out, index=False)

    summary = daily.groupby("Product")["Sales"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    print(f"Wrote {a.out}: {daily['Product'].nunique()} groups x {daily['Date'].nunique()} days "
          f"({daily['Date'].min().date()} to {daily['Date'].max().date()})")
    print(summary.round(1).to_string())
    print("\nNote: H&M prices are normalised and there is no stock data. Pass assumed unit economics "
          "to run_benchmark.py (--economics) and treat stockouts as unobserved.")


if __name__ == "__main__":
    main()

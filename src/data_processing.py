"""Synthetic data generation and loading."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = os.path.join("data", "raw", "fashion_sales_data.csv")
PRODUCTS = ["Floral Dress", "Leather Jacket", "Denim Jeans", "Chunky Sneaker"]


def generate_fashion_data(output_path: str | None = DEFAULT_DATA_PATH, start_date: str = "2023-01-01",
                          periods: int = 730, seed: int = 42, stockout_prob: float = 0.0) -> pd.DataFrame:
    """Generate synthetic daily sales for 4 fashion products.

    Demand model per product and day (units):
      * base level 100
      * Floral Dress: +150 in Apr-Aug (summer peak)
      * Leather Jacket: +200 in Oct-Feb (winter peak)
      * Chunky Sneaker: +50 constant uplift (higher baseline)
      * Denim Jeans: no seasonality
      * +15 units per calendar year after 2023 (step trend)
      * Gaussian noise, sd = 20; truncated at 0 and floored to an integer

    Returns a DataFrame with columns ['Date', 'Product', 'Sales'] and writes it to
    ``output_path`` unless that is None.

    ``stockout_prob > 0`` simulates censored demand: on that fraction of days the
    shelf runs out after 50-95% of demand is served, so ``Sales < Demand``. Two
    extra columns are then added: ``Demand`` (true demand) and ``InStock``
    (False on stockout days). This is what real sales data looks like - you only
    observe sales, and sales understate demand whenever you ran out.
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, periods=periods, freq="D")
    rows = []
    for date in dates:
        month = date.month
        for product in PRODUCTS:
            base_demand = 100
            if product == "Floral Dress" and 4 <= month <= 8:
                base_demand += 150
            elif product == "Leather Jacket" and (month >= 10 or month <= 2):
                base_demand += 200
            elif product == "Chunky Sneaker":
                base_demand += 50
            noise = np.random.normal(0, 20)
            trend = (date.year - 2023) * 15
            rows.append([date, product, max(0, int(base_demand + noise + trend))])

    df = pd.DataFrame(rows, columns=["Date", "Product", "Sales"])
    if stockout_prob > 0:
        rng = np.random.default_rng(seed + 1)
        df["Demand"] = df["Sales"]
        out = rng.random(len(df)) < stockout_prob
        served = rng.uniform(0.5, 0.95, len(df))
        df.loc[out, "Sales"] = np.floor(df.loc[out, "Demand"] * served[out]).astype(int)
        df["InStock"] = ~out
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


def load_sales_data(path: str = DEFAULT_DATA_PATH, generate_if_missing: bool = True) -> pd.DataFrame:
    """Load the sales CSV, generating the default synthetic dataset if it is missing."""
    if not os.path.exists(path):
        if not generate_if_missing:
            raise FileNotFoundError(path)
        return generate_fashion_data(output_path=path)
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def product_series(df: pd.DataFrame, product: str) -> pd.DataFrame:
    """Rows for one product, sorted by date: ['Date', 'Sales'] (+ 'Demand', 'InStock' if present)."""
    cols = ["Date", "Sales"] + [c for c in ("Demand", "InStock") if c in df.columns]
    out = df.loc[df["Product"] == product, cols].sort_values("Date")
    if out.empty:
        raise KeyError(f"Unknown product {product!r}. Available: {sorted(df['Product'].unique())}")
    return out.reset_index(drop=True)


def mask_stockouts(series: pd.DataFrame) -> pd.DataFrame:
    """Treat sales on stockout days as missing (they only give a lower bound on demand).

    Returns a copy where ``Sales`` is NaN on days with ``InStock == False``.
    Prophet skips NaN days when fitting; use ``fill_stockouts`` for models that
    cannot handle gaps.
    """
    out = series.copy()
    if "InStock" in out.columns:
        out.loc[~out["InStock"].astype(bool), "Sales"] = np.nan
    return out


def fill_stockouts(series: pd.DataFrame) -> pd.DataFrame:
    """``mask_stockouts`` then linear interpolation of the masked days."""
    out = mask_stockouts(series)
    out["Sales"] = out["Sales"].interpolate(limit_direction="both")
    return out

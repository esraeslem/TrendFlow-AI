"""Synthetic data generation and loading."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = os.path.join("data", "raw", "fashion_sales_data.csv")
PRODUCTS = ["Floral Dress", "Leather Jacket", "Denim Jeans", "Chunky Sneaker"]


def generate_fashion_data(output_path: str | None = DEFAULT_DATA_PATH, start_date: str = "2023-01-01",
                          periods: int = 730, seed: int = 42) -> pd.DataFrame:
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
    """Rows for one product, sorted by date, columns ['Date', 'Sales']."""
    out = df.loc[df["Product"] == product, ["Date", "Sales"]].sort_values("Date")
    if out.empty:
        raise KeyError(f"Unknown product {product!r}. Available: {sorted(df['Product'].unique())}")
    return out.reset_index(drop=True)

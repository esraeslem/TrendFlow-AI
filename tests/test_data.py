import os

import pandas as pd

from src.data_processing import DEFAULT_DATA_PATH, PRODUCTS, generate_fashion_data, product_series


def test_committed_dataset_matches_generator():
    committed = pd.read_csv(DEFAULT_DATA_PATH, parse_dates=["Date"])
    assert committed.equals(generate_fashion_data(output_path=None))


def test_dataset_schema_and_values():
    df = pd.read_csv(DEFAULT_DATA_PATH, parse_dates=["Date"])
    assert list(df.columns) == ["Date", "Product", "Sales"]
    assert sorted(df["Product"].unique()) == sorted(PRODUCTS)
    assert len(df) == 730 * 4
    assert df["Sales"].notna().all() and (df["Sales"] >= 0).all()
    assert not df.duplicated(["Date", "Product"]).any()


def test_seasonality_is_present():
    df = generate_fashion_data(output_path=None)
    dress = df[df.Product == "Floral Dress"].assign(m=lambda d: d.Date.dt.month)
    assert dress[dress.m.between(4, 8)].Sales.mean() > dress[~dress.m.between(4, 8)].Sales.mean() + 100


def test_product_series():
    df = generate_fashion_data(output_path=None, periods=10)
    s = product_series(df, "Denim Jeans")
    assert list(s.columns) == ["Date", "Sales"] and len(s) == 10


def test_generator_writes_file(tmp_path):
    out = tmp_path / "x" / "data.csv"
    generate_fashion_data(output_path=str(out), periods=5, seed=0)
    assert os.path.exists(out)

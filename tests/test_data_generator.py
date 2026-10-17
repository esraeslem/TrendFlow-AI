"""The legacy entry point ``data_generator.py`` must keep working."""
import pandas as pd

from data_generator import generate_fashion_data


def test_generate_writes_file_and_has_expected_shape(tmp_path):
    out = tmp_path / "test_fashion.csv"
    generate_fashion_data(output_path=str(out), start_date="2023-01-01", periods=10, seed=0)
    df = pd.read_csv(out)
    assert list(df.columns) == ["Date", "Product", "Sales"]
    assert df.shape[0] == 10 * 4
    assert df["Date"].min() == "2023-01-01"


def test_generate_returns_dataframe_without_writing():
    df = generate_fashion_data(output_path=None, periods=5, seed=1)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 5 * 4

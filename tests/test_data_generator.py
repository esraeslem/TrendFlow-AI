import sys
import os
import pandas as pd
# Ensure project root is on sys.path so tests can import top-level modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_generator import generate_fashion_data


def test_generate_writes_file_and_has_expected_shape(tmp_path):
    out = tmp_path / "test_fashion.csv"
    df = generate_fashion_data(output_path=str(out), start_date='2023-01-01', periods=10, seed=0)

    # file was written
    assert out.exists(), "CSV output was not created"

    # Read and validate
    df2 = pd.read_csv(out)
    assert list(df2.columns) == ['Date', 'Product', 'Sales']
    assert df2.shape[0] == 10 * 4
    assert df2['Date'].min() == '2023-01-01'


def test_generate_returns_dataframe_without_writing(tmp_path):
    df = generate_fashion_data(output_path=None, periods=5, seed=1)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 5 * 4
    assert list(df.columns) == ['Date', 'Product', 'Sales']

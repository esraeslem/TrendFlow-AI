"""Generate the synthetic dataset: ``python data_generator.py``.

The implementation lives in ``src/data_processing.py``; this script is kept so
existing commands and imports keep working.
"""

from src.data_processing import DEFAULT_DATA_PATH, generate_fashion_data

__all__ = ["generate_fashion_data"]

if __name__ == "__main__":
    df = generate_fashion_data(DEFAULT_DATA_PATH)
    print(f"Data written to {DEFAULT_DATA_PATH}: {len(df)} rows, columns {list(df.columns)}")

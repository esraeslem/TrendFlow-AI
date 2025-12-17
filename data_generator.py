import os
import pandas as pd
import numpy as np
import random


def generate_fashion_data(output_path: str = 'data/raw/fashion_sales_data.csv', start_date: str = '2023-01-01', periods: int = 730, seed: int = 42) -> pd.DataFrame:
    """Generates synthetic fashion retail data with seasonality.

    Parameters
    - output_path: Path to write the CSV. If None, the CSV won't be written.
    - start_date: Starting date for the generated series.
    - periods: Number of days to generate.
    - seed: RNG seed for reproducibility.

    Returns
    - pandas.DataFrame with columns ['Date', 'Product', 'Sales']
    """
    np.random.seed(seed)  # For reproducibility

    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    products = ['Floral Dress', 'Leather Jacket', 'Denim Jeans', 'Chunky Sneaker']
    data = []

    for date in dates:
        month = date.month
        for product in products:
            base_demand = 100

            # Seasonality Logic (The "Why")
            if product == 'Floral Dress' and 4 <= month <= 8:
                base_demand += 150  # Summer peak
            elif product == 'Leather Jacket' and (month >= 10 or month <= 2):
                base_demand += 200  # Winter peak
            elif product == 'Chunky Sneaker':
                base_demand += 50   # Slight trend

            # Add Random Noise and Trend
            noise = np.random.normal(0, 20)
            trend = (date.year - 2023) * 15

            sales = max(0, int(base_demand + noise + trend))
            data.append([date, product, sales])

    df = pd.DataFrame(data, columns=['Date', 'Product', 'Sales'])

    # Save to the correct folder if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Data generated successfully at: {output_path}")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {list(df.columns)}")

    return df

if __name__ == "__main__":
    generate_fashion_data()

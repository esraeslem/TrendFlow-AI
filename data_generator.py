import pandas as pd
import numpy as np
import random

def generate_fashion_data():
    """Generates synthetic fashion retail data with seasonality."""
    np.random.seed(42)  # For reproducibility (Professor will love this)
    
    dates = pd.date_range(start='2023-01-01', periods=730, freq='D')
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
    
    # Save to the correct folder
    output_path = 'data/raw/fashion_sales_data.csv'
    df.to_csv(output_path, index=False)
    print(f"✅ Data generated successfully at: {output_path}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")

if __name__ == "__main__":
    generate_fashion_data()

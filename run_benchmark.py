import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm
import sys
import os

# 1. Load Data
if not os.path.exists('data/raw/fashion_sales_data.csv'):
    print("❌ Data not found. Running generator...")
    import data_generator
    data_generator.generate_fashion_data()

df = pd.read_csv('data/raw/fashion_sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

print("✅ Data Loaded. Starting Benchmark...")
print("-" * 60)

# 2. Define Protocol
TRAIN_SPLIT = 0.8
split_idx = int(len(df) * TRAIN_SPLIT)
products = df['Product'].unique()
results = []

# 3. Run Loop for Each Product
for product in products:
    print(f"Processing {product}...")
    product_df = df[df['Product'] == product].sort_values('Date')
    
    # Train/Test Split (Time Series Cross-Validation logic)
    train = product_df.iloc[:split_idx]
    test = product_df.iloc[split_idx:]
    
    # Fit Prophet
    m = Prophet(daily_seasonality=False, yearly_seasonality=True)
    m.fit(train.rename(columns={'Date': 'ds', 'Sales': 'y'}))
    
    # Forecast
    future = m.make_future_dataframe(periods=len(test))
    forecast = m.predict(future)
    forecast_test = forecast.tail(len(test))
    
    # Metrics
    y_true = test['Sales'].values
    y_pred = forecast_test['yhat'].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Business Impact (Newsvendor)
    # Assumptions: Price=20, Cost=0, Salvage=0
    price, cost, salvage = 120, 50, 10
    cu, co = price - cost, cost - salvage
    cr = cu / (cu + co)
    
    # Sigma calculation (approximate from CI)
    sigma = (forecast_test['yhat_upper'] - forecast_test['yhat_lower']).mean() / 3.92
    
    # Optimal Quantity
    q_opt = norm.ppf(cr, loc=y_pred.sum(), scale=sigma * np.sqrt(len(test)))
    
    # Simple Profit Calculation
    actual_demand = y_true.sum()
    profit_model = (min(actual_demand, q_opt) * price) - (q_opt * cost) + (max(0, q_opt - actual_demand) * salvage)
    
    results.append({
        "Product": product,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "Optimal Order": int(q_opt),
        "Exp. Profit": int(profit_model)
    })

# 4. Output Results Table
results_df = pd.DataFrame(results)
print("-" * 60)
print("📊 FINAL BENCHMARK REPORT")
print("-" * 60)
print(results_df.to_string(index=False))
print("-" * 60)
results_df.to_csv("benchmark_results.csv", index=False)
print("✅ Results saved to 'benchmark_results.csv'")

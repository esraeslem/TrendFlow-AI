import pandas as pd
import numpy as np
import os
import sys
from scipy.stats import norm

# 1. Define or Import Model
try:
    sys.path.append(os.getcwd())
    from model import TrendFlow
    print("✅ Using your TrendFlow model.")
except:
    print("⚠️ Using Internal Validation Logic.")
    class TrendFlow:
        def fit(self, df): self.history = df
        def predict(self, periods):
            last_date = self.history['ds'].max()
            future_dates = pd.date_range(start=last_date, periods=periods+1)[1:]
            forecast = pd.DataFrame({'ds': future_dates})
            # Add a slight multiplier for weekends (Beverage logic)
            forecast['yhat'] = self.history['y'].mean() * np.where(forecast['ds'].dt.weekday >= 5, 1.3, 0.9)
            forecast['sigma'] = self.history['y'].std() * 0.4
            return forecast

# 2. Process
df = pd.read_csv('train.csv')
real_data = df[(df['store_nbr'] == 44) & (df['family'] == 'BEVERAGES')].copy()
real_data = real_data[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
real_data['ds'] = pd.to_datetime(real_data['ds'])

test_days = 60
train_df, test_df = real_data.iloc[:-test_days], real_data.iloc[-test_days:]

# 3. Optimize (High Margin = High Service Level)
price, cost, salvage = 15.0, 5.0, 2.0
cr = (price - cost) / (price - salvage)
z = norm.ppf(cr)

model = TrendFlow()
model.fit(train_df)
forecast = model.predict(test_days)

# AI vs Human
ai_orders = forecast['yhat'].values + (z * forecast['sigma'].values)
human_orders = np.full(test_days, train_df['y'].mean())
actual = test_df['y'].values

def calc_profit(q, d):
    sold = np.minimum(q, d)
    return np.sum((sold * price) - (q * cost) + (np.maximum(0, q-d) * salvage))

p_ai, p_human = calc_profit(ai_orders, actual), calc_profit(human_orders, actual)

print(f"\n" + "="*30)
print(f"💰 AI Profit:      ${p_ai:,.2f}")
print(f"👨‍💼 Baseline Profit: ${p_human:,.2f}")
print(f"📈 IMPROVEMENT:     {((p_ai-p_human)/p_human)*100:.2f}%")
print("="*30)

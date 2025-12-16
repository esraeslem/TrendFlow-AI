import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------
# CONFIGURATION & SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="TrendFlow: AI Supply Chain Optimizer", page_icon="📈", layout="wide")

st.title("📈 TrendFlow: AI-Powered Inventory Optimization")
st.markdown("""
**B2B Logic:** This tool combines **Probabilistic Forecasting** (Prophet) with **Stochastic Optimization** (Newsvendor Model) 
to determine the mathematically optimal inventory level that maximizes profit.
""")

# ---------------------------------------------------------
# SIDEBAR: BUSINESS INPUTS
# ---------------------------------------------------------
st.sidebar.header("1. Inventory Configuration")
selected_product = st.sidebar.selectbox("Select Product:", 
                                        ['Floral Dress', 'Leather Jacket', 'Denim Jeans', 'Chunky Sneaker'])
days_to_predict = st.sidebar.slider("Forecast Horizon (Days):", 30, 90, 60)

st.sidebar.header("2. Unit Economics (The 'Management' Part)")
# These inputs drive the Newsvendor Logic
selling_price = st.sidebar.number_input("Unit Selling Price ($)", value=85.0, step=5.0)
cost_price = st.sidebar.number_input("Unit Cost ($)", value=35.0, step=5.0)
salvage_value = st.sidebar.number_input("Salvage Value ($)", value=15.0, step=5.0, 
                                        help="Price you can sell leftover items for (e.g., clearance)")

if cost_price >= selling_price:
    st.sidebar.error("Error: Cost cannot be higher than Price!")

# ---------------------------------------------------------
# DATA GENERATION (Simulating Real Data)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    dates = pd.date_range(start='2023-01-01', periods=730, freq='D')
    products = ['Floral Dress', 'Leather Jacket', 'Denim Jeans', 'Chunky Sneaker']
    data = []
    
    # Add seasonality patterns
    for date in dates:
        month = date.month
        for product in products:
            base = 100
            # Seasonality Logic
            if product == 'Floral Dress' and 4 <= month <= 8: base += 150
            elif product == 'Leather Jacket' and (month >= 10 or month <= 2): base += 200
            elif product == 'Chunky Sneaker': base += 50
            
            # Add noise and trend
            noise = np.random.normal(0, 20)
            trend = (date.year - 2023) * 20
            sales = max(0, int(base + noise + trend))
            data.append([date, product, sales])
            
    return pd.DataFrame(data, columns=['Date', 'Product', 'Sales'])

df = load_data()
product_df = df[df['Product'] == selected_product].copy()

# ---------------------------------------------------------
# MODELING: PROPHET FORECAST
# ---------------------------------------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader(f"📊 Demand Forecast: {selected_product}")
    
with st.spinner("Training Prophet Model..."):
    df_prophet = product_df.rename(columns={'Date': 'ds', 'Sales': 'y'})
    m = Prophet(interval_width=0.95) # 95% Confidence Interval needed for optimization
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=days_to_predict)
    forecast = m.predict(future)

# Visualization
fig = plot_plotly(m, forecast)
fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# THE CORE LOGIC: NEWSVENDOR MODEL
# ---------------------------------------------------------
st.markdown("---")
st.header("🧠 Decision Engine: Newsvendor Optimization")

# 1. Get Forecast Distribution for the Period
future_period = forecast.tail(days_to_predict)
mu = future_period['yhat'].sum()           # Expected mean demand
sigma = np.sqrt((future_period['yhat_upper'] - future_period['yhat_lower']).pow(2).sum()) / 4 # Approx std dev

# 2. Calculate Critical Ratio (The "Management" Math)
cost_underestimation = selling_price - cost_price  # Profit lost if we stock out
cost_overestimation = cost_price - salvage_value   # Loss if we have leftovers

# Critical Fractal (Service Level)
critical_ratio = cost_underestimation / (cost_underestimation + cost_overestimation)

# 3. Calculate Optimal Order Quantity (Q*)
# We use the inverse CDF (Percent Point Function) of the normal distribution
optimal_order_quantity = norm.ppf(critical_ratio, loc=mu, scale=sigma)

# ---------------------------------------------------------
# RESULTS DASHBOARD
# ---------------------------------------------------------
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric(label="Predicted Demand (Mean)", 
              value=f"{int(mu):,} units",
              help="The raw AI prediction (50% probability)")

with col_b:
    st.metric(label="Optimal Order Quantity (Q*)", 
              value=f"{int(optimal_order_quantity):,} units",
              delta=f"Service Level: {critical_ratio:.1%}",
              help="The calculated quantity that maximizes PROFIT, not accuracy.")

with col_c:
    expected_profit = (min(mu, optimal_order_quantity) * selling_price) - \
                      (optimal_order_quantity * cost_price) + \
                      (max(0, optimal_order_quantity - mu) * salvage_value)
    st.metric(label="Exp. Projected Profit", 
              value=f"${int(expected_profit):,}")

# Strategy Comparison
st.subheader("💡 Why Use This Model?")
st.caption("Comparison of ordering strategies based on current unit economics:")

naive_order = mu # Just ordering the mean
naive_profit = (min(mu, naive_order) * selling_price) - (naive_order * cost_price)
improvement = expected_profit - naive_profit

comp_data = pd.DataFrame({
    "Strategy": ["Naive (Order Mean)", "TrendFlow (Newsvendor)"],
    "Order Quantity": [int(naive_order), int(optimal_order_quantity)],
    "Exp. Profit": [f"${int(naive_profit):,}", f"${int(expected_profit):,}"],
    "Safety Buffer": ["0 units", f"{int(optimal_order_quantity - mu)} units"]
})
st.table(comp_data)

if improvement > 0:
    st.success(f"🚀 **Business Impact:** Using the Newsvendor model adds **${int(improvement):,}** in expected profit compared to standard forecasting.")
else:
    st.info("Market conditions suggest conservative ordering is optimal.")

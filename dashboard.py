"""
TrendFlow Dashboard 
----------------------------------------
Author: Esra Eslem Savaş
Date: December 2025
"""

import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import random
import numpy as np

# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="TrendFlow: Supply Chain AI", 
    page_icon="📈", 
    layout="wide"
)

st.title("📈 TrendFlow: AI-Powered Supply Chain Optimizer")
st.markdown("### Profit-Aware Inventory Planning with Newsvendor Model")

# ========================================
# SIDEBAR CONFIGURATION
# ========================================
st.sidebar.header("⚙️ Configuration")

# Product selection
selected_product = st.sidebar.selectbox(
    "Select Product to Forecast:", 
    ['Floral Dress', 'Leather Jacket', 'Denim Jeans', 'Chunky Sneaker']
)

# Forecast horizon
days_to_predict = st.sidebar.slider(
    "Forecast Horizon (Days):", 
    30, 365, 90
)

# Economic parameters
st.sidebar.markdown("### 💰 Unit Economics")
unit_cost = st.sidebar.number_input(
    "Unit Cost ($)", 
    min_value=1.0, 
    max_value=1000.0, 
    value=50.0, 
    step=5.0,
    help="Cost per unit (overstocking penalty)"
)

unit_price = st.sidebar.number_input(
    "Unit Price ($)", 
    min_value=1.0, 
    max_value=2000.0, 
    value=120.0, 
    step=5.0,
    help="Selling price per unit"
)

salvage_value = st.sidebar.number_input(
    "Salvage Value ($)", 
    min_value=0.0, 
    max_value=1000.0, 
    value=10.0, 
    step=5.0,
    help="Recovery value for unsold inventory"
)

# ========================================
# DATA GENERATION
# ========================================
@st.cache_data
def generate_synthetic_data():
    """
    Generate synthetic fashion retail sales data.
    
    Simulates realistic patterns:
    - Seasonal trends (summer for dresses, winter for jackets)
    - Random noise (±20 units)
    - Product-specific base demand
    
    Returns:
        pd.DataFrame: Historical sales data with columns [Date, Product, Sales]
    """
    dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
    products = ['Floral Dress', 'Leather Jacket', 'Denim Jeans', 'Chunky Sneaker']
    data = []
    
    for date in dates:
        month = date.month
        for product in products:
            base_demand = random.randint(50, 200)
            
            # Seasonal adjustments
            if product == 'Floral Dress' and 4 <= month <= 8:
                base_demand += random.randint(300, 500)
            elif product == 'Leather Jacket' and (month >= 10 or month <= 2):
                base_demand += random.randint(200, 400)
            elif product == 'Denim Jeans':
                base_demand += random.randint(50, 100)
            
            # Convert to sales with noise
            actual_sales = int(base_demand * 0.45) + random.randint(-20, 20)
            data.append([date, product, max(0, actual_sales)])
            
    return pd.DataFrame(data, columns=['Date', 'Product', 'Sales'])

# ========================================
# NEWSVENDOR MODEL
# ========================================
def calculate_newsvendor_quantity(
    forecast_mean: float,
    forecast_lower: float,
    forecast_upper: float,
    unit_cost: float,
    unit_price: float,
    salvage_value: float = 0.0
) -> dict:
    """
    Calculate optimal order quantity using Newsvendor Model.
    
    The Newsvendor Model maximizes expected profit by balancing:
    - Overstocking cost: c_o = unit_cost - salvage_value
    - Understocking cost: c_u = unit_price - unit_cost
    
    Critical Ratio: CR = c_u / (c_u + c_o)
    Optimal Quantity: Q* = F^(-1)(CR) where F is demand distribution CDF
    
    Args:
        forecast_mean: Mean predicted demand
        forecast_lower: Lower bound of 95% CI
        forecast_upper: Upper bound of 95% CI
        unit_cost: Cost per unit purchased
        unit_price: Revenue per unit sold
        salvage_value: Recovery value for unsold units (default 0)
        
    Returns:
        dict: {
            'optimal_quantity': int,
            'critical_ratio': float,
            'expected_profit': float,
            'service_level': float
        }
        
    References:
        Schweitzer & Cachon (2000). "Decision Bias in the Newsvendor Problem"
    """
    # Calculate costs
    overstocking_cost = unit_cost - salvage_value  # Penalty for excess
    understocking_cost = unit_price - unit_cost     # Lost profit per stockout
    
    # Critical ratio (optimal service level)
    critical_ratio = understocking_cost / (understocking_cost + overstocking_cost)
    
    # Map critical ratio to demand distribution
    # Assuming normal distribution from Prophet's confidence interval
    optimal_quantity = forecast_lower + critical_ratio * (forecast_upper - forecast_lower)
    
    # Expected profit calculation (simplified)
    # Assumes demand follows forecast distribution
    expected_sales = min(optimal_quantity, forecast_mean)
    expected_revenue = expected_sales * unit_price
    expected_cost = optimal_quantity * unit_cost
    expected_salvage = max(0, optimal_quantity - forecast_mean) * salvage_value
    expected_profit = expected_revenue - expected_cost + expected_salvage
    
    return {
        'optimal_quantity': int(round(optimal_quantity)),
        'critical_ratio': round(critical_ratio, 3),
        'expected_profit': round(expected_profit, 2),
        'service_level': round(critical_ratio * 100, 1)
    }

# ========================================
# FORECASTING
# ========================================
def train_prophet_model(data: pd.DataFrame) -> tuple:
    """
    Train Facebook Prophet model on historical sales data.
    
    Args:
        data: DataFrame with columns ['Date', 'Sales']
        
    Returns:
        tuple: (trained_model, forecast_dataframe)
    """
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    df_prophet = data.rename(columns={'Date': 'ds', 'Sales': 'y'})
    
    # Initialize and train model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95  # 95% confidence interval
    )
    model.fit(df_prophet)
    
    return model

# ========================================
# MAIN APPLICATION
# ========================================

# Load data
df = generate_synthetic_data()
product_df = df[df['Product'] == selected_product].copy()

# Display historical data
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"📊 Historical Sales: {selected_product}")
    st.line_chart(product_df.set_index('Date')['Sales'])

with col2:
    st.info("""
    ℹ️ **Model Architecture:**
    - **Stage 1**: Facebook Prophet (time-series forecasting)
    - **Stage 2**: Newsvendor Model (profit optimization)
    """)

st.markdown("---")

# Train model and generate forecast
with st.spinner(f"🤖 Training AI on {selected_product} demand patterns..."):
    model = train_prophet_model(product_df)
    
    # Generate forecast
    future = model.make_future_dataframe(periods=days_to_predict)
    forecast = model.predict(future)

# Display forecast visualization
st.subheader(f"🔮 AI Forecast: Next {days_to_predict} Days")
fig = plot_plotly(model, forecast)
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# Extract future predictions
future_data = forecast.tail(days_to_predict)
total_forecast_mean = future_data['yhat'].sum()
avg_daily_forecast = future_data['yhat'].mean()
forecast_lower_sum = future_data['yhat_lower'].sum()
forecast_upper_sum = future_data['yhat_upper'].sum()

# Calculate profit margin
profit_margin = unit_price - unit_cost
margin_percentage = (profit_margin / unit_price) * 100

# ========================================
# NEWSVENDOR OPTIMIZATION
# ========================================
st.markdown("---")
st.subheader("🎯 Profit Optimization (Newsvendor Model)")

# Calculate optimal order quantity
optimization_result = calculate_newsvendor_quantity(
    forecast_mean=total_forecast_mean,
    forecast_lower=forecast_lower_sum,
    forecast_upper=forecast_upper_sum,
    unit_cost=unit_cost,
    unit_price=unit_price,
    salvage_value=salvage_value
)

# Display key metrics
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric(
    label="📈 Forecasted Demand",
    value=f"{int(total_forecast_mean):,} units",
    help="Mean prediction from Prophet"
)

metric_col2.metric(
    label="🎯 Optimal Order Quantity",
    value=f"{optimization_result['optimal_quantity']:,} units",
    delta=f"{optimization_result['optimal_quantity'] - int(total_forecast_mean):+,} vs forecast",
    help="Quantity that maximizes expected profit (Newsvendor Model)"
)

metric_col3.metric(
    label="💰 Expected Profit",
    value=f"${optimization_result['expected_profit']:,.2f}",
    help="Projected profit from optimal ordering"
)

metric_col4.metric(
    label="📊 Service Level",
    value=f"{optimization_result['service_level']}%",
    help="Target fill rate (Critical Ratio)"
)

# Display economic insights
st.markdown("### 💡 Economic Analysis")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.markdown(f"""
    **Unit Economics:**
    - **Cost per unit**: ${unit_cost:.2f}
    - **Price per unit**: ${unit_price:.2f}
    - **Margin per unit**: ${profit_margin:.2f} ({margin_percentage:.1f}%)
    - **Salvage value**: ${salvage_value:.2f}
    """)

with insight_col2:
    st.markdown(f"""
    **Optimization Parameters:**
    - **Critical Ratio**: {optimization_result['critical_ratio']:.3f}
    - **Overstocking penalty**: ${unit_cost - salvage_value:.2f}/unit
    - **Understocking penalty**: ${profit_margin:.2f}/unit
    - **Risk tolerance**: {optimization_result['service_level']}% service level
    """)

# ========================================
# BUSINESS RECOMMENDATIONS
# ========================================
st.markdown("---")
st.subheader("📝 Strategic Recommendations")

# Recommendation logic based on critical ratio
if optimization_result['critical_ratio'] > 0.8:
    st.success(f"""
    🚀 **HIGH MARGIN PRODUCT** (CR={optimization_result['critical_ratio']:.2f})
    - Order aggressively: {optimization_result['optimal_quantity']:,} units
    - Low risk of overstocking (high profit margins justify excess inventory)
    - Recommended: Secure supplier contracts immediately
    """)
elif optimization_result['critical_ratio'] < 0.5:
    st.warning(f"""
    ⚠️ **LOW MARGIN PRODUCT** (CR={optimization_result['critical_ratio']:.2f})
    - Conservative ordering: {optimization_result['optimal_quantity']:,} units
    - High risk of markdown losses (low margins cannot absorb excess)
    - Recommended: Consider pre-orders or made-to-order
    """)
else:
    st.info(f"""
    ✅ **BALANCED PRODUCT** (CR={optimization_result['critical_ratio']:.2f})
    - Standard ordering: {optimization_result['optimal_quantity']:,} units
    - Moderate risk profile
    - Recommended: Monitor early sales signals
    """)

# Demand level alert
if total_forecast_mean > 5000:
    st.error(f"🔥 **HIGH DEMAND ALERT**: Projected sales exceed {int(total_forecast_mean):,} units. Validate supplier capacity.")
elif total_forecast_mean < 1000:
    st.warning("📉 **LOW DEMAND**: Consider promotional discounts or bundling.")

# ========================================
# COMPARISON TABLE
# ========================================
st.markdown("---")
st.subheader("📊 Ordering Strategy Comparison")

comparison_data = {
    'Strategy': [
        'Order = Forecast Mean',
        'Order = Upper Bound (Conservative)',
        'TrendFlow (Newsvendor Optimal)'
    ],
    'Order Quantity': [
        int(total_forecast_mean),
        int(forecast_upper_sum),
        optimization_result['optimal_quantity']
    ],
    'Service Level': [
        '50%',
        '97.5%',
        f"{optimization_result['service_level']}%"
    ],
    'Expected Profit': [
        f"${int(total_forecast_mean) * profit_margin:,.2f}",
        f"${int(forecast_upper_sum) * profit_margin - (forecast_upper_sum - total_forecast_mean) * (unit_cost - salvage_value):,.2f}",
        f"${optimization_result['expected_profit']:,.2f}"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True)

# ========================================
# FOOTER
# ========================================
st.markdown("---")
st.caption("""
📖 **Methodology**: This system integrates Facebook Prophet (time-series forecasting) with the Newsvendor Model 
(Operations Research) to recommend profit-maximizing order quantities. Unlike traditional approaches that optimize 
for forecast accuracy, TrendFlow optimizes for business outcomes.

🔬 **References**: 
- Schweitzer & Cachon (2000). "Decision Bias in the Newsvendor Problem with Known Demand Distribution"
- Taylor & Letham (2018). "Forecasting at Scale" (Facebook Prophet)
""")

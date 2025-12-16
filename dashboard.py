
import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import random

st.set_page_config(page_title="TrendFlow: Supply Chain AI", page_icon="📈", layout="wide")

st.title("📈 TrendFlow: AI Demand Forecasting")
st.markdown("### Intelligent Inventory Planning for Hocam Couture")

st.sidebar.header("Configuration")
selected_product = st.sidebar.selectbox("Select Product to Forecast:", 
                                        ['Floral Dress', 'Leather Jacket', 'Denim Jeans', 'Chunky Sneaker'])
days_to_predict = st.sidebar.slider("Forecast Horizon (Days):", 30, 365, 90)

@st.cache_data
def load_data():
    dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
    products = ['Floral Dress', 'Leather Jacket', 'Denim Jeans', 'Chunky Sneaker']
    data = []
    
    for date in dates:
        month = date.month
        for product in products:
            base_mentions = random.randint(50, 200)
            if product == 'Floral Dress' and 4 <= month <= 8: base_mentions += random.randint(300, 500)
            elif product == 'Leather Jacket' and (month >= 10 or month <= 2): base_mentions += random.randint(200, 400)
            elif product == 'Denim Jeans': base_mentions += random.randint(50, 100)
            
            social_media_mentions = base_mentions
            actual_sales = int(social_media_mentions * 0.45) + random.randint(-20, 20)
            data.append([date, product, max(0, actual_sales)])
            
    return pd.DataFrame(data, columns=['Date', 'Product', 'Sales'])

df = load_data()
product_df = df[df['Product'] == selected_product].copy()

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader(f"📊 Historical Sales: {selected_product}")
    st.line_chart(product_df.set_index('Date')['Sales'])
with col2:
    st.info("ℹ️ **Model Logic:** Uses Facebook Prophet to detect seasonality.")

st.markdown("---")

with st.spinner(f"Training AI on {selected_product} patterns..."):
    df_prophet = product_df.rename(columns={'Date': 'ds', 'Sales': 'y'})
    m = Prophet()
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=days_to_predict)
    forecast = m.predict(future)

st.subheader(f"🔮 AI Forecast: Next {days_to_predict} Days")
fig = plot_plotly(m, forecast)
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

future_data = forecast.tail(days_to_predict)
total_predicted_demand = int(future_data['yhat'].sum())
avg_daily_sales = int(future_data['yhat'].mean())

st.markdown("### 📝 Deployment Recommendation")
metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric(label="Total Projected Sales", value=f"{total_predicted_demand} units")
metric_col2.metric(label="Avg Daily Demand", value=f"{avg_daily_sales} units/day")
metric_col3.metric(label="Recommended Order", value=f"{int(total_predicted_demand * 1.1)} units", delta="Includes 10% Buffer")

if total_predicted_demand > 5000:
    st.success("🚀 HIGH DEMAND ALERT: Secure supplier contracts immediately.")
elif total_predicted_demand < 1000:
    st.warning("📉 LOW DEMAND: Consider discounts.")
else:
    st.info("✅ STABLE DEMAND: Standard re-stocking.")

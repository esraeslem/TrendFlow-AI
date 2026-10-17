"""
TrendFlow Dashboard
-------------------
Run with:  streamlit run dashboard.py

Author: Esra Eslem Savaş
"""

import pandas as pd
import streamlit as st
from plotly import graph_objs as go

from src.data_processing import PRODUCTS, load_sales_data, product_series
from src.forecasting import ProphetForecaster
from src.optimization import NewsvendorOptimizer

st.set_page_config(page_title="TrendFlow: Supply Chain AI", page_icon="📈", layout="wide")
st.title("📈 TrendFlow: AI-Powered Supply Chain Optimizer")
st.markdown("### Profit-Aware Inventory Planning with the Newsvendor Model")

# ========================================
# SIDEBAR
# ========================================
st.sidebar.header("⚙️ Configuration")
selected_product = st.sidebar.selectbox("Select Product to Forecast:", PRODUCTS)
days_to_predict = st.sidebar.slider("Planning period (days):", 7, 180, 30,
                                    help="One order is placed to cover this whole period")

st.sidebar.markdown("### 💰 Unit Economics")
unit_cost = st.sidebar.number_input("Unit Cost ($)", min_value=1.0, max_value=1000.0, value=50.0, step=5.0)
unit_price = st.sidebar.number_input("Unit Price ($)", min_value=1.0, max_value=2000.0, value=120.0, step=5.0)
salvage_value = st.sidebar.number_input("Salvage Value ($)", min_value=0.0, max_value=1000.0, value=10.0, step=5.0,
                                        help="Recovery value for each unsold unit")

try:
    optimizer = NewsvendorOptimizer(unit_cost=unit_cost, unit_price=unit_price, salvage_value=salvage_value)
except ValueError as e:
    st.error(f"Invalid unit economics: {e}. Require salvage < cost < price.")
    st.stop()


# ========================================
# DATA + MODEL (cached)
# ========================================
@st.cache_data
def load_data() -> pd.DataFrame:
    return load_sales_data()  # generates the synthetic dataset if it is missing


@st.cache_resource(show_spinner=False)
def fit_forecaster(product: str) -> ProphetForecaster:
    return ProphetForecaster().fit(product_series(load_data(), product))


@st.cache_data(show_spinner=False)
def forecast(product: str, periods: int):
    f = fit_forecaster(product)
    return f.predict(periods), f.forecast_total_demand(periods)


df = load_data()
product_df = product_series(df, selected_product)

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader(f"📊 Historical Sales: {selected_product}")
    st.line_chart(product_df.set_index("Date")["Sales"])
with col2:
    st.info("""
    ℹ️ **Model Architecture:**
    - **Stage 1**: Prophet forecast of daily demand
    - **Stage 2**: Newsvendor model on the predictive distribution of *total* demand for the period
    """)

st.markdown("---")
with st.spinner(f"🤖 Training Prophet on {selected_product} demand patterns..."):
    daily_fc, total = forecast(selected_product, days_to_predict)

st.subheader(f"🔮 Forecast: Next {days_to_predict} Days")
fig = go.Figure()
fig.add_trace(go.Scatter(x=product_df["Date"], y=product_df["Sales"], mode="markers", name="Actual",
                         marker=dict(size=3, color="black", opacity=0.5)))
fig.add_trace(go.Scatter(x=daily_fc["ds"], y=daily_fc["yhat_upper"], line=dict(width=0),
                         showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=daily_fc["ds"], y=daily_fc["yhat_lower"], line=dict(width=0), fill="tonexty",
                         fillcolor="rgba(0,114,178,0.2)", name="95% interval (daily)"))
fig.add_trace(go.Scatter(x=daily_fc["ds"], y=daily_fc["yhat"], name="Forecast", line=dict(color="#0072B2")))
fig.update_layout(height=450, xaxis_title="Date", yaxis_title="Units / day", hovermode="x unified")
st.plotly_chart(fig, width="stretch")

# ========================================
# NEWSVENDOR OPTIMISATION
# ========================================
st.markdown("---")
st.subheader("🎯 Profit Optimization (Newsvendor Model)")

samples = total["samples"]
result = optimizer.optimize_samples(samples)
profit_margin = unit_price - unit_cost

m1, m2, m3, m4 = st.columns(4)
m1.metric("📈 Forecasted Demand", f"{total['mean']:,.0f} units",
          help=f"95% interval: {total['lower']:,.0f} – {total['upper']:,.0f}")
m2.metric("🎯 Optimal Order Quantity", f"{result['optimal_quantity']:,} units",
          delta=f"{result['optimal_quantity'] - round(total['mean']):+,} vs forecast")
m3.metric("💰 Expected Profit", f"${result['expected_profit']:,.0f}",
          help="Averaged over Prophet's predictive distribution of demand")
m4.metric("📊 In-stock Probability", f"{result['service_level']}%",
          help="Probability the period ends without a stockout (equals the critical ratio at the optimum)")

st.markdown("### 💡 Economic Analysis")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"""
    **Unit Economics:**
    - **Cost per unit**: ${unit_cost:.2f}
    - **Price per unit**: ${unit_price:.2f}
    - **Margin per unit**: ${profit_margin:.2f} ({profit_margin / unit_price * 100:.1f}%)
    - **Salvage value**: ${salvage_value:.2f}
    """)
with c2:
    st.markdown(f"""
    **Optimization Parameters:**
    - **Critical Ratio**: {optimizer.critical_ratio:.3f}
    - **Overstocking cost (c_o)**: ${optimizer.overstock_cost:.2f}/unit
    - **Understocking cost (c_u)**: ${optimizer.understock_cost:.2f}/unit
    - **Expected fill rate**: {result['fill_rate']}%
    """)

# ========================================
# RECOMMENDATIONS
# ========================================
st.markdown("---")
st.subheader("📝 Strategic Recommendations")
cr = optimizer.critical_ratio
if cr > 0.8:
    st.success(f"🚀 **HIGH MARGIN PRODUCT** (CR={cr:.2f}) — order above the forecast "
               f"({result['optimal_quantity']:,} units): a lost sale costs far more than a leftover unit.")
elif cr < 0.5:
    st.warning(f"⚠️ **LOW MARGIN PRODUCT** (CR={cr:.2f}) — order below the forecast "
               f"({result['optimal_quantity']:,} units): leftovers cost more than lost sales. "
               "Consider pre-orders or made-to-order.")
else:
    st.info(f"✅ **BALANCED PRODUCT** (CR={cr:.2f}) — order close to the forecast "
            f"({result['optimal_quantity']:,} units) and monitor early sales signals.")

# ========================================
# STRATEGY COMPARISON (all evaluated on the same demand distribution)
# ========================================
st.markdown("---")
st.subheader("📊 Ordering Strategy Comparison")
strategies = {
    "Order = forecast mean": total["mean"],
    "Order = upper 97.5% bound": total["upper"],
    "TrendFlow (newsvendor optimum)": result["order_quantity_exact"],
}
rows = []
for name, q in strategies.items():
    out = optimizer.expected_outcome_samples(q, samples)
    rows.append({"Strategy": name, "Order Quantity": f"{round(q):,}",
                 "In-stock Probability": f"{out['service_level']}%",
                 "Expected Profit": f"${out['expected_profit']:,.0f}"})
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
st.caption("Expected profit for each strategy is averaged over the same Prophet predictive samples, "
           "so the newsvendor optimum is by construction the highest. Backtested results on held-out "
           "data are in experiments/results/summary.md.")

st.markdown("---")
st.caption("""
📖 **Methodology**: Prophet forecasts daily demand; posterior predictive samples are summed over the
planning period to get the distribution of total demand, and the order is set at its critical-ratio quantile.

🔬 **References**: Arrow, Harris & Marschak (1951); Taylor & Letham (2018), "Forecasting at Scale".
""")

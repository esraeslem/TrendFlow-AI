"""
TrendFlow Dashboard
-------------------
Run with:  streamlit run dashboard.py

Tabs
* Executive summary - one recommendation per product, portfolio totals, editable unit economics
* Product deep-dive  - forecast chart, newsvendor detail and strategy comparison for one product

Data: the bundled synthetic dataset, or upload any CSV with columns Date, Product, Sales
(e.g. the H&M aggregate produced by prepare_hm.py).

Author: Esra Eslem Savaş
"""

import json
import os

import pandas as pd
import streamlit as st
from plotly import graph_objs as go

from src.data_processing import load_sales_data, mask_stockouts, product_series
from src.forecasting import ProphetForecaster
from src.optimization import NewsvendorOptimizer

ECONOMICS_FILE = os.path.join("experiments", "economics.json")
DEFAULT_ECON = {"price": 120.0, "cost": 50.0, "salvage": 10.0}

st.set_page_config(page_title="TrendFlow: Supply Chain AI", page_icon="📈", layout="wide")
st.title("📈 TrendFlow: AI-Powered Supply Chain Optimizer")
st.caption("Prophet demand forecasts → newsvendor order quantities that maximise expected profit")


# ========================================
# DATA + MODELS (cached)
# ========================================
@st.cache_data(show_spinner=False)
def load_data(csv_bytes: bytes | None) -> pd.DataFrame:
    if csv_bytes is None:
        return load_sales_data()  # bundled synthetic data (generated if missing)
    import io

    df = pd.read_csv(io.BytesIO(csv_bytes), parse_dates=["Date"])
    missing = {"Date", "Product", "Sales"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    return df


@st.cache_resource(show_spinner=False)
def fit_forecaster(csv_bytes: bytes | None, product: str) -> ProphetForecaster:
    series = mask_stockouts(product_series(load_data(csv_bytes), product))  # no-op without InStock
    return ProphetForecaster().fit(series[["Date", "Sales"]])


@st.cache_data(show_spinner=False)
def forecast(csv_bytes: bytes | None, product: str, periods: int):
    f = fit_forecaster(csv_bytes, product)
    return f.predict(periods), f.forecast_total_demand(periods)


def default_economics(products) -> pd.DataFrame:
    table = {}
    if os.path.exists(ECONOMICS_FILE):
        with open(ECONOMICS_FILE) as fh:
            table = json.load(fh)
    rows = [{"Product": p, **{k: float(v) for k, v in table.get(p, DEFAULT_ECON).items()}} for p in products]
    return pd.DataFrame(rows).set_index("Product")[["price", "cost", "salvage"]]


# ========================================
# SIDEBAR
# ========================================
st.sidebar.header("⚙️ Configuration")
upload = st.sidebar.file_uploader("Sales data (CSV: Date, Product, Sales)", type="csv",
                                  help="Leave empty to use the bundled synthetic dataset")
csv_bytes = upload.getvalue() if upload else None
try:
    df = load_data(csv_bytes)
except ValueError as e:
    st.error(str(e))
    st.stop()
products = sorted(df["Product"].unique())
days_to_predict = st.sidebar.slider("Planning period (days)", 7, 180, 30,
                                    help="One order is placed to cover this whole period")
st.sidebar.caption(f"{len(products)} products · {df['Date'].min():%d %b %Y} – {df['Date'].max():%d %b %Y}")

tab_exec, tab_detail = st.tabs(["📋 Executive summary", "🔍 Product deep-dive"])

# ========================================
# EXECUTIVE SUMMARY
# ========================================
with tab_exec:
    st.markdown(f"#### Order plan for the next {days_to_predict} days")
    with st.expander("💰 Unit economics per product (editable)", expanded=False):
        st.caption("Price, unit cost and salvage value (what an unsold unit is worth at the end of the period).")
        econ = st.data_editor(default_economics(products), width="stretch", key="econ")

    rows, problems = [], []
    progress = st.progress(0.0, text="Forecasting demand...")
    for i, p in enumerate(products):
        e = econ.loc[p]
        try:
            opt = NewsvendorOptimizer(unit_cost=e.cost, unit_price=e.price, salvage_value=e.salvage)
        except ValueError:
            problems.append(p)
            continue
        _, total = forecast(csv_bytes, p, days_to_predict)
        best = opt.optimize_samples(total["samples"])
        naive = opt.expected_outcome_samples(total["mean"], total["samples"])
        rows.append({
            "Product": p,
            "Forecast demand": total["mean"],
            "95% range": f"{total['lower']:,.0f} – {total['upper']:,.0f}",
            "Recommended order": best["optimal_quantity"],
            "vs. forecast": best["optimal_quantity"] - round(total["mean"]),
            "Expected profit": best["expected_profit"],
            "Gain vs. ordering the forecast": best["expected_profit"] - naive["expected_profit"],
            "In-stock probability": best["service_level"],
            "Expected unsold units": best["expected_leftover"],
            "Critical ratio": opt.critical_ratio,
        })
        progress.progress((i + 1) / len(products), text=f"Forecasting demand... {p}")
    progress.empty()
    if problems:
        st.warning(f"Skipped {', '.join(problems)}: unit economics must satisfy salvage < cost < price.")

    if rows:
        plan = pd.DataFrame(rows)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Units to order", f"{plan['Recommended order'].sum():,.0f}")
        c2.metric("Expected profit", f"${plan['Expected profit'].sum():,.0f}")
        c3.metric("Gain vs. ordering the forecast", f"${plan['Gain vs. ordering the forecast'].sum():,.0f}",
                  help="Expected extra profit from ordering the newsvendor quantity instead of the mean forecast")
        c4.metric("Expected unsold units", f"{plan['Expected unsold units'].sum():,.0f}")

        st.dataframe(
            plan.drop(columns=["Critical ratio"]), hide_index=True, width="stretch",
            column_config={
                "Forecast demand": st.column_config.NumberColumn(format="%.0f"),
                "Recommended order": st.column_config.NumberColumn(format="%d"),
                "vs. forecast": st.column_config.NumberColumn(format="%+d"),
                "Expected profit": st.column_config.NumberColumn(format="$%.0f"),
                "Gain vs. ordering the forecast": st.column_config.NumberColumn(format="$%.0f"),
                "In-stock probability": st.column_config.NumberColumn(format="%.0f%%"),
                "Expected unsold units": st.column_config.NumberColumn(format="%.0f"),
            })

        # Plain-language takeaways
        st.markdown("##### Key takeaways")
        above = plan[plan["vs. forecast"] > 0]["Product"].tolist()
        below = plan[plan["vs. forecast"] < 0]["Product"].tolist()
        top = plan.sort_values("Expected profit", ascending=False).iloc[0]
        widest = plan.assign(rel=(plan["Recommended order"] - plan["Forecast demand"]).abs()
                             / plan["Forecast demand"]).sort_values("rel", ascending=False).iloc[0]
        notes = [f"**{top['Product']}** is the largest profit driver "
                 f"(${top['Expected profit']:,.0f}, {top['Expected profit'] / plan['Expected profit'].sum():.0%} "
                 "of the total)."]
        if above:
            notes.append(f"Order **above** the forecast for {', '.join(above)}: a lost sale costs more "
                         "than a leftover unit.")
        if below:
            notes.append(f"Order **below** the forecast for {', '.join(below)}: leftovers are expensive "
                         "relative to the margin.")
        notes.append(f"The biggest adjustment is **{widest['Product']}** "
                     f"({widest['vs. forecast']:+,} units vs. the forecast).")
        st.markdown("\n".join(f"- {n}" for n in notes))
        st.caption("Expected values are averaged over Prophet's predictive distribution. Backtests show "
                   "Prophet's intervals are somewhat too narrow, so treat the in-stock probabilities as optimistic.")

# ========================================
# PRODUCT DEEP-DIVE
# ========================================
with tab_detail:
    selected_product = st.selectbox("Product", products)
    e = econ.loc[selected_product]
    try:
        optimizer = NewsvendorOptimizer(unit_cost=e.cost, unit_price=e.price, salvage_value=e.salvage)
    except ValueError as err:
        st.error(f"Invalid unit economics for {selected_product}: {err}")
        st.stop()
    unit_price, unit_cost, salvage_value = e.price, e.cost, e.salvage
    product_df = product_series(df, selected_product)

    with st.spinner(f"Training Prophet on {selected_product}..."):
        daily_fc, total = forecast(csv_bytes, selected_product, days_to_predict)

    st.subheader(f"🔮 Forecast: next {days_to_predict} days")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=product_df["Date"], y=product_df["Sales"], mode="markers", name="Actual",
                             marker=dict(size=3, color="black", opacity=0.5)))
    fig.add_trace(go.Scatter(x=daily_fc["ds"], y=daily_fc["yhat_upper"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=daily_fc["ds"], y=daily_fc["yhat_lower"], line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(0,114,178,0.2)", name="95% interval (daily)"))
    fig.add_trace(go.Scatter(x=daily_fc["ds"], y=daily_fc["yhat"], name="Forecast", line=dict(color="#0072B2")))
    fig.update_layout(height=420, xaxis_title="Date", yaxis_title="Units / day", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    st.subheader("🎯 Newsvendor recommendation")
    samples = total["samples"]
    result = optimizer.optimize_samples(samples)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Forecasted demand", f"{total['mean']:,.0f} units",
              help=f"95% interval: {total['lower']:,.0f} – {total['upper']:,.0f}")
    m2.metric("Optimal order", f"{result['optimal_quantity']:,} units",
              delta=f"{result['optimal_quantity'] - round(total['mean']):+,} vs forecast")
    m3.metric("Expected profit", f"${result['expected_profit']:,.0f}")
    m4.metric("In-stock probability", f"{result['service_level']}%",
              help="Probability the period ends without a stockout (equals the critical ratio at the optimum)")

    c1, c2 = st.columns(2)
    margin = unit_price - unit_cost
    c1.markdown(f"""
    **Unit economics**
    - Price ${unit_price:.2f} · cost ${unit_cost:.2f} · salvage ${salvage_value:.2f}
    - Margin ${margin:.2f} ({margin / unit_price * 100:.1f}%)
    """)
    c2.markdown(f"""
    **Optimisation**
    - Critical ratio {optimizer.critical_ratio:.3f}
    - Overstock cost c_o ${optimizer.overstock_cost:.2f}/unit · understock cost c_u ${optimizer.understock_cost:.2f}/unit
    - Expected fill rate {result['fill_rate']}%
    """)

    cr = optimizer.critical_ratio
    if cr > 0.8:
        st.success(f"🚀 **High-margin product** (CR={cr:.2f}): order above the forecast.")
    elif cr < 0.5:
        st.warning(f"⚠️ **Low-margin product** (CR={cr:.2f}): order below the forecast; "
                   "consider pre-orders or made-to-order.")
    else:
        st.info(f"✅ **Balanced product** (CR={cr:.2f}): order close to the forecast and monitor early sales.")

    st.subheader("📊 Ordering strategy comparison")
    strategies = {"Order = forecast mean": total["mean"], "Order = upper 97.5% bound": total["upper"],
                  "TrendFlow (newsvendor optimum)": result["order_quantity_exact"]}
    comp = []
    for name, q in strategies.items():
        out = optimizer.expected_outcome_samples(q, samples)
        comp.append({"Strategy": name, "Order quantity": f"{round(q):,}",
                     "In-stock probability": f"{out['service_level']}%",
                     "Expected profit": f"${out['expected_profit']:,.0f}"})
    st.dataframe(pd.DataFrame(comp), width="stretch", hide_index=True)
    st.caption("All strategies are evaluated on the same predictive samples. Backtested results on held-out "
               "data: experiments/results/summary.md.")

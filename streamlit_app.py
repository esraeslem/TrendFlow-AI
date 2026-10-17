"""
TrendFlow - Türkçe demo arayüzü (model karşılaştırmalı).
Çalıştırma:  streamlit run streamlit_app.py
"""

import pandas as pd
import plotly.graph_objs as go
import streamlit as st

from src.baselines import MovingAverageForecaster, NaiveForecaster
from src.data_processing import load_sales_data, product_series
from src.forecasting import ProphetForecaster
from src.optimization import NewsvendorOptimizer

st.set_page_config(page_title="TrendFlow-AI: Ultimate Dashboard", layout="wide")

METHODS = ["Prophet", "Naive (Son Değer)", "Hareketli Ortalama (7 Gün)"]


# --- FONKSİYONLAR ---
@st.cache_data
def load_data() -> pd.DataFrame:
    return load_sales_data()  # dosya yoksa sentetik veriyi üretir


@st.cache_resource(show_spinner=False)
def fit_prophet(product: str, weekly: bool, yearly: bool, changepoint: float) -> ProphetForecaster:
    return ProphetForecaster(weekly_seasonality=weekly, yearly_seasonality=yearly,
                             changepoint_prior_scale=changepoint).fit(product_series(load_data(), product))


def build_forecast(product, daily_df, periods, method, weekly=True, yearly=True, changepoint=0.05):
    """Gelecek ``periods`` gün için günlük tahmin ve (Prophet ise) toplam talep örnekleri döndürür."""
    if method == "Prophet":
        f = fit_prophet(product, weekly, yearly, changepoint)
        return f.predict(periods), f.forecast_total_demand(periods)["samples"]

    model = NaiveForecaster() if method == METHODS[1] else MovingAverageForecaster(7)
    yhat = model.fit(daily_df["Sales"]).predict(periods)
    future_dates = pd.date_range(daily_df["Date"].max() + pd.Timedelta(days=1), periods=periods)
    fc = pd.DataFrame({"ds": future_dates, "yhat": yhat})
    # Basit belirsizlik bandı: geçmiş günlük oynaklık (yalnızca görselleştirme için)
    sd = daily_df["Sales"].tail(30).std()
    fc["yhat_lower"] = (fc["yhat"] - 1.96 * sd).clip(lower=0)
    fc["yhat_upper"] = fc["yhat"] + 1.96 * sd
    return fc, None


def plot_forecast(forecast_df, daily_df, title=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat_upper"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat_lower"], mode="lines",
                             line=dict(width=0), fill="tonexty", fillcolor="rgba(0,100,80,0.2)",
                             name="%95 aralık"))
    fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat"], mode="lines", name="Tahmin",
                             line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=daily_df["Date"], y=daily_df["Sales"], mode="markers", name="Geçmiş Veri",
                             marker=dict(color="black", size=4, opacity=0.5)))
    fig.update_layout(title=title, xaxis_title="Tarih", yaxis_title="Satış Adedi", hovermode="x unified")
    return fig


# --- ANA UYGULAMA ---
def main():
    st.title("📈 TrendFlow: AI Tedarik Zinciri Optimizasyonu")
    df = load_data()

    st.sidebar.header("1. Tahmin Ayarları")
    product = st.sidebar.selectbox("Ürün Seçin", sorted(df["Product"].unique()))
    method = st.sidebar.selectbox("Tahmin Yöntemi", METHODS)
    horizon = st.sidebar.slider("Planlama Dönemi (Gün)", 7, 180, 30)

    weekly, yearly, changepoint = True, True, 0.05
    if method == "Prophet":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Prophet Hiperparametreleri")
        weekly = st.sidebar.checkbox("Haftalık Mevsimsellik", value=True)
        yearly = st.sidebar.checkbox("Yıllık Mevsimsellik", value=True)
        changepoint = st.sidebar.slider("Trend Değişim Hassasiyeti", 0.001, 0.5, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.header("2. Finansal Parametreler")
    price = st.sidebar.number_input("Birim Satış Fiyatı ($)", value=120.0)
    cost = st.sidebar.number_input("Birim Maliyet ($)", value=50.0)
    salvage = st.sidebar.number_input("Hurda Değeri ($)", value=10.0)
    try:
        optimizer = NewsvendorOptimizer(unit_cost=cost, unit_price=price, salvage_value=salvage)
    except ValueError as e:
        st.error(f"Geçersiz parametreler: {e} (hurda < maliyet < fiyat olmalı)")
        return

    daily = product_series(df, product)
    with st.spinner(f"{method} modeli çalıştırılıyor..."):
        forecast, samples = build_forecast(product, daily, horizon, method, weekly, yearly, changepoint)

    tab1, tab2, tab3 = st.tabs(["📊 Tahmin Analizi", "💰 Stok Optimizasyonu", "📂 Ham Veri"])

    with tab1:
        st.subheader(f"{product} için {horizon} Günlük Talep Tahmini")
        st.plotly_chart(plot_forecast(forecast, daily, title=f"Model: {method}"), width="stretch")

    with tab2:
        st.subheader("Newsvendor Kâr Maksimizasyonu")
        mu = forecast["yhat"].sum()
        if samples is not None:
            res = optimizer.optimize_samples(samples)
        else:
            # Baseline yöntemlerin olasılıksal tahmini yok: sipariş = ortalama tahmin
            res = optimizer.expected_outcome_normal(mu, mu, 0.0)

        col1, col2, col3 = st.columns(3)
        col1.metric("Tahmini Talep (Adet)", f"{mu:,.0f}")
        col2.metric("Optimal Sipariş (Q*)", f"{res['optimal_quantity']:,}",
                    delta=f"{res['optimal_quantity'] - round(mu):+,} güvenlik stoğu")
        col3.metric("Kritik Oran (hedef stokta kalma olasılığı)", f"%{optimizer.critical_ratio * 100:.1f}")
        st.success(f"Bu strateji ile beklenen kâr: **${res['expected_profit']:,.0f}**")

        if samples is None:
            st.warning("⚠️ Seçilen baseline yöntem talep belirsizliğini modellemez; sipariş ortalama tahmine "
                       "eşitlenir. Belirsizliği hesaba katan optimizasyon için Prophet'i seçin.")

    with tab3:
        st.dataframe(daily.sort_values("Date", ascending=False))
        st.download_button("Veriyi İndir (CSV)", daily.to_csv(index=False), "data.csv")


if __name__ == "__main__":
    main()

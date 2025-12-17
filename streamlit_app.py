import streamlit as st
import pandas as pd
import numpy as np
<<<<<<< HEAD
import io
import time
from data_generator import generate_fashion_data
import plotly.express as px
import plotly.graph_objs as go
=======
>>>>>>> d4490aac2358938a1280be3d62614f68967677c2
from prophet import Prophet
import plotly.graph_objs as go
from scipy.stats import norm

# Sayfa Ayarları
st.set_page_config(page_title="TrendFlow-AI: Ultimate Dashboard", layout="wide")

<<<<<<< HEAD
# Cache-trained Prophet model to avoid re-training for the same data + params
@st.cache_resource
def _trained_prophet_model(data_csv: str, weekly: bool, yearly: bool, changepoint: float):
    # data_csv is a CSV string of columns ['ds','y'] to make the cache key deterministic
    dfp = pd.read_csv(io.StringIO(data_csv))
    dfp['ds'] = pd.to_datetime(dfp['ds'])
    m = Prophet(weekly_seasonality=weekly, yearly_seasonality=yearly, changepoint_prior_scale=changepoint, stan_backend='CMDSTANPY')
    t0 = time.time()
    m.fit(dfp)
    elapsed = time.time() - t0
    return m, elapsed


=======
# --- FONKSİYONLAR ---
>>>>>>> d4490aac2358938a1280be3d62614f68967677c2
@st.cache_data
def load_data():
    try:
        # data_generator.py ile üretilen dosyayı oku
        df = pd.read_csv('data/raw/fashion_sales_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error("❌ Veri bulunamadı. Lütfen önce data_generator.py çalıştırın.")
        return pd.DataFrame()

def build_forecast(daily_df, periods, method='Prophet', weekly=True, yearly=True, changepoint=0.05):
    """
    Seçilen yönteme göre tahmin üretir.
    Prophet: Gelişmiş AI tahmini
    Naive/MA: Baseline karşılaştırma modelleri
    """
    dfp = daily_df.rename(columns={'Date': 'ds', 'Sales': 'y'})
    
    if method == 'Prophet':
        m = Prophet(weekly_seasonality=weekly, yearly_seasonality=yearly, changepoint_prior_scale=changepoint)
        m.fit(dfp)
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    # Baseline Yöntemler
    last_date = dfp['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    
    if method == 'Naive (Son Değer)':
        value = dfp['y'].iloc[-1]
        future_vals = [value] * periods
        std_dev = dfp['y'].std() # Basit belirsizlik aralığı için
    elif method == 'Hareketli Ortalama (7 Gün)':
        window = min(7, len(dfp))
        value = float(dfp['y'].tail(window).mean())
        future_vals = [value] * periods
        std_dev = dfp['y'].tail(window).std()
    else:
        future_vals = [0] * periods
        std_dev = 0
        
    future_df = pd.DataFrame({'ds': future_dates, 'yhat': future_vals})
    # Baseline için basit güven aralığı
    future_df['yhat_lower'] = future_df['yhat'] - 1.96 * std_dev
    future_df['yhat_upper'] = future_df['yhat'] + 1.96 * std_dev
    
    # Geçmiş veriyi formatla
    history_df = pd.DataFrame({'ds': dfp['ds'], 'yhat': dfp['y']})
    history_df['yhat_lower'] = history_df['yhat']
    history_df['yhat_upper'] = history_df['yhat']
    
    return pd.concat([history_df, future_df], ignore_index=True)

def plot_forecast(forecast_df, daily_df, title=None):
    fig = go.Figure()
    # Tahmin Çizgisi
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Tahmin', line=dict(color='blue')))
    # Belirsizlik Aralığı
    if 'yhat_lower' in forecast_df:
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,80,0.2)', showlegend=False, hoverinfo='skip'))
    # Gerçekleşen (Geçmiş)
    fig.add_trace(go.Scatter(x=daily_df['Date'], y=daily_df['Sales'], mode='markers', name='Geçmiş Veri', marker=dict(color='black', size=4, opacity=0.5)))
    
    fig.update_layout(title=title, xaxis_title='Tarih', yaxis_title='Satış Adedi', hovermode="x unified")
    return fig

# --- ANA UYGULAMA ---
def main():
    st.title('📈 TrendFlow: AI Tedarik Zinciri Optimizasyonu')
    
    df = load_data()
    if df.empty: return

    # --- SIDEBAR ---
    st.sidebar.header('1. Tahmin Ayarları')
    product = st.sidebar.selectbox('Ürün Seçin', sorted(df['Product'].unique()))
    method = st.sidebar.selectbox('Tahmin Yöntemi', ['Prophet', 'Naive (Son Değer)', 'Hareketli Ortalama (7 Gün)'])
    horizon = st.sidebar.slider('Tahmin Ufku (Gün)', 7, 180, 90)
    
    # Prophet Parametreleri (Sadece Prophet seçiliyse göster)
    weekly, yearly, changepoint = True, True, 0.05
    if method == 'Prophet':
        st.sidebar.markdown("---")
        st.sidebar.subheader("Prophet Hiperparametreleri")
        weekly = st.sidebar.checkbox('Haftalık Mevsimsellik', value=True)
        yearly = st.sidebar.checkbox('Yıllık Mevsimsellik', value=True)
        changepoint = st.sidebar.slider('Trend Değişim Hassasiyeti', 0.001, 0.5, 0.05)
    
    # Newsvendor Parametreleri (Yönetim Kısmı)
    st.sidebar.markdown("---")
    st.sidebar.header('2. Finansal Parametreler')
    price = st.sidebar.number_input("Birim Satış Fiyatı ($)", value=120.0)
    cost = st.sidebar.number_input("Birim Maliyet ($)", value=50.0)
    salvage = st.sidebar.number_input("Hurda Değeri ($)", value=10.0)
    
    # --- HESAPLAMA ---
    daily = df[df['Product'] == product].copy()
    
    with st.spinner(f'{method} modeli çalıştırılıyor...'):
        forecast = build_forecast(daily, horizon, method, weekly, yearly, changepoint)

    # --- SEKME YAPISI ---
    tab1, tab2, tab3 = st.tabs(["📊 Tahmin Analizi", "💰 Stok Optimizasyonu", "📂 Ham Veri"])
    
    with tab1:
        st.subheader(f"{product} için {horizon} Günlük Talep Tahmini")
        fig = plot_forecast(forecast, daily, title=f"Model: {method}")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("Newsvendor Kâr Maksimizasyonu")
        
        # Gelecek Dönem İstatistikleri
        future_data = forecast.tail(horizon)
        mu = future_data['yhat'].sum()
        
        # Standart sapma tahmini (Güven aralığından)
        # Sigma approx = (Upper - Lower) / 3.92 (Normal dağılım varsayımı ile %95 CI)
        sigma = ((future_data['yhat_upper'] - future_data['yhat_lower']) / 3.92).sum() if method == 'Prophet' else 0
        
        # Kritik Oran (Critical Ratio)
        cu = price - cost
        co = cost - salvage
        if cu + co > 0:
            cr = cu / (cu + co)
        else:
            cr = 0.5
            
        # Optimal Sipariş (Q*)
        if sigma > 0:
            optimal_order = norm.ppf(cr, loc=mu, scale=sigma)
        else:
            optimal_order = mu # Deterministik yöntemler için ortalama alınır
            
        # Sonuç Kartları
        col1, col2, col3 = st.columns(3)
        col1.metric("Tahmini Talep (Adet)", f"{int(mu):,}")
        col2.metric("Optimal Sipariş (Q*)", f"{int(optimal_order):,}", delta=f"{int(optimal_order - mu)} Güvenlik Stoğu")
        col3.metric("Hedef Hizmet Seviyesi", f"%{cr*100:.1f}")
        
        # Kâr Hesabı
        expected_profit = (min(mu, optimal_order) * price) - (optimal_order * cost) + (max(0, optimal_order - mu) * salvage)
        
        st.success(f"Bu strateji ile beklenen kâr: **${int(expected_profit):,}**")
        
        if method != 'Prophet':
            st.warning("⚠️ Not: Seçilen 'Baseline' yöntem belirsizliği (sigma) tam ölçemez. En iyi finansal optimizasyon için Prophet kullanın.")

    with tab3:
        st.dataframe(daily.sort_values('Date', ascending=False))
        st.download_button("Veriyi İndir (CSV)", daily.to_csv(), "data.csv")

if __name__ == "__main__":
    main()


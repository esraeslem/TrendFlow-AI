import streamlit as st
import pandas as pd
import numpy as np
import io
import time
from data_generator import generate_fashion_data
import plotly.express as px
import plotly.graph_objs as go
from prophet import Prophet


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


@st.cache_data
def load_data():
    path = 'data/raw/fashion_sales_data.csv'
    try:
        df = pd.read_csv(path)
    except Exception:
        df = generate_fashion_data(output_path=None, periods=365, seed=0)
    return df


def plot_series(df, product):
    dfp = df[df['Product'] == product].copy()
    dfp['Date'] = pd.to_datetime(dfp['Date'])
    daily = dfp.groupby('Date')['Sales'].sum().reset_index()
    fig = px.line(daily, x='Date', y='Sales', title=f'Sales - {product}')
    return fig, daily


def build_forecast(daily_df, periods=30, method='Prophet', weekly=True, yearly=True, changepoint=0.05):
    dfp = daily_df.rename(columns={'Date': 'ds', 'Sales': 'y'})
    dfp['ds'] = pd.to_datetime(dfp['ds'])

    if method == 'Prophet':
        m = Prophet(weekly_seasonality=weekly, yearly_seasonality=yearly, changepoint_prior_scale=changepoint)
        m.fit(dfp)
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # Baseline methods (Naive / Moving Average)
    last_date = dfp['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)

    if method == 'Naive (last)':
        value = dfp['y'].iloc[-1]
        future_vals = [value] * periods
    elif method == 'Moving Average (7d)':
        window = min(7, len(dfp))
        value = float(dfp['y'].tail(window).mean())
        future_vals = [value] * periods
    else:
        future_vals = [dfp['y'].iloc[-1]] * periods

    future_df = pd.DataFrame({'ds': future_dates, 'yhat': future_vals})
    future_df['yhat_lower'] = future_df['yhat'] * 0.9
    future_df['yhat_upper'] = future_df['yhat'] * 1.1

    history_df = pd.DataFrame({'ds': dfp['ds'], 'yhat': dfp['y']})
    history_df['yhat_lower'] = history_df['yhat']
    history_df['yhat_upper'] = history_df['yhat']

    full = pd.concat([history_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], future_df], ignore_index=True)
    return full


def plot_forecast(forecast_df, daily_df, title=None):
    fig = go.Figure()
    # yhat
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Forecast'))
    # uncertainty band
    if 'yhat_lower' in forecast_df and 'yhat_upper' in forecast_df:
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,80,0.2)', showlegend=False))
    # history
    fig.add_trace(go.Scatter(x=daily_df['Date'], y=daily_df['Sales'], mode='markers', name='History'))
    fig.update_layout(title=title or 'Forecast', xaxis_title='Date', yaxis_title='Sales')
    return fig


def main():
    st.title('TrendFlow Demo — Data generator & forecasting parameters')

    df = load_data()

    st.sidebar.header('Forecast options')
    product = st.sidebar.selectbox('Product', sorted(df['Product'].unique()))
    method = st.sidebar.selectbox('Method', ['Prophet', 'Naive (last)', 'Moving Average (7d)'])
    horizon = st.sidebar.slider('Horizon (days)', min_value=7, max_value=180, value=30, step=1)

    # Prophet-specific parameters
    weekly = True
    yearly = True
    changepoint = 0.05
    if method == 'Prophet':
        weekly = st.sidebar.checkbox('Weekly seasonality', value=True)
        yearly = st.sidebar.checkbox('Yearly seasonality', value=True)
        changepoint = st.sidebar.slider('Changepoint prior scale', 0.001, 0.5, 0.05)

    show_history = st.sidebar.checkbox('Show history points', value=True)

    fig, daily = plot_series(df, product)
    st.plotly_chart(fig, use_container_width=True)

    if st.sidebar.button('Run forecast'):
        with st.spinner('Running forecast...'):
            forecast = build_forecast(daily, periods=horizon, method=method, weekly=weekly, yearly=yearly, changepoint=changepoint)
        fig2 = plot_forecast(forecast, daily, title=f'{horizon}-day forecast for {product} ({method})')
        st.plotly_chart(fig2, use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button('Download CSV', csv, file_name='fashion_sales_data.csv', mime='text/csv')


if __name__ == '__main__':
    main()

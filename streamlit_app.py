import streamlit as st
import pandas as pd
from data_generator import generate_fashion_data
import plotly.express as px
from prophet import Prophet


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


def build_forecast(daily_df, periods=30):
    dfp = daily_df.rename(columns={'Date': 'ds', 'Sales': 'y'})
    m = Prophet()
    m.fit(dfp)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return forecast


def main():
    st.title('TrendFlow Demo — Data generator & simple forecasting')

    df = load_data()

    product = st.sidebar.selectbox('Product', sorted(df['Product'].unique()))

    fig, daily = plot_series(df, product)
    st.plotly_chart(fig, use_container_width=True)

    if st.sidebar.button('Run 30-day forecast'):
        with st.spinner('Training Prophet model...'):
            forecast = build_forecast(daily, periods=30)
        fig2 = px.line(forecast, x='ds', y='yhat', title=f'30-day forecast for {product}')
        fig2.add_scatter(x=daily['Date'], y=daily['Sales'], mode='markers', name='history')
        st.plotly_chart(fig2, use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button('Download CSV', csv, file_name='fashion_sales_data.csv', mime='text/csv')


if __name__ == '__main__':
    main()

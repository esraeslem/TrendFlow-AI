"""
Model Comparison Module for TrendFlow-AI
Compares Prophet vs ARIMA vs Moving Average
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ForecastComparison:
    """Compare multiple forecasting models"""
    
    def __init__(self, df, date_col='date', value_col='sales'):
        """
        Initialize with historical data
        
        Args:
            df: DataFrame with historical data
            date_col: Name of date column
            value_col: Name of value column (sales)
        """
        self.df = df.copy()
        self.date_col = date_col
        self.value_col = value_col
        self.results = {}
        
    def train_test_split(self, test_size=0.2):
        """Split data into train and test sets"""
        split_idx = int(len(self.df) * (1 - test_size))
        self.train = self.df[:split_idx]
        self.test = self.df[split_idx:]
        
    def forecast_prophet(self, periods=30):
        """Train Prophet model and forecast"""
        # Prepare data for Prophet
        df_prophet = self.train.rename(columns={
            self.date_col: 'ds',
            self.value_col: 'y'
        })
        
        # Train model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        model.fit(df_prophet)
        
        # Forecast
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    
    def forecast_arima(self, periods=30, order=(5,1,0)):
        """Train ARIMA model and forecast"""
        # Train model
        model = ARIMA(self.train[self.value_col], order=order)
        fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.forecast(steps=periods)
        
        # Create forecast dataframe
        last_date = self.train[self.date_col].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=periods,
            freq='D'
        )
        
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast.values
        })
    
    def forecast_moving_average(self, periods=30, window=7):
        """Simple Moving Average forecast"""
        # Calculate moving average
        ma = self.train[self.value_col].rolling(window=window).mean().iloc[-1]
        
        # Create forecast (naive: repeat last MA)
        last_date = self.train[self.date_col].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=periods,
            freq='D'
        )
        
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': [ma] * periods
        })
    
    def calculate_metrics(self, actual, predicted):
        """Calculate performance metrics"""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def compare_all_models(self, periods=30):
        """Run all models and compare"""
        # Get forecasts
        prophet_fc = self.forecast_prophet(periods)
        arima_fc = self.forecast_arima(periods)
        ma_fc = self.forecast_moving_average(periods)
        
        # Evaluate on test set if available
        if len(self.test) > 0:
            test_periods = min(periods, len(self.test))
            actual = self.test[self.value_col].values[:test_periods]
            
            # Get predictions for test period
            prophet_pred = prophet_fc['yhat'].values[:test_periods]
            arima_pred = arima_fc['yhat'].values[:test_periods]
            ma_pred = ma_fc['yhat'].values[:test_periods]
            
            # Calculate metrics
            metrics = {
                'Prophet': self.calculate_metrics(actual, prophet_pred),
                'ARIMA': self.calculate_metrics(actual, arima_pred),
                'Moving Average': self.calculate_metrics(actual, ma_pred)
            }
        else:
            metrics = None
        
        self.results = {
            'forecasts': {
                'Prophet': prophet_fc,
                'ARIMA': arima_fc,
                'Moving Average': ma_fc
            },
            'metrics': metrics
        }
        
        return self.results
    
    def plot_comparison(self):
        """Create interactive comparison plot"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Forecast Comparison', 'Model Performance Metrics'),
            row_heights=[0.7, 0.3],
            vertical_spacing=0.15
        )
        
        # Plot historical data
        fig.add_trace(
            go.Scatter(
                x=self.train[self.date_col],
                y=self.train[self.value_col],
                mode='lines',
                name='Historical',
                line=dict(color='gray', width=1)
            ),
            row=1, col=1
        )
        
        # Plot test data if available
        if len(self.test) > 0:
            fig.add_trace(
                go.Scatter(
                    x=self.test[self.date_col],
                    y=self.test[self.value_col],
                    mode='lines',
                    name='Actual (Test)',
                    line=dict(color='black', width=2, dash='dot')
                ),
                row=1, col=1
            )
        
        # Plot forecasts
        colors = {'Prophet': 'blue', 'ARIMA': 'red', 'Moving Average': 'green'}
        for model_name, forecast in self.results['forecasts'].items():
            fig.add_trace(
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name=f'{model_name} Forecast',
                    line=dict(color=colors[model_name], width=2)
                ),
                row=1, col=1
            )
        
        # Plot metrics comparison
        if self.results['metrics']:
            metrics_df = pd.DataFrame(self.results['metrics']).T
            
            for metric in ['RMSE', 'MAE', 'MAPE']:
                fig.add_trace(
                    go.Bar(
                        x=metrics_df.index,
                        y=metrics_df[metric],
                        name=metric
                    ),
                    row=2, col=1
                )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="TrendFlow Model Comparison Dashboard",
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Sales", row=1, col=1)
        fig.update_xaxes(title_text="Model", row=2, col=1)
        fig.update_yaxes(title_text="Error", row=2, col=1)
        
        return fig
    
    def get_best_model(self):
        """Determine best performing model"""
        if not self.results['metrics']:
            return None
        
        metrics_df = pd.DataFrame(self.results['metrics']).T
        
        # Lower is better for all metrics
        best_model = metrics_df['RMSE'].idxmin()
        
        return {
            'model': best_model,
            'rmse': metrics_df.loc[best_model, 'RMSE'],
            'improvement_vs_baseline': (
                (metrics_df.loc['Moving Average', 'RMSE'] - 
                 metrics_df.loc[best_model, 'RMSE']) / 
                metrics_df.loc['Moving Average', 'RMSE'] * 100
            )
        }


# Example usage in your Streamlit dashboard
def add_to_dashboard(df):
    """Add model comparison to your Streamlit dashboard"""
    import streamlit as st
    
    st.header("🔬 Model Comparison & Evaluation")
    
    # Create comparison
    comparison = ForecastComparison(df, date_col='date', value_col='sales')
    comparison.train_test_split(test_size=0.2)
    
    # Run comparison
    with st.spinner("Training multiple models..."):
        results = comparison.compare_all_models(periods=30)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    best = comparison.get_best_model()
    if best:
        with col1:
            st.metric("Best Model", best['model'])
        with col2:
            st.metric("RMSE", f"{best['rmse']:.2f}")
        with col3:
            st.metric("Improvement", f"{best['improvement_vs_baseline']:.1f}%", delta="better")
    
    # Show comparison plot
    fig = comparison.plot_comparison()
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed metrics table
    if results['metrics']:
        st.subheader("Detailed Performance Metrics")
        metrics_df = pd.DataFrame(results['metrics']).T
        st.dataframe(metrics_df.style.highlight_min(axis=0, color='lightgreen'))
        
        st.info("""
        📊 **Metric Definitions:**
        - **MAE**: Mean Absolute Error - Average prediction error
        - **RMSE**: Root Mean Squared Error - Penalizes large errors more
        - **MAPE**: Mean Absolute Percentage Error - Error as % of actual value
        
        Lower values = Better performance
        """)

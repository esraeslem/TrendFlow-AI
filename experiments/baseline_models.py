"""
Baseline models for comparison with TrendFlow.

Implements:
1. Naive Forecasting (last week = this week)
2. Moving Average (simple, weighted, exponential)
3. ARIMA (classical time-series)
4. LSTM (deep learning baseline)
5. Prophet without optimization (accuracy-only)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

class NaiveForecaster:
    """Last observation carried forward."""
    def fit(self, y):
        self.last_value = y.iloc[-1]
        
    def predict(self, steps):
        return [self.last_value] * steps

class MovingAverageForecaster:
    """Simple moving average."""
    def __init__(self, window=7):
        self.window = window
        
    def fit(self, y):
        self.ma = y.rolling(window=self.window).mean().iloc[-1]
        
    def predict(self, steps):
        return [self.ma] * steps

# Add ARIMA, LSTM, etc.

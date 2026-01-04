from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import numpy as np

app = FastAPI(
    title="TrendFlow-AI API",
    description="AI-Powered Fashion Demand Forecasting & Inventory Optimization",
    version="1.0.0"
)

# Pydantic models for request/response validation
class HistoricalData(BaseModel):
    date: List[str]
    sales: List[float]
    
class ForecastRequest(BaseModel):
    historical_data: HistoricalData
    forecast_periods: int = Field(default=30, ge=1, le=365, description="Number of days to forecast")
    
class NewsvendorRequest(BaseModel):
    forecasted_demand: float
    demand_std: float
    cost_price: float
    selling_price: float
    salvage_value: float = 0.0

class ForecastResponse(BaseModel):
    forecast_date: List[str]
    forecast_value: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    mean_forecast: float
    std_forecast: float

class NewsvendorResponse(BaseModel):
    optimal_order_quantity: float
    critical_ratio: float
    expected_profit: float
    risk_of_stockout: float

@app.get("/")
def home():
    """Health check endpoint"""
    return {
        "service": "TrendFlow-AI API",
        "status": "operational",
        "version": "1.0.0",
        "endpoints": ["/predict", "/optimize", "/docs"]
    }

@app.post("/predict", response_model=ForecastResponse)
def predict_demand(request: ForecastRequest):
    """
    Forecast future fashion demand using Facebook Prophet
    
    - **historical_data**: Past sales data with dates and values
    - **forecast_periods**: Number of days to forecast (default: 30)
    """
    try:
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': pd.to_datetime(request.historical_data.date),
            'y': request.historical_data.sales
        })
        
        # Train Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            interval_width=0.95
        )
        model.fit(df)
        
        # Make future predictions
        future = model.make_future_dataframe(periods=request.forecast_periods)
        forecast = model.predict(future)
        
        # Extract only future predictions (not historical)
        future_forecast = forecast.tail(request.forecast_periods)
        
        return ForecastResponse(
            forecast_date=future_forecast['ds'].dt.strftime('%Y-%m-%d').tolist(),
            forecast_value=future_forecast['yhat'].tolist(),
            lower_bound=future_forecast['yhat_lower'].tolist(),
            upper_bound=future_forecast['yhat_upper'].tolist(),
            mean_forecast=float(future_forecast['yhat'].mean()),
            std_forecast=float(future_forecast['yhat'].std())
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Forecasting error: {str(e)}")

@app.post("/optimize", response_model=NewsvendorResponse)
def optimize_inventory(request: NewsvendorRequest):
    """
    Calculate optimal order quantity using Newsvendor Model
    
    - **forecasted_demand**: Expected demand from forecast
    - **demand_std**: Standard deviation of demand
    - **cost_price**: Cost per unit
    - **selling_price**: Selling price per unit
    - **salvage_value**: Salvage value if unsold (default: 0)
    """
    try:
        # Newsvendor critical ratio
        underage_cost = request.selling_price - request.cost_price
        overage_cost = request.cost_price - request.salvage_value
        critical_ratio = underage_cost / (underage_cost + overage_cost)
        
        # Calculate optimal order quantity (assumes normal distribution)
        from scipy.stats import norm
        z_score = norm.ppf(critical_ratio)
        optimal_quantity = request.forecasted_demand + z_score * request.demand_std
        
        # Calculate expected profit (simplified)
        expected_profit = (
            request.selling_price * min(optimal_quantity, request.forecasted_demand)
            - request.cost_price * optimal_quantity
        )
        
        # Risk of stockout
        stockout_risk = 1 - critical_ratio
        
        return NewsvendorResponse(
            optimal_order_quantity=float(max(0, optimal_quantity)),
            critical_ratio=float(critical_ratio),
            expected_profit=float(expected_profit),
            risk_of_stockout=float(stockout_risk)
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Optimization error: {str(e)}")

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": True
    }

# To run this API locally:
# uvicorn api:app --reload --host 0.0.0.0 --port 8000

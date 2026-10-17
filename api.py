"""
TrendFlow REST API.

Run locally:  uvicorn api:app --reload --port 8000   (docs at http://localhost:8000/docs)
"""

from datetime import datetime
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from src.optimization import NewsvendorOptimizer

app = FastAPI(
    title="TrendFlow-AI API",
    description="Demand forecasting (Prophet) and newsvendor inventory optimisation",
    version="1.1.0",
)


# ----------------------------------------------------------------- schemas
class HistoricalData(BaseModel):
    date: List[str]
    sales: List[float]

    @model_validator(mode="after")
    def same_length(self):
        if len(self.date) != len(self.sales):
            raise ValueError("date and sales must have the same length")
        if len(self.date) < 30:
            raise ValueError("at least 30 daily observations are required")
        return self


class ForecastRequest(BaseModel):
    historical_data: HistoricalData
    forecast_periods: int = Field(default=30, ge=1, le=365, description="Number of days to forecast")


class ForecastResponse(BaseModel):
    forecast_date: List[str]
    forecast_value: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    total_demand_mean: float
    total_demand_std: float = Field(description="Std of total demand over the horizon (from predictive samples)")


class NewsvendorRequest(BaseModel):
    forecasted_demand: float = Field(ge=0, description="Expected total demand for the period")
    demand_std: float = Field(ge=0, description="Std of total demand for the period")
    cost_price: float = Field(gt=0)
    selling_price: float = Field(gt=0)
    salvage_value: float = Field(default=0.0, ge=0)


class NewsvendorResponse(BaseModel):
    optimal_order_quantity: float
    critical_ratio: float
    expected_profit: float
    expected_fill_rate: float
    risk_of_stockout: float


# --------------------------------------------------------------- endpoints
@app.get("/")
def home():
    return {"service": "TrendFlow-AI API", "status": "operational", "version": app.version,
            "endpoints": ["/predict", "/optimize", "/health", "/docs"]}


@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/predict", response_model=ForecastResponse)
def predict_demand(request: ForecastRequest):
    """Fit Prophet on the supplied daily history and forecast the next ``forecast_periods`` days."""
    from src.forecasting import ProphetForecaster

    try:
        df = pd.DataFrame({"ds": pd.to_datetime(request.historical_data.date),
                           "y": request.historical_data.sales})
        f = ProphetForecaster().fit(df)
        fc = f.predict(request.forecast_periods)
        total = f.forecast_total_demand(request.forecast_periods)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=f"Forecasting error: {e}")

    return ForecastResponse(
        forecast_date=fc["ds"].dt.strftime("%Y-%m-%d").tolist(),
        forecast_value=fc["yhat"].tolist(),
        lower_bound=fc["yhat_lower"].tolist(),
        upper_bound=fc["yhat_upper"].tolist(),
        total_demand_mean=total["mean"],
        total_demand_std=total["std"],
    )


@app.post("/optimize", response_model=NewsvendorResponse)
def optimize_inventory(request: NewsvendorRequest):
    """Newsvendor optimum assuming total demand ~ Normal(forecasted_demand, demand_std)."""
    try:
        opt = NewsvendorOptimizer(unit_cost=request.cost_price, unit_price=request.selling_price,
                                  salvage_value=request.salvage_value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    res = opt.optimize_normal(request.forecasted_demand, request.demand_std)
    return NewsvendorResponse(
        optimal_order_quantity=res["order_quantity_exact"],
        critical_ratio=opt.critical_ratio,
        expected_profit=res["expected_profit"],
        expected_fill_rate=res["fill_rate"],
        risk_of_stockout=round(100 - res["service_level"], 1) / 100,
    )

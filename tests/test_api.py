import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "healthy"


def test_optimize_uses_salvage_and_normal_quantile():
    r = client.post("/optimize", json={"forecasted_demand": 1000, "demand_std": 100,
                                       "cost_price": 50, "selling_price": 120, "salvage_value": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["critical_ratio"] == pytest.approx(70 / 110)
    assert body["optimal_order_quantity"] == pytest.approx(1034.9, abs=0.5)
    assert body["risk_of_stockout"] == pytest.approx(1 - 70 / 110, abs=0.01)


def test_optimize_rejects_bad_economics():
    r = client.post("/optimize", json={"forecasted_demand": 1000, "demand_std": 100,
                                       "cost_price": 50, "selling_price": 40})
    assert r.status_code == 400


def test_predict_validates_lengths():
    r = client.post("/predict", json={"historical_data": {"date": ["2024-01-01"], "sales": [1, 2]}})
    assert r.status_code == 422


@pytest.mark.slow
def test_predict():
    dates = pd.date_range("2024-01-01", periods=120).strftime("%Y-%m-%d").tolist()
    sales = [100 + (i % 7) * 5 for i in range(120)]
    r = client.post("/predict", json={"historical_data": {"date": dates, "sales": sales}, "forecast_periods": 14})
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecast_value"]) == 14 and body["total_demand_std"] > 0

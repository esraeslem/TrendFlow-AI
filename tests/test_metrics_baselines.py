import numpy as np
import pandas as pd
import pytest

from src.baselines import (ARIMAForecaster, MovingAverageForecaster, NaiveForecaster,
                           SeasonalNaiveForecaster)
from src.metrics import inventory_report, mae, mape, r2, rmse


def test_metrics():
    y, p = [10, 20, 0, 40], [12, 18, 1, 40]
    assert mae(y, p) == pytest.approx(5 / 4)
    assert rmse(y, p) == pytest.approx(np.sqrt(9 / 4))
    assert mape(y, p) == pytest.approx((20 + 10 + 0) / 3)  # zero actual is skipped
    assert r2(y, y) == 1.0


def test_inventory_report():
    r = inventory_report([100, 100], [80, 120])
    assert r["stockout_rate"] == 50 and r["service_level"] == 50
    assert r["fill_rate"] == pytest.approx(180 / 200 * 100)


@pytest.fixture
def series():
    rng = np.random.default_rng(0)
    return pd.Series(100 + 10 * np.sin(np.arange(400) * 2 * np.pi / 7) + rng.normal(0, 1, 400))


def test_baselines_shapes(series):
    for m in (NaiveForecaster(), MovingAverageForecaster(7), SeasonalNaiveForecaster(7),
              SeasonalNaiveForecaster(365), ARIMAForecaster((1, 0, 0))):
        out = m.fit(series).predict(30)
        assert out.shape == (30,) and np.isfinite(out).all()


def test_seasonal_naive_repeats_season(series):
    out = SeasonalNaiveForecaster(7).fit(series).predict(14)
    assert np.allclose(out[:7], series.iloc[-7:]) and np.allclose(out[7:], series.iloc[-7:])

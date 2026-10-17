import numpy as np
import pytest

from src.data_processing import generate_fashion_data, product_series
from src.forecasting import ProphetForecaster


@pytest.fixture(scope="module")
def fitted():
    df = generate_fashion_data(output_path=None)
    return ProphetForecaster(uncertainty_samples=300).fit(product_series(df, "Leather Jacket"))


@pytest.mark.slow
def test_predict_future_only(fitted):
    fc = fitted.predict(30)
    assert len(fc) == 30
    assert fc["ds"].min() > fitted.history["ds"].max()
    assert (fc["yhat_lower"] <= fc["yhat"]).all() and (fc["yhat"] <= fc["yhat_upper"]).all()


@pytest.mark.slow
def test_total_demand_interval_narrower_than_summed_daily_bounds(fitted):
    fc = fitted.predict(30)
    total = fitted.forecast_total_demand(30)
    assert total["lower"] < total["mean"] < total["upper"]
    assert (total["upper"] - total["lower"]) < (fc["yhat_upper"].sum() - fc["yhat_lower"].sum())
    assert total["samples"].shape == (300,)


@pytest.mark.slow
def test_reproducible(fitted):
    assert np.array_equal(fitted.total_demand_samples(10), fitted.total_demand_samples(10))


def test_rejects_multi_product_frame():
    df = generate_fashion_data(output_path=None, periods=60)
    with pytest.raises(ValueError, match="Duplicate dates"):
        ProphetForecaster().fit(df)


def test_predict_before_fit():
    with pytest.raises(RuntimeError):
        ProphetForecaster().predict(5)

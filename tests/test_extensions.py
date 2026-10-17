"""LightGBM baseline, censored demand, H&M preparation and the benchmark pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.baselines import LightGBMForecaster
from src.data_processing import fill_stockouts, generate_fashion_data, mask_stockouts, product_series
from src.forecasting import ProphetForecaster


def test_lightgbm_forecast_shape_and_no_leakage():
    s = product_series(generate_fashion_data(None), "Floral Dress").set_index("Date")["Sales"]
    m = LightGBMForecaster(horizon=30).fit(s.iloc[:500])
    out = m.predict(30)
    assert out.shape == (30,) and (out >= 0).all()
    with pytest.raises(ValueError):
        m.predict(31)  # lags would need unseen data


def test_stockout_simulation_censors_sales():
    df = generate_fashion_data(None, stockout_prob=0.2)
    assert {"Demand", "InStock"} <= set(df.columns)
    assert (df["Sales"] <= df["Demand"]).all()
    assert (df.loc[df.InStock, "Sales"] == df.loc[df.InStock, "Demand"]).all()
    assert 0.15 < (~df.InStock).mean() < 0.25
    # default dataset is unaffected
    assert list(generate_fashion_data(None).columns) == ["Date", "Product", "Sales"]


def test_mask_and_fill_stockouts():
    s = product_series(generate_fashion_data(None, periods=60, stockout_prob=0.3), "Denim Jeans")
    masked = mask_stockouts(s)
    assert masked.loc[~s.InStock, "Sales"].isna().all()
    assert fill_stockouts(s)["Sales"].notna().all()


@pytest.mark.slow
def test_prophet_horizon_starts_after_last_date_even_if_masked():
    s = product_series(generate_fashion_data(None, periods=200), "Denim Jeans")
    s["InStock"] = True
    s.loc[s.index[-1], "InStock"] = False          # last day is a stockout
    f = ProphetForecaster(uncertainty_samples=100).fit(mask_stockouts(s)[["Date", "Sales"]])
    assert f.predict(5)["ds"].iloc[0] == s["Date"].iloc[-1] + pd.Timedelta(days=1)
    assert len(f.history) == len(s) - 1


def test_prepare_hm_aggregate(tmp_path):
    from prepare_hm import aggregate, select_top

    pd.DataFrame({"article_id": ["0101", "0102", "0201"],
                  "product_type_name": ["Dress", "Dress", "Socks"]}).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame({"t_dat": ["2020-01-01", "2020-01-01", "2020-01-03", "2020-01-01"],
                  "customer_id": "c", "article_id": ["0101", "0102", "0101", "0201"],
                  "price": 0.03, "sales_channel_id": 2}).to_csv(tmp_path / "t.csv", index=False)
    daily = aggregate(str(tmp_path / "t.csv"), str(tmp_path / "a.csv"), chunksize=2)
    dress = daily[daily.Product == "Dress"].set_index("Date")["Sales"]
    assert dress.tolist() == [2, 0, 1]                  # missing day filled with 0
    assert select_top(daily, top=1, min_daily_mean=0)["Product"].unique().tolist() == ["Dress"]


@pytest.mark.slow
def test_benchmark_runs_with_per_product_economics(tmp_path):
    import json
    import run_benchmark

    econ = tmp_path / "econ.json"
    econ.write_text(json.dumps({"Floral Dress": {"price": 60, "cost": 50, "salvage": 0}}))
    args = run_benchmark.parse_args(["--products", "Floral Dress", "--no-arima", "--first-origin", "640",
                                     "--step", "30", "--economics", str(econ), "--out", str(tmp_path)])
    res = run_benchmark.run(args)
    assert res["per_product"]["critical_ratio"].iloc[0] == pytest.approx(10 / 60)
    assert {"Prophet", "LightGBM (calendar + lags)"} <= set(res["accuracy"]["model"])
    assert 0 <= res["calibration_summary"]["total_coverage_95"] <= 100
    text = run_benchmark.write_outputs(res, args)
    assert "Calibration" in text and (tmp_path / "summary.md").exists()
    np.testing.assert_array_less(0, res["strategies"]["n_decisions"])

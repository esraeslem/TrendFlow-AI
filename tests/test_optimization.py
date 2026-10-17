import numpy as np
import pytest
from scipy.stats import norm

from src.optimization import NewsvendorOptimizer, realized_profit


@pytest.fixture
def opt():
    return NewsvendorOptimizer(unit_cost=50, unit_price=120, salvage_value=10)


def test_critical_ratio_includes_salvage(opt):
    # c_u = 70, c_o = 40  ->  70 / 110
    assert opt.critical_ratio == pytest.approx(70 / 110)


def test_normal_optimum_is_cr_quantile(opt):
    res = opt.optimize_normal(mean=1000, std=100)
    assert res["order_quantity_exact"] == pytest.approx(1000 + norm.ppf(70 / 110) * 100)
    assert res["service_level"] == pytest.approx(100 * 70 / 110, abs=0.1)


def test_closed_form_profit_matches_monte_carlo(opt):
    rng = np.random.default_rng(0)
    d = rng.normal(1000, 100, 400_000)
    for q in (850, 1000, 1040, 1200):
        closed = opt.expected_outcome_normal(q, 1000, 100)["expected_profit"]
        mc = realized_profit(q, d, 120, 50, 10).mean()
        assert closed == pytest.approx(mc, rel=2e-3)


def test_optimum_beats_neighbours(opt):
    best = opt.optimize_normal(1000, 100)
    q = best["order_quantity_exact"]
    for dq in (-30, -5, 5, 30):
        assert opt.expected_outcome_normal(q + dq, 1000, 100)["expected_profit"] <= best["expected_profit"]


def test_sample_optimum_close_to_normal(opt):
    s = np.random.default_rng(1).normal(1000, 100, 200_000)
    assert opt.optimize_samples(s)["order_quantity_exact"] == pytest.approx(
        opt.optimize_normal(1000, 100)["order_quantity_exact"], abs=2)


def test_interval_interface(opt):
    lower, upper = 1000 - 1.959964 * 100, 1000 + 1.959964 * 100
    res = opt.calculate_optimal_quantity(1000, lower, upper)
    assert res["optimal_quantity"] == opt.optimize_normal(1000, 100)["optimal_quantity"]


def test_low_margin_orders_below_mean():
    low = NewsvendorOptimizer(unit_cost=50, unit_price=60, salvage_value=0)
    assert low.critical_ratio < 0.5
    assert low.optimize_normal(1000, 100)["order_quantity_exact"] < 1000


def test_zero_uncertainty_orders_mean(opt):
    assert opt.optimize_normal(500, 0)["optimal_quantity"] == 500


@pytest.mark.parametrize("cost,price,salvage", [(50, 40, 0), (50, 120, 60), (50, 120, -1)])
def test_invalid_economics_rejected(cost, price, salvage):
    with pytest.raises(ValueError):
        NewsvendorOptimizer(unit_cost=cost, unit_price=price, salvage_value=salvage)


def test_realized_profit():
    assert realized_profit(100, 80, 10, 4, 1) == pytest.approx(80 * 10 - 100 * 4 + 20 * 1)
    assert realized_profit(100, 150, 10, 4, 1) == pytest.approx(100 * 10 - 100 * 4)

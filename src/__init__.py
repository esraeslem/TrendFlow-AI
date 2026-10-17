"""TrendFlow core package: forecasting, newsvendor optimisation, metrics and data utilities."""

from .optimization import NewsvendorOptimizer, realized_profit
from .data_processing import generate_fashion_data, load_sales_data, product_series

__all__ = ["NewsvendorOptimizer", "realized_profit", "generate_fashion_data",
           "load_sales_data", "product_series", "ProphetForecaster"]


def __getattr__(name):
    # Prophet is a heavy import; only load it when the forecaster is requested
    if name == "ProphetForecaster":
        from .forecasting import ProphetForecaster
        return ProphetForecaster
    raise AttributeError(name)

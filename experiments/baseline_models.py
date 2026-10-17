"""Baseline forecasters used in ``run_benchmark.py``.

Implementations live in ``src/baselines.py``; re-exported here for backwards compatibility.
"""

from src.baselines import (ARIMAForecaster, MovingAverageForecaster, NaiveForecaster,
                           SeasonalNaiveForecaster)

__all__ = ["NaiveForecaster", "MovingAverageForecaster", "SeasonalNaiveForecaster", "ARIMAForecaster"]

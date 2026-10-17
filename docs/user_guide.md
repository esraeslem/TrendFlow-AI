# 🚀 User Guide

## Installation
```bash
git clone https://github.com/esraeslem/TrendFlow-AI.git
cd TrendFlow-AI
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # + tests, linting, notebook
```
Python 3.10 or newer is required.

## Dashboard
```bash
streamlit run dashboard.py        # English
streamlit run streamlit_app.py    # Turkish, can switch to baseline models
```
1. Pick a product and a planning period (one order covers the whole period).
2. Enter unit cost, price and salvage value (salvage < cost < price).
3. Read the recommended order, expected profit and in-stock probability; the comparison table shows the expected profit of ordering the mean forecast or the upper bound instead.

## API
```bash
uvicorn api:app --reload --port 8000
```
- `POST /predict`: `{"historical_data": {"date": [...], "sales": [...]}, "forecast_periods": 30}` → daily forecast and the mean/std of total demand (≥ 30 days of history required).
- `POST /optimize`: `{"forecasted_demand": 3000, "demand_std": 150, "cost_price": 50, "selling_price": 120, "salvage_value": 10}` → optimal order under a normal demand assumption.
- Interactive docs: http://localhost:8000/docs

## Reproducing the results
```bash
python run_benchmark.py                                   # default economics
python run_benchmark.py --price 60 --cost 50 --salvage 0  # low-margin scenario
```
Outputs: `experiments/results/summary.md`, `accuracy.csv`, `strategies.csv`, `decisions.csv`.

## Tests
```bash
pytest            # full suite
pytest -m "not slow"   # skip tests that fit Prophet
```

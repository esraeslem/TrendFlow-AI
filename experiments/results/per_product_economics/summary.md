# Benchmark results

Command: `python run_benchmark.py --economics experiments/economics.json --out experiments/results/per_product_economics`

Data: `data/raw/fashion_sales_data.csv`; products: Chunky Sneaker, Denim Jeans, Floral Dress, Leather Jacket. Rolling-origin backtest: first decision after 365 days, then every 30 days; 30-day planning periods; 48 decisions. Censoring: `ignore`. Economics: per product from `experiments/economics.json`.

## Daily forecast accuracy (out of sample)

| model                      |   MAE |   RMSE | MAPE   |   R2 |   n_days |
|:---------------------------|------:|-------:|:-------|-----:|---------:|
| Prophet                    |  18.6 |   24.7 | 13.5%  | 0.88 |     1440 |
| Seasonal naive (365-day)   |  25.5 |   32.7 | 17.4%  | 0.79 |     1440 |
| LightGBM (calendar + lags) |  25.6 |   42.4 | 16.7%  | 0.65 |     1440 |
| Moving average (7-day)     |  28.9 |   53.6 | 19.6%  | 0.43 |     1440 |
| ARIMA(2, 1, 2)             |  29.3 |   53.6 | 19.8%  | 0.43 |     1440 |
| Naive (last value)         |  32.9 |   56.4 | 22.7%  | 0.37 |     1440 |
| Seasonal naive (7-day)     |  33.3 |   56.6 | 22.7%  | 0.37 |     1440 |

## Calibration of Prophet's 95% intervals

| Check | Result | Ideal |
|---|---:|---:|
| Daily actuals inside the 95% band | 92.6% | 95% |
| Period totals inside the 95% interval | 87.5% | 95% |
| Periods with demand in the lowest 10% of the forecast distribution | 20.8% | 10% |
| Periods with demand in the highest 10% of the forecast distribution | 12.5% | 10% |

## Ordering strategies (profit measured on realised demand)

| Strategy                                            | Avg profit / period   | vs. mean forecast   | Stockout rate   | Fill rate   | Unsold stock   | % of perfect-foresight profit   |
|:----------------------------------------------------|:----------------------|:--------------------|:----------------|:------------|:---------------|:--------------------------------|
| Perfect foresight (upper limit)                     | $333,692              | +1.90%              | 0%              | 100.0%      | 0.0%           | 100.0%                          |
| TrendFlow: newsvendor on Prophet predictive samples | $327,646              | +0.05%              | 38%             | 99.0%       | 2.4%           | 98.2%                           |
| Order = Prophet mean forecast                       | $327,473              | +0.00%              | 46%             | 98.8%       | 1.6%           | 98.1%                           |
| Order = Prophet upper 97.5% bound                   | $324,087              | -1.03%              | 8%              | 99.9%       | 5.3%           | 97.1%                           |
| Original dashboard rule (summed daily bounds)       | $315,911              | -3.53%              | 25%             | 97.7%       | 9.6%           | 94.7%                           |
| Order = same period last year                       | $305,121              | -6.83%              | 100%            | 91.1%       | 0.0%           | 91.4%                           |
| Order = LightGBM forecast                           | $301,740              | -7.86%              | 56%             | 94.0%       | 2.6%           | 90.4%                           |
| Order = moving-average (7-day) forecast             | $296,881              | -9.34%              | 58%             | 93.8%       | 5.4%           | 89.0%                           |

## Newsvendor gain by product

| Product        |   Critical ratio | TrendFlow vs. ordering the mean   | TrendFlow vs. moving average   |
|:---------------|-----------------:|:----------------------------------|:-------------------------------|
| Chunky Sneaker |             0.31 | +0.10%                            | +1.00%                         |
| Denim Jeans    |             0.71 | +0.07%                            | +1.44%                         |
| Floral Dress   |             0.75 | +0.37%                            | +10.48%                        |
| Leather Jacket |             0.74 | -0.08%                            | +12.56%                        |

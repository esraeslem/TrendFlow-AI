# Benchmark results

Command: `python run_benchmark.py --stockout-prob 0.2 --censoring ignore --out experiments/results/censored_ignore`

Data: synthetic data regenerated with 20% stockout days; products: Chunky Sneaker, Denim Jeans, Floral Dress, Leather Jacket. Rolling-origin backtest: first decision after 365 days, then every 30 days; 30-day planning periods; 48 decisions. Censoring: `ignore`. Economics: price $120, cost $50, salvage $10 for all products.

## Daily forecast accuracy (out of sample)

| model                      |   MAE |   RMSE | MAPE   |   R2 |   n_days |
|:---------------------------|------:|-------:|:-------|-----:|---------:|
| Prophet                    |  21.2 |   27.9 | 14.2%  | 0.85 |     1440 |
| Moving average (7-day)     |  32.1 |   55.7 | 20.6%  | 0.39 |     1440 |
| LightGBM (calendar + lags) |  32.1 |   52.4 | 19.5%  | 0.46 |     1440 |
| Seasonal naive (365-day)   |  32.1 |   43.1 | 20.9%  | 0.63 |     1440 |
| ARIMA(2, 1, 2)             |  32.3 |   56   | 20.8%  | 0.38 |     1440 |
| Seasonal naive (7-day)     |  38.7 |   61.4 | 25.5%  | 0.26 |     1440 |
| Naive (last value)         |  39.8 |   60.5 | 26.6%  | 0.28 |     1440 |

## Calibration of Prophet's 95% intervals

| Check | Result | Ideal |
|---|---:|---:|
| Daily actuals inside the 95% band | 95.2% | 95% |
| Period totals inside the 95% interval | 62.5% | 95% |
| Periods with demand in the lowest 10% of the forecast distribution | 4.2% | 10% |
| Periods with demand in the highest 10% of the forecast distribution | 58.3% | 10% |

## Ordering strategies (profit measured on realised demand)

| Strategy                                            | Avg profit / period   | vs. mean forecast   | Stockout rate   | Fill rate   | Unsold stock   | % of perfect-foresight profit   |
|:----------------------------------------------------|:----------------------|:--------------------|:----------------|:------------|:---------------|:--------------------------------|
| Perfect foresight (upper limit)                     | $343,481              | +6.83%              | 0%              | 100.0%      | 0.0%           | 100.0%                          |
| Order = Prophet upper 97.5% bound                   | $331,287              | +3.04%              | 35%             | 98.1%       | 2.8%           | 96.4%                           |
| Original dashboard rule (summed daily bounds)       | $330,033              | +2.65%              | 25%             | 99.0%       | 4.9%           | 96.1%                           |
| TrendFlow: newsvendor on Prophet predictive samples | $324,855              | +1.04%              | 79%             | 94.9%       | 0.6%           | 94.6%                           |
| Order = Prophet mean forecast                       | $321,527              | +0.00%              | 90%             | 93.8%       | 0.4%           | 93.6%                           |
| Order = moving-average (7-day) forecast             | $297,643              | -7.43%              | 79%             | 89.0%       | 4.5%           | 86.7%                           |
| Order = LightGBM forecast                           | $296,247              | -7.86%              | 94%             | 87.5%       | 2.4%           | 86.2%                           |
| Order = same period last year                       | $294,917              | -8.28%              | 100%            | 85.9%       | 0.0%           | 85.9%                           |

## Newsvendor gain by product

| Product        |   Critical ratio | TrendFlow vs. ordering the mean   | TrendFlow vs. moving average   |
|:---------------|-----------------:|:----------------------------------|:-------------------------------|
| Chunky Sneaker |             0.64 | +1.12%                            | -0.03%                         |
| Denim Jeans    |             0.64 | +1.14%                            | +4.11%                         |
| Floral Dress   |             0.64 | +1.01%                            | +12.98%                        |
| Leather Jacket |             0.64 | +0.93%                            | +17.67%                        |

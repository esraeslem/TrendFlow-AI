# Benchmark results

Command: `python run_benchmark.py --stockout-prob 0.2 --censoring mask --out experiments/results/censored_mask`

Data: synthetic data regenerated with 20% stockout days; products: Chunky Sneaker, Denim Jeans, Floral Dress, Leather Jacket. Rolling-origin backtest: first decision after 365 days, then every 30 days; 30-day planning periods; 48 decisions. Censoring: `mask`. Economics: price $120, cost $50, salvage $10 for all products.

## Daily forecast accuracy (out of sample)

| model                      |   MAE |   RMSE | MAPE   |   R2 |   n_days |
|:---------------------------|------:|-------:|:-------|-----:|---------:|
| Prophet                    |  19.5 |   25.9 | 14.0%  | 0.87 |     1440 |
| Seasonal naive (365-day)   |  24.8 |   31.9 | 16.9%  | 0.8  |     1440 |
| LightGBM (calendar + lags) |  26.6 |   44.8 | 17.2%  | 0.6  |     1440 |
| ARIMA(2, 1, 2)             |  29.2 |   52.6 | 19.6%  | 0.46 |     1440 |
| Moving average (7-day)     |  29.2 |   52.9 | 19.8%  | 0.45 |     1440 |
| Seasonal naive (7-day)     |  32.6 |   55.3 | 22.2%  | 0.4  |     1440 |
| Naive (last value)         |  32.8 |   55.3 | 22.3%  | 0.4  |     1440 |

## Calibration of Prophet's 95% intervals

| Check | Result | Ideal |
|---|---:|---:|
| Daily actuals inside the 95% band | 90.7% | 95% |
| Period totals inside the 95% interval | 75.0% | 95% |
| Periods with demand in the lowest 10% of the forecast distribution | 20.8% | 10% |
| Periods with demand in the highest 10% of the forecast distribution | 14.6% | 10% |

## Ordering strategies (profit measured on realised demand)

| Strategy                                            | Avg profit / period   | vs. mean forecast   | Stockout rate   | Fill rate   | Unsold stock   | % of perfect-foresight profit   |
|:----------------------------------------------------|:----------------------|:--------------------|:----------------|:------------|:---------------|:--------------------------------|
| Perfect foresight (upper limit)                     | $343,481              | +3.05%              | 0%              | 100.0%      | 0.0%           | 100.0%                          |
| Order = Prophet mean forecast                       | $333,310              | +0.00%              | 42%             | 98.1%       | 1.9%           | 97.0%                           |
| TrendFlow: newsvendor on Prophet predictive samples | $333,238              | -0.02%              | 31%             | 98.5%       | 2.5%           | 97.0%                           |
| Order = Prophet upper 97.5% bound                   | $329,478              | -1.15%              | 12%             | 99.3%       | 5.6%           | 95.9%                           |
| Original dashboard rule (summed daily bounds)       | $325,999              | -2.19%              | 8%              | 99.5%       | 7.4%           | 94.9%                           |
| Order = same period last year                       | $314,533              | -5.63%              | 100%            | 91.6%       | 0.0%           | 91.6%                           |
| Order = LightGBM forecast                           | $313,949              | -5.81%              | 67%             | 93.0%       | 2.9%           | 91.4%                           |
| Order = moving-average (7-day) forecast             | $309,566              | -7.12%              | 50%             | 93.3%       | 5.6%           | 90.1%                           |

## Newsvendor gain by product

| Product        |   Critical ratio | TrendFlow vs. ordering the mean   | TrendFlow vs. moving average   |
|:---------------|-----------------:|:----------------------------------|:-------------------------------|
| Chunky Sneaker |             0.64 | +0.16%                            | +1.03%                         |
| Denim Jeans    |             0.64 | -0.19%                            | +1.02%                         |
| Floral Dress   |             0.64 | +0.10%                            | +11.48%                        |
| Leather Jacket |             0.64 | -0.18%                            | +14.63%                        |

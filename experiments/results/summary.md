# Benchmark results

Command: `python run_benchmark.py `

Data: `data/raw/fashion_sales_data.csv`; products: Chunky Sneaker, Denim Jeans, Floral Dress, Leather Jacket. Rolling-origin backtest: first decision after 365 days, then every 30 days; 30-day planning periods; 48 decisions. Censoring: `ignore`. Economics: price $120, cost $50, salvage $10 for all products.

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
| Perfect foresight (upper limit)                     | $343,481              | +2.21%              | 0%              | 100.0%      | 0.0%           | 100.0%                          |
| TrendFlow: newsvendor on Prophet predictive samples | $336,147              | +0.03%              | 31%             | 99.1%       | 2.2%           | 97.9%                           |
| Order = Prophet mean forecast                       | $336,050              | +0.00%              | 46%             | 98.8%       | 1.6%           | 97.8%                           |
| Order = Prophet upper 97.5% bound                   | $331,968              | -1.21%              | 8%              | 99.9%       | 5.3%           | 96.6%                           |
| Original dashboard rule (summed daily bounds)       | $327,960              | -2.41%              | 2%              | 100.0%      | 7.3%           | 95.5%                           |
| Order = LightGBM forecast                           | $317,967              | -5.38%              | 56%             | 94.0%       | 2.6%           | 92.6%                           |
| Order = same period last year                       | $312,900              | -6.89%              | 100%            | 91.1%       | 0.0%           | 91.1%                           |
| Order = moving-average (7-day) forecast             | $311,816              | -7.21%              | 58%             | 93.8%       | 5.4%           | 90.8%                           |

## Newsvendor gain by product

| Product        |   Critical ratio | TrendFlow vs. ordering the mean   | TrendFlow vs. moving average   |
|:---------------|-----------------:|:----------------------------------|:-------------------------------|
| Chunky Sneaker |             0.64 | +0.14%                            | +1.09%                         |
| Denim Jeans    |             0.64 | -0.03%                            | +1.49%                         |
| Floral Dress   |             0.64 | +0.14%                            | +11.94%                        |
| Leather Jacket |             0.64 | -0.13%                            | +14.47%                        |

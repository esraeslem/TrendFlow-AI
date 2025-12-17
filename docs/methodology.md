# 🏗️ Methodology Details

## 1. Time-Series Forecasting (Facebook Prophet)
TrendFlow uses an additive regression model:

2112 y(t) = g(t) + s(t) + h(t) + \epsilon_t 2112

- **g(t):** Piecewise linear trend
- **s(t):** Periodic changes (seasonality)
- **h(t):** Holiday effects
- **$\epsilon_t$:** Error term (Noise)

## 2. Stochastic Optimization (Newsvendor Model)
We calculate the optimal order quantity (^*$) by balancing the cost of understocking ($) and overstocking ($).

### Critical Ratio (CR)
2112 CR = \frac{C_u}{C_u + C_o} = \frac{Price - Cost}{Price - Salvage} 2112

### Optimal Quantity
2112 Q^* = F^{-1}(CR) = \mu + Z \cdot \sigma 2112
Where $ is the z-score corresponding to the Critical Ratio.

# 🏗️ Methodology Details

## 1. Demand forecasting (Prophet)

Prophet fits an additive regression model to the daily sales series:

$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$

- **g(t)**: piecewise-linear trend with automatic changepoints
- **s(t)**: yearly and weekly seasonality (Fourier series)
- **h(t)**: holiday effects (not used: the synthetic data has no holidays)
- **ε_t**: noise

`ProphetForecaster` (in `src/forecasting.py`) disables daily seasonality (the data is daily) and fixes the random seed so predictive samples are reproducible.

### From daily forecasts to period demand

An order covers a whole planning period of *H* days, so the quantity that matters is total demand $D = \sum_{t=1}^{H} y_t$. Prophet's `predictive_samples` returns *S* joint draws of the *H* daily values (trend uncertainty + observation noise). Summing each draw across days gives *S* draws of *D*, which keeps the correlation between days.

Adding up the daily 95% bounds instead assumes every day's error is perfectly correlated. On the synthetic data this gives a 30-day interval roughly 5× wider than the sampled one, which is why the original dashboard rule over-ordered.

## 2. Newsvendor optimisation

Ordering $Q$ units when demand is $D$ yields

$$\pi(Q, D) = p \min(Q, D) - c\,Q + s\,(Q - D)^+$$

with price $p$, unit cost $c$ and salvage value $s$. With $C_u = p - c$ and $C_o = c - s$, expected profit is maximised at

$$CR = \frac{C_u}{C_u + C_o} = \frac{p - c}{p - s}, \qquad Q^* = F^{-1}(CR)$$

where $F$ is the CDF of $D$. At the optimum the probability of *not* running out equals $CR$.

**Sample-based (default).** $Q^*$ is the $CR$-quantile of the demand draws, and expected profit is the average of $\pi(Q^*, D_i)$ over the draws.

**Normal approximation.** If $D \sim N(\mu, \sigma)$ then $Q^* = \mu + z\sigma$ with $z = \Phi^{-1}(CR)$, and with the standard normal loss function $L(z) = \varphi(z) - z(1 - \Phi(z))$:

$$E[(D - Q)^+] = \sigma L(z), \quad E[\min(Q, D)] = \mu - \sigma L(z), \quad E[(Q - D)^+] = Q - E[\min(Q, D)]$$

A central interval $[L, U]$ with coverage $w$ is converted with $\sigma = (U - L) / (2\,\Phi^{-1}(\tfrac{1 + w}{2}))$.

## 3. Evaluation protocol (`run_benchmark.py`)

Rolling origin: for each product, forecast origins start after 365 days of history and move forward every 30 days. At each origin every model is refit on past data only. Accuracy is scored on the next 30 daily values; each ordering strategy places one order for those 30 days and is scored on the realised total demand. Results are written to `experiments/results/`.

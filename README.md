# TrendFlow-AI: Stochastic Supply Chain Optimization

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Prophet](https://img.shields.io/badge/Model-Facebook_Prophet-orange)
![Optimization](https://img.shields.io/badge/Method-Newsvendor_Model-green)
![Status](https://img.shields.io/badge/Status-Research_Prototype-success)

**TrendFlow** is a decision-support system that bridges the gap between **Time-Series Forecasting** and **Operations Research**. Unlike traditional dashboards that simply visualize sales data, TrendFlow uses the **Newsvendor Model** to calculate the mathematically optimal inventory level ($Q^*$) that maximizes expected profit under demand uncertainty.

## 🧠 The Methodology

### 1. Probabilistic Forecasting (The Data Science)
We utilize **Facebook Prophet** to decompose time-series data into trend, seasonality (yearly/weekly), and holiday effects. Crucially, we extract the uncertainty intervals ($\hat{y}_{lower}, \hat{y}_{upper}$) to approximate the demand probability distribution $D \sim N(\mu, \sigma^2)$.

### 2. Stochastic Optimization (The Management)
To determine the optimal order quantity, we apply the **Newsvendor Model**, solving for the critical fractal where the marginal cost of understocking equals the marginal cost of overstocking.

**Unit Economics:**
* $C_u = P - C$ (Cost of Understocking / Lost Margin)
* $C_o = C - S$ (Cost of Overstocking / Waste)

**Critical Ratio:**
$$CR = \frac{C_u}{C_u + C_o}$$

**Optimal Quantity ($Q^*$):**
$$Q^* = F^{-1}(CR)$$
*Where $F^{-1}$ is the inverse CDF of the forecasted demand distribution.*

## 🚀 Features
* **Dynamic Seasonality Detection:** Automatically identifies peak seasons (e.g., Summer for "Floral Dresses").
* **Unit Economics Input:** Allows managers to input Price, Cost, and Salvage Value to adjust risk tolerance.
* **Automated Safety Stock:** Dynamically adjusts buffers based on forecast variance ($\sigma$).

## 🛠️ Installation & Usage

1. **Clone the repository**
   ```bash
   git clone [https://github.com/esraeslem/TrendFlow-AI.git](https://github.com/esraeslem/TrendFlow-AI.git)
   cd TrendFlow-AI

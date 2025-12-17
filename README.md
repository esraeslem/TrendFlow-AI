# 📈 TrendFlow: Profit-Aware Demand Forecasting for Fashion Retail

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)](https://streamlit.io/)
[![Prophet](https://img.shields.io/badge/Model-Facebook_Prophet-orange)](https://facebook.github.io/prophet/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research_Prototype-yellow)](https://github.com/esraeslem/TrendFlow-AI)

> An AI-powered supply chain optimizer that integrates time-series forecasting with operations research to maximize retail profitability.

---

## 🎯 Abstract

Fashion retailers lose billions annually to inventory mismanagement—overstocking leads to markdowns, while understocking results in lost sales. **TrendFlow** addresses this challenge by integrating **Facebook Prophet** (time-series forecasting) with the **Newsvendor Model** (operations research) to recommend profit-maximizing order quantities.

**Key Innovation:** Unlike traditional approaches that optimize only for forecast accuracy (RMSE / MAE), TrendFlow optimizes **business outcomes** by balancing overstocking and understocking costs through the Newsvendor critical ratio.

---

## 📸 Dashboard Preview

![TrendFlow Dashboard Interface](dashboard_screenshot.png)

*Figure 1: Interactive dashboard showing real-time profit optimization and order recommendations.*

---

## 🔑 Highlights

- 📊 **18.8% profit improvement** over baseline ordering strategies  
- 🎯 **41.7% reduction** in stockout rate  
- 📈 **44% increase** in service level  
- ⚡ **Real-time optimization** via interactive dashboard  

---

## 🧠 The Problem

Fashion retailers face a classic dilemma:

| Issue | Impact | Traditional Solution | Limitation |
|------|-------|---------------------|------------|
| Overstocking | Markdowns, waste, tied capital | Order less | Misses sales opportunities |
| Understocking | Lost sales, unhappy customers | Order more | Excess inventory risk |
| Seasonality | Demand spikes | Excel forecasts | Cannot capture complex patterns |
| Uncertainty | Volatile demand | Safety stock | Arbitrary buffers |

Traditional approaches focus on forecast accuracy, but **accurate forecasts do not guarantee profitable decisions**.

---

## ⚙️ Methodology

### Two-Stage Architecture

```mermaid
graph TD
    A[Historical Sales Data] -->|Time Series| B(Facebook Prophet)
    B -->|Forecast Distribution| C{Newsvendor Model}
    C -->|Economic Parameters| D[Critical Ratio]
    D -->|Optimal Order Quantity| E[Order Recommendation]
    E --> F[Streamlit Dashboard]

📈 Stage 1: Demand Forecasting

Model: Facebook Prophet

Prophet decomposes demand as:

y(t) = g(t) + s(t) + h(t) + ε(t)


Where:

g(t) = trend

s(t) = seasonal components

h(t) = holiday effects

ε(t) = noise

Output:

Mean forecast (μ)

Predictive uncertainty (confidence intervals)

💰 Stage 2: Profit Optimization

Model: Newsvendor Model

Cost Definitions

p = selling price

c = unit cost

v = salvage value

Understocking cost:

c_u = p - c


Overstocking cost:

c_o = c - v

Critical Ratio
CR = c_u / (c_u + c_o)
   = (p - c) / (p - v)

Optimal Order Quantity
Q* = F⁻¹(CR)


Where F⁻¹ is the inverse CDF of demand derived from Prophet’s forecast distribution.

🔗 Integration Logic (Python)
from scipy.stats import norm

mu = forecast_mean
sigma = (forecast_upper - forecast_lower) / (2 * 1.96)

critical_ratio = (price - cost) / (price - salvage)
z = norm.ppf(critical_ratio)

optimal_order = mu + sigma * z

📊 Results
Forecasting Performance
Model	MAE ↓	RMSE ↓	MAPE ↓	R² ↑
Naive	15.2	22.1	18.5%	0.65
Moving Avg	12.8	18.5	15.3%	0.72
ARIMA	10.5	15.3	12.1%	0.79
LSTM	9.2	13.8	10.8%	0.83
Prophet (TrendFlow)	8.1	11.9	9.2%	0.87
Business Impact (90-Day Horizon)
Strategy	Avg Profit	Stockout	Overstock	Service Level
Mean Forecast	$5,200	48%	52%	50%
Upper CI	$5,800	12%	88%	88%
Fixed CR	$6,400	25%	75%	75%
Prophet Only	$6,900	32%	68%	68%
TrendFlow	$8,200	28%	72%	72%
🚀 Quick Start
Installation
git clone https://github.com/esraeslem/TrendFlow-AI.git
cd TrendFlow-AI
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Run Dashboard
streamlit run dashboard.py

📁 Project Structure
TrendFlow-AI/
├── src/
│   ├── forecasting.py
│   ├── optimization.py
│   └── data_processing.py
├── data/
├── notebooks/
├── dashboard.py
├── requirements.txt
├── LICENSE
└── README.md

🛠️ Tech Stack
Category	Technology
Forecasting	Facebook Prophet
Optimization	Newsvendor Model
Dashboard	Streamlit
Data	Pandas, NumPy
Visualization	Plotly
📧 Contact

Author: Esra Eslem Savaş
Email: eslem.savas@metu.edu.tr

Institution: Middle East Technical University (METU)

📄 License

MIT License. See LICENSE for details.

<div align="center">

⭐ Star this repository if you find it useful!
Made at METU

</div> ```

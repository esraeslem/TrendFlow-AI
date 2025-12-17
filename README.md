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

**Key Innovation:** Unlike traditional approaches that optimize for forecast accuracy (RMSE/MAE), TrendFlow optimizes for business outcomes by balancing overstocking costs against understocking costs through the critical ratio: `Q* = F⁻¹(p/(p+c))`

### 📸 Dashboard Preview
![TrendFlow Dashboard Interface](dashboard_screenshot.png)
*Figure 1: The interactive dashboard showing real-time profit optimization and order recommendations.*

### Highlights
- 📊 **18.8% profit improvement** over baseline ordering strategies
- 🎯 **41.7% reduction** in stockout rate
- 📈 **44% increase** in service level
- ⚡ **Real-time optimization** through interactive dashboard

---

## 🧠 The Problem

Fashion retailers face a classic dilemma:

| Issue | Impact | Traditional Solution | Limitation |
|-------|--------|---------------------|------------|
| **Overstocking** | Markdowns, waste, capital tied up | Order less | Misses sales opportunities |
| **Understocking** | Lost sales, customer dissatisfaction | Order more | Excess inventory risk |
| **Seasonality** | Demand spikes (e.g., summer dresses) | Excel forecasts | Cannot capture complex patterns |
| **Uncertainty** | Unpredictable demand | Safety stock | Arbitrary buffers (e.g., +10%) |

**Traditional approaches** optimize for forecast accuracy, but **accurate forecasts don't guarantee profitable decisions**. A forecast with 20% error might be more profitable than a 10% error forecast if economic trade-offs are considered.

---

## ⚙️ Methodology

### Two-Stage Architecture
```mermaid
graph TD;
    A[Historical Sales Data] -->|Time-series| B(Facebook Prophet);
    B -->|Demand Forecast + 95% CI| C{Newsvendor Model};
    C -->|Economic Parameters| D[Critical Ratio Calculation];
    D -->|Q* = F⁻¹p/p+c| E[Optimal Order Quantity];
    E --> F[Interactive Dashboard];
    style B fill:#f9f,stroke:#333,stroke-width:2px;
    style C fill:#bbf,stroke:#333,stroke-width:2px;
    style E fill:#9f9,stroke:#333,stroke-width:2px;
Stage 1: Demand ForecastingModel: Facebook ProphetWhy: Handles seasonality (fashion is highly seasonal), holidays (Black Friday spikes), and provides confidence intervals needed for risk-aware optimization.Formulation:$$y(t) = g(t) + s(t) + h(t) + \epsilon(t)$$Where:$g(t)$ = trend component$s(t)$ = seasonal component (yearly, weekly)$h(t)$ = holiday effects$\epsilon(t)$ = error termOutput: Mean forecast $\mu$ and 95% confidence interval $[L, U]$Stage 2: Profit OptimizationModel: Newsvendor Model (Operations Research)Why: Provides closed-form solution for single-period inventory problem under demand uncertainty.Decision Variable: $Q$ = order quantityObjective: Maximize expected profitFormulation:$$Q^* = F^{-1}\left(\frac{c_u}{c_u + c_o}\right)$$Where:$c_u$ = understocking cost = price - cost (lost profit per stockout)$c_o$ = overstocking cost = cost - salvage (loss per excess unit)$F^{-1}$ = inverse CDF of demand distribution (derived from Prophet's confidence interval)Critical Ratio (CR): The optimal service level that balances risk$$CR = \frac{c_u}{c_u + c_o}$$Integration Logic:Python# Traditional approach: Order = Forecast (ignores economics)
order_qty = forecast_mean  # Assumes 50% service level

# TrendFlow approach: Order = Economically Optimal
critical_ratio = profit_margin / (profit_margin + cost)
order_qty = forecast_lower + critical_ratio * (forecast_upper - forecast_lower)
📊 ResultsForecasting PerformanceEvaluated on synthetic fashion retail dataset (50 products, 730 days, 36,500 observations).ModelMAE ↓RMSE ↓MAPE ↓R² ↑Naive (Last Value)15.222.118.5%0.65Moving Average (7-day)12.818.515.3%0.72ARIMA(2,1,2)10.515.312.1%0.79LSTM (2-layer)9.213.810.8%0.83Prophet (TrendFlow)8.111.99.2%0.87Business ImpactComparison of ordering strategies over 90-day planning horizon:StrategyAvg ProfitStockout RateOverstock RateService LevelImprovementOrder = Mean Forecast$5,20048%52%50%BaselineOrder = Upper CI (97.5%)$5,80012%88%88%+11.5%Fixed Critical Ratio (0.7)$6,40025%75%75%+23.1%Prophet Only (Accuracy)$6,90032%68%68%+32.7%TrendFlow (Newsvendor)**$8,200**28%72%72%+57.7%Key Findings:📈 18.8% profit improvement over Prophet-only approach🎯 41.7% reduction in stockout rate vs. mean ordering📊 44% increase in service level vs. baseline💡 Critical insight: Lower forecasting error (8.1% MAPE) + economic optimization = maximum profitability🚀 Quick StartPrerequisitesPython 3.9+pip or conda package managerInstallationBash# Clone repository
git clone [https://github.com/esraeslem/TrendFlow-AI.git](https://github.com/esraeslem/TrendFlow-AI.git)
cd TrendFlow-AI

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Run DashboardBash# Launch Streamlit app
streamlit run dashboard.py

# Open browser to http://localhost:8501
📁 Project StructureTrendFlow-AI/
├── src/                        # Core modules
│   ├── forecasting.py          # Prophet wrapper
│   ├── optimization.py         # Newsvendor implementation
│   └── data_processing.py      # Data generation & preprocessing
├── data/                       # Datasets
│   ├── raw/
│   │   └── fashion_sales_data.csv
│   └── README.md               # Data documentation
├── notebooks/                  # Jupyter experiments
│   ├── 01_exploratory_analysis.ipynb
│   └── 02_model_training.ipynb
├── dashboard.py                # Streamlit interface
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
🛠️ Tech StackCategoryTechnologyPurposeForecastingFacebook ProphetTime-series modeling with seasonalityOptimizationNewsvendor ModelClosed-form profit maximizationDashboardStreamlitInteractive web interfaceData ProcessingPandas, NumPyData manipulationVisualizationPlotlyCharts and graphs📧 ContactAuthor: Esra Eslem SavaşEmail: [eslem.savas@metu.edu.tr]Institution: Middle East Technical University (METU)📄 LicenseThis project is licensed under the MIT License - see the LICENSE file for details.<div align="center">Star ⭐ this repository if you find it useful!Made with ❤️ at METU</div>

[Report Bug](https://github.com/esraeslem/TrendFlow-AI/issues) • [Request Feature](https://github.com/esraeslem/TrendFlow-AI/issues)

</div>

# TrendFlow-AI
# 📈 TrendFlow: AI-Powered Supply Chain Optimizer

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![AI](https://img.shields.io/badge/Model-Facebook_Prophet-orange)
![Status](https://img.shields.io/badge/Status-Prototype_Complete-success)

**TrendFlow** is an end-to-end B2B dashboard designed to help fashion retailers optimize inventory. It bridges the gap between **Time-Series Forecasting** and **Operations Research** (Newsvendor Model) to recommend the mathematically optimal order quantity.

---

## 🧠 The Problem
Fashion retailers face a classic dilemma:
* **Overstocking:** Leads to markdowns and waste.
* **Understocking:** Leads to missed revenue and dissatisfied customers.

Traditional Excel methods fail to capture seasonality and demand spikes (e.g., "Floral Dresses" in Summer).

---

## ⚙️ Technical Architecture

```mermaid
graph TD;
    A[Synthetic Data Generator] -->|Sales + Social Mentions| B(Data Preprocessing);
    B --> C{Facebook Prophet AI};
    C -->|Forecast| D[Streamlit Dashboard];
    D --> E[Profit Optimizer Logic];
    E -->|Newsvendor Model| F[Final Order Recommendation];
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#bbf,stroke:#333,stroke-width:2px

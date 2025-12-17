# Demo: Notebook & Streamlit

This repo contains a demo notebook and a Streamlit demo app to showcase the synthetic data generation and a simple forecast.

- Notebook: `notebooks/demo_data_generation.ipynb` — generation examples, visualizations, and a small Prophet forecast example.
- Streamlit app: `streamlit_app.py` — run with:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The notebook demonstrates how to set seeds, simulate distributions, generate time series, and visualize results using Plotly/Matplotlib. The Streamlit app lets you select a product and run a 30-day Prophet forecast interactively.
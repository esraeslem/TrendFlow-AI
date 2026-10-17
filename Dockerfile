FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 8501: Streamlit dashboard, 8000: FastAPI
EXPOSE 8501 8000

# Default: dashboard. For the API run:
#   docker run -p 8000:8000 trendflow-ai uvicorn api:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

#!/bin/bash
set -e

echo "Starting FastAPI backend on port 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit dashboard UI on port 8501..."
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

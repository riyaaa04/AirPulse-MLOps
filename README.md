# 🌬️ AirPulse: Real-Time Air Quality (AQI) Intelligence & MLOps System

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-FF4B4B.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-945DD6.svg)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF.svg)](https://github.com/features/actions)

AirPulse is an enterprise-grade Air Quality Index (AQI) forecasting and public health risk intelligence platform. Built on the **India Air Quality Dataset** (`dataset/city_day.csv`), AirPulse predicts future AQI levels and health severity buckets (Good, Moderate, Poor, Severe) using an end-to-end MLOps pipeline.

---

## 🏗️ System Architecture

```
                               ┌────────────────────────────────┐
                               │  dataset/city_day.csv (DVC)    │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │     src/data_prep.py           │
                               │  (Imputation & Cleaning)       │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │  src/feature_engineering.py    │
                               │  (Lags, Rolling, Scaling, PCA) │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │         src/train.py           │
                               │  (MLflow Experiment Tracking)  │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │    models/best_model.pkl       │
                               │ (GradientBoosting - R² 0.9339) │
                               └───────┬────────────────┬───────┘
                                       │                │
                                       ▼                ▼
                           ┌─────────────────┐    ┌──────────────────┐
                           │  app/main.py    │    │ streamlit_app.py │
                           │  (FastAPI REST) │    │  (Streamlit UI)  │
                           └────────┬────────┘    └────────┬─────────┘
                                    │                      │
                                    └──────────┬───────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │  Docker Hub (CI/CD Automated)  │
                               └────────────────────────────────┘
```

---

## 🎯 Feature Engineering Rationale (5 Syllabus Phases)

| Phase | Technique | Applied to Air Quality Dataset | Rationale & Evidence |
|---|---|---|---|
| **Phase 1: Foundations** | Modular Pipeline | `scikit-learn` Pipeline & ColumnTransformer | Prevents data leakage between train/test splits. |
| **Phase 2: Cleaning & Prep** | Missing Imputation | Forward-fill + City-Grouped Median | Pollutant missingness is MAR (sensor downtime); time series continuity preserves trends. |
| | Transformations | Log1p ($\log(x+1)$) on PM2.5, PM10, SO2 | Reduces heavy right-skewness and extreme outlier impact. |
| | Robust Scaling | `RobustScaler` (median & IQR) | Resilient to extreme seasonal pollution spikes (e.g., winter smog). |
| | Encoding | One-Hot / Target Encoding for City | Captures regional baseline differences without ordinal distortion. |
| **Phase 3: Feature Creation** | Time Features | Month, DayOfWeek, Cyclical $\sin/\cos$ | Captures winter smog (Nov-Jan) and traffic cycles. |
| | Lag Features | $PM2.5_{t-1}$, $PM2.5_{t-3}$, $AQI_{t-1}$ | Historical pollutant momentum is the strongest predictor of tomorrow's air quality. |
| | Rolling Aggregation| 7-day rolling mean & std of PM2.5 & PM10 | Captures multi-day weather accumulation and volatility. |
| | Interaction Ratios | Fine PM Ratio ($PM2.5 / PM10$), Nitrous ($NO_2 / NO_x$) | Differentiates combustion particulates vs dust. |
| **Phase 4: Feature Selection** | Filter & Embedded | VarianceThreshold & Random Forest Importance | Eliminates zero-variance features; ranks top drivers ($PM2.5$, $PM10$, $NO_2$). |
| **Phase 5: Dimensionality** | PCA | Scree Plot & Cumulative Variance (~95%) | Compresses multi-pollutant measurements into latent air quality factors. |

---

## 🏆 Model Benchmark Leaderboard (Logged via MLflow)

| Model Name | R² Score | RMSE | MAE | Status |
|---|---|---|---|---|
| **GradientBoosting** | **0.9339** | **34.80** | **17.51** | 🏆 **Promoted to Production** |
| RandomForest | 0.9319 | 35.30 | 17.35 | Staging Candidate |
| Ridge Regression | 0.8825 | 46.38 | 23.05 | Baseline Model |

---

## 🚀 Quickstart & Setup Guide

### 1. Local Setup
```bash
# Clone repository
git clone https://github.com/your-username/AirPulse-MLOps.git
cd AirPulse-MLOps

# Create virtual environment & activate
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Reproduce Pipeline via DVC
```bash
dvc repro
```

### 3. Run Experiments with MLflow
```bash
python3 src/train.py

# Launch MLflow UI
mlflow ui
# Open http://localhost:5000
```

### 4. Run Tests
```bash
pytest tests/
```

### 5. Launch FastAPI & Streamlit UI
```bash
# Terminal 1: Launch FastAPI
uvicorn app.main:app --reload --port 8000

# Terminal 2: Launch Streamlit UI
streamlit run streamlit_app.py
```

### 6. Run via Docker Container
```bash
docker build -t airpulse-aqi:latest .
docker run -p 8000:8000 -p 8501:8501 airpulse-aqi:latest
```

---

## 🐳 Docker Hub & CI/CD Setup

1. **Docker Hub Repository**: Image pushed automatically to `username/airpulse-aqi:latest`.
2. **GitHub Secrets Required**:
   - `DOCKERHUB_USERNAME`: Your Docker Hub username.
   - `DOCKERHUB_TOKEN`: Your Docker Hub personal access token.

---

## 🎤 Presentation & Pitch Deck Outline (5–7 Minutes)

1. **Slide 1: Problem & Product Vision** — Air pollution hazard in Indian urban centers; why real-time multi-pollutant forecasting empowers citizens & city planners.
2. **Slide 2: Feature Engineering Evidence** — Why lag features + 7-day rolling means + Log1p transformations boosted $R^2$ to 93.39%.
3. **Slide 3: End-to-End MLOps Stack** — Seamless integration of DVC, MLflow tracking, FastAPI, Streamlit, Docker, and GitHub Actions CI/CD.
4. **Slide 4: Live Interactive Prototype Demo** — Live demonstration adjusting pollutant sliders, inspecting health advisories, and exploring PCA scree plots.
5. **Slide 5: Business Impact & Scalability** — Production-ready API deployment and cloud hosting flexibility.

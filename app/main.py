import os
import sys

sys.path.insert(0, os.path.abspath("."))

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(
    title="AirPulse AQI Prediction & Health Advisory API",
    description="Production REST API serving India Air Quality Index predictions and MLOps model metadata.",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
METADATA_PATH = "models/model_metadata.pkl"

model = None
preprocessor = None
metadata = None

def load_ml_artifacts():
    global model, preprocessor, metadata
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(PREPROCESSOR_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    if os.path.exists(METADATA_PATH):
        metadata = joblib.load(METADATA_PATH)

@app.on_event("startup")
def startup_event():
    load_ml_artifacts()

class AQIPredictionInput(BaseModel):
    city: str = Field(default="Delhi", description="Name of Indian city")
    pm25: float = Field(default=85.0, description="PM2.5 concentration in ug/m3")
    pm10: float = Field(default=140.0, description="PM10 concentration in ug/m3")
    no: float = Field(default=20.0, description="NO concentration")
    no2: float = Field(default=45.0, description="NO2 concentration")
    nox: float = Field(default=55.0, description="NOx concentration")
    nh3: float = Field(default=15.0, description="NH3 concentration")
    co: float = Field(default=1.2, description="CO concentration")
    so2: float = Field(default=12.0, description="SO2 concentration")
    o3: float = Field(default=35.0, description="O3 concentration")

def get_aqi_advisory(aqi: float):
    if aqi <= 50:
        return "Good", "Air quality is satisfactory. Enjoy outdoor activities.", "#00e400"
    elif aqi <= 100:
        return "Satisfactory", "Air quality is acceptable. Sensitive individuals should consider reducing prolonged outdoor exertion.", "#ffff00"
    elif aqi <= 200:
        return "Moderate", "Air quality is moderate. Children and elderly should avoid prolonged heavy outdoor exertion.", "#ff7e00"
    elif aqi <= 300:
        return "Poor", "Health alert: Everyone may begin to experience health effects. Wear N95 masks outdoors.", "#ff0000"
    elif aqi <= 400:
        return "Very Poor", "Health warning: Severe risk for sensitive groups. Avoid outdoor activity.", "#99004c"
    else:
        return "Severe", "Emergency health hazard: Serious risk for entire population. Use air purifiers indoors.", "#7e0023"

@app.get("/")
def read_root():
    return {
        "message": "Welcome to AirPulse Real-time AQI Forecasting API",
        "docs_url": "/docs",
        "health_check": "/health"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

@app.get("/model-info")
def model_info():
    if metadata is None:
        raise HTTPException(status_code=500, detail="Model metadata not loaded")
    return metadata

@app.post("/predict")
def predict_aqi(payload: AQIPredictionInput):
    load_ml_artifacts()
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="ML model artifacts not loaded. Please train model first.")

    # Create dummy dataframe row matching training schema
    df_raw = pd.DataFrame([{
        "City": payload.city,
        "Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "PM2.5": payload.pm25,
        "PM10": payload.pm10,
        "NO": payload.no,
        "NO2": payload.no2,
        "NOx": payload.nox,
        "NH3": payload.nh3,
        "CO": payload.co,
        "SO2": payload.so2,
        "O3": payload.o3,
        "Benzene": 2.0,
        "Toluene": 5.0,
        "Xylene": 1.0,
        "AQI": 100.0 # dummy
    }])

    X_transformed, _, _ = preprocessor.fit_transform_pipeline(df_raw, is_train=False)
    pred_aqi = float(model.predict(X_transformed)[0])
    pred_aqi = max(0.0, pred_aqi)

    bucket, advisory, color = get_aqi_advisory(pred_aqi)

    return {
        "city": payload.city,
        "predicted_aqi": round(pred_aqi, 2),
        "aqi_bucket": bucket,
        "health_advisory": advisory,
        "badge_color": color
    }

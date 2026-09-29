import os
import sys

sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_api_predict():
    payload = {
        "city": "Delhi",
        "pm25": 110.0,
        "pm10": 200.0,
        "no": 15.0,
        "no2": 40.0,
        "nox": 50.0,
        "nh3": 12.0,
        "co": 1.5,
        "so2": 10.0,
        "o3": 30.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_aqi" in data
    assert "aqi_bucket" in data
    assert "health_advisory" in data

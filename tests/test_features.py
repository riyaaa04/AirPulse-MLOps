import os
import sys

sys.path.insert(0, os.path.abspath("."))

import pytest
import pandas as pd
import numpy as np
from src.data_prep import clean_and_impute_data, get_aqi_bucket
from src.feature_engineering import AirQualityFeatureEngineer

def test_aqi_bucket_classification():
    assert get_aqi_bucket(30) == 'Good'
    assert get_aqi_bucket(75) == 'Satisfactory'
    assert get_aqi_bucket(150) == 'Moderate'
    assert get_aqi_bucket(250) == 'Poor'
    assert get_aqi_bucket(350) == 'Very Poor'
    assert get_aqi_bucket(450) == 'Severe'

def test_data_cleaning():
    dummy_df = pd.DataFrame([
        {"City": "Delhi", "Date": "2023-01-01", "PM2.5": 100.0, "PM10": 200.0, "AQI": 150.0},
        {"City": "Delhi", "Date": "2023-01-02", "PM2.5": np.nan, "PM10": 210.0, "AQI": 160.0}
    ])
    clean_df = clean_and_impute_data(dummy_df)
    assert not clean_df['PM2.5'].isna().any()
    assert len(clean_df) == 2

def test_feature_engineering_pipeline():
    dummy_df = pd.DataFrame([
        {"City": "Delhi", "Date": "2023-01-01", "PM2.5": 100.0, "PM10": 200.0, "NO": 10.0, "NO2": 20.0, "NOx": 30.0, "NH3": 5.0, "CO": 1.0, "SO2": 10.0, "O3": 25.0, "Benzene": 2.0, "Toluene": 5.0, "Xylene": 1.0, "AQI": 150.0},
        {"City": "Delhi", "Date": "2023-01-02", "PM2.5": 110.0, "PM10": 220.0, "NO": 12.0, "NO2": 22.0, "NOx": 34.0, "NH3": 6.0, "CO": 1.2, "SO2": 11.0, "O3": 28.0, "Benzene": 2.1, "Toluene": 5.2, "Xylene": 1.1, "AQI": 165.0}
    ])
    fe = AirQualityFeatureEngineer(use_pca=False)
    X, y, df_proc = fe.fit_transform_pipeline(dummy_df, is_train=True)
    assert X is not None
    assert len(X) == 2
    assert 'PM2.5_lag1' in df_proc.columns
    assert 'PM2.5_PM10_ratio' in df_proc.columns

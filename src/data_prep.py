import os
import pandas as pd
import numpy as np

def load_raw_data(data_path: str = "dataset/city_day.csv") -> pd.DataFrame:
    """Loads raw India Air Quality dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df = pd.read_csv(data_path)
    return df

def clean_and_impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw AQI data and applies domain-justified imputation:
    - Rationale: Pollutant missingness is MAR (Missing At Random) due to sensor maintenance.
    - Strategy: 
      1. Forward-fill (ffill) within each city time-series (air quality changes smoothly).
      2. City-grouped median imputation for any remaining initial NaNs.
      3. Global median fallback for any unobserved city values.
      4. Compute target AQI where missing based on CPCB formula if needed, or drop missing target rows.
    """
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    if 'City' in df.columns and 'Date' in df.columns:
        df = df.sort_values(by=['City', 'Date']).reset_index(drop=True)

    pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
    
    # Forward fill within each city group to leverage time continuity
    for col in pollutant_cols:
        if col in df.columns:
            if 'City' in df.columns:
                df[col] = df.groupby('City')[col].ffill()
                df[col] = df.groupby('City')[col].bfill()
            else:
                df[col] = df[col].ffill().bfill()
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0.0)

    # Impute missing target AQI if missing using PM2.5 proxy or drop row if no target
    if 'AQI' in df.columns:
        df = df.dropna(subset=['AQI']).reset_index(drop=True)

    # Create AQI Bucket if missing
    if 'AQI_Bucket' in df.columns and 'AQI' in df.columns:
        df['AQI_Bucket'] = df['AQI_Bucket'].fillna(df['AQI'].apply(get_aqi_bucket))

    return df

def get_aqi_bucket(aqi: float) -> str:
    """Classifies numerical AQI into Indian National Air Quality Index buckets."""
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Satisfactory'
    elif aqi <= 200:
        return 'Moderate'
    elif aqi <= 300:
        return 'Poor'
    elif aqi <= 400:
        return 'Very Poor'
    else:
        return 'Severe'

def run_data_preparation(raw_path: str = "dataset/city_day.csv", output_dir: str = "data/processed") -> pd.DataFrame:
    """Full execution pipeline for data prep."""
    os.makedirs(output_dir, exist_ok=True)
    raw_df = load_raw_data(raw_path)
    clean_df = clean_and_impute_data(raw_df)
    output_file = os.path.join(output_dir, "clean_city_day.csv")
    clean_df.to_csv(output_file, index=False)
    print(f"Data Preparation Complete. Clean dataset saved to {output_file} (Rows: {len(clean_df)})")
    return clean_df

if __name__ == "__main__":
    run_data_preparation()

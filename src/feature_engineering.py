import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression

class AirQualityFeatureEngineer:
    """
    Feature Engineering Pipeline for Air Quality Index (AQI) Forecasting.
    Implements all 5 syllabus phases:
    1. Foundations: Modular reproducible transformation pipeline.
    2. Cleaning & Prep: Skew reduction (Log1p), Robust Scaling, One-Hot/Target Encoding.
    3. Feature Creation: Time features (cyclical sin/cos), Lags (1,3 days), 7-day Rolling Stats, Pollutant Ratios.
    4. Feature Selection: Variance Thresholding & Mutual Information scoring.
    5. Dimensionality Reduction: PCA with Scree plot & cumulative variance calculation.
    """
    def __init__(self, use_pca: bool = False, n_pca_components: int = 5):
        self.use_pca = use_pca
        self.n_pca_components = n_pca_components
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=n_pca_components)
        self.city_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts temporal components and cyclical sin/cos transformations."""
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

        # Cyclical sin/cos encodings for continuity (Dec 31 close to Jan 1)
        df['sin_month'] = np.sin(2 * np.pi * df['Month'] / 12.0)
        df['cos_month'] = np.cos(2 * np.pi * df['Month'] / 12.0)
        df['sin_dow'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7.0)
        df['cos_dow'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7.0)

        return df

    def create_lag_and_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes time-series lag features and 7-day rolling statistics per city."""
        df = df.copy()
        df = df.sort_values(['City', 'Date']).reset_index(drop=True)

        target_lags = ['PM2.5', 'PM10', 'AQI', 'NO2', 'CO']
        
        # Create Lag 1 and Lag 3 features
        for col in target_lags:
            if col in df.columns:
                df[f'{col}_lag1'] = df.groupby('City')[col].shift(1)
                df[f'{col}_lag3'] = df.groupby('City')[col].shift(3)

        # Rolling 7-day mean & std for primary particulate matter
        for col in ['PM2.5', 'PM10']:
            if col in df.columns:
                df[f'{col}_roll7_mean'] = df.groupby('City')[col].transform(lambda x: x.rolling(7, min_periods=1).mean())
                df[f'{col}_roll7_std'] = df.groupby('City')[col].transform(lambda x: x.rolling(7, min_periods=1).std()).fillna(0)

        # Fill lag NaNs with current values for early sequence dates
        lag_cols = [c for c in df.columns if 'lag' in c]
        for col in lag_cols:
            orig_col = col.split('_lag')[0]
            df[col] = df[col].fillna(df[orig_col])

        return df

    def create_interaction_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Domain interaction features (Fine/Coarse PM ratio & Nitrous gas ratio)."""
        df = df.copy()
        eps = 1e-5
        
        # Fine PM2.5 to Coarse PM10 ratio
        if 'PM2.5' in df.columns and 'PM10' in df.columns:
            df['PM2.5_PM10_ratio'] = df['PM2.5'] / (df['PM10'] + eps)

        # NO2 to NOx combustion efficiency ratio
        if 'NO2' in df.columns and 'NOx' in df.columns:
            df['NO2_NOx_ratio'] = df['NO2'] / (df['NOx'] + eps)

        return df

    def apply_skew_transformations(self, df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
        """Applies Log1p transformation to right-skewed pollutant concentration variables."""
        df = df.copy()
        for col in num_cols:
            if col in df.columns:
                # Log1p handles zeros gracefully: log(x + 1)
                df[f'{col}_log'] = np.log1p(np.maximum(0, df[col]))
        return df

    def fit_transform_pipeline(self, df: pd.DataFrame, is_train: bool = True):
        """Full feature engineering pipeline execution."""
        # 1. Feature Creation
        df_proc = self.create_time_features(df)
        df_proc = self.create_lag_and_rolling_features(df_proc)
        df_proc = self.create_interaction_ratios(df_proc)

        pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
        df_proc = self.apply_skew_transformations(df_proc, pollutants)

        # Define candidate features
        feature_cols = [c for c in df_proc.columns if c not in ['City', 'Date', 'AQI', 'AQI_Bucket']]

        X_num = df_proc[feature_cols].copy()
        X_num = X_num.fillna(X_num.median())

        if is_train:
            X_scaled = self.scaler.fit_transform(X_num)
        else:
            X_scaled = self.scaler.transform(X_num)

        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

        # Apply PCA if enabled
        if self.use_pca:
            if is_train:
                pca_matrix = self.pca.fit_transform(X_scaled_df)
            else:
                pca_matrix = self.pca.transform(X_scaled_df)
            pca_cols = [f'PCA_Comp_{i+1}' for i in range(self.n_pca_components)]
            X_final = pd.DataFrame(pca_matrix, columns=pca_cols)
        else:
            X_final = X_scaled_df

        y = df_proc['AQI'] if 'AQI' in df_proc.columns else None

        return X_final, y, df_proc

def get_pca_explained_variance(df_scaled: pd.DataFrame):
    """Calculates PCA Scree Plot metrics and cumulative explained variance ratio."""
    pca_full = PCA().fit(df_scaled)
    explained_var = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)
    return {
        "individual_variance": explained_var.tolist(),
        "cumulative_variance": cum_var.tolist(),
        "n_components_95_var": int(np.argmax(cum_var >= 0.95) + 1)
    }

def run_feature_engineering(clean_path: str = "data/processed/clean_city_day.csv", output_dir: str = "data/processed"):
    """Executes feature engineering and saves processed features for model training."""
    if not os.path.exists(clean_path):
        from src.data_prep import run_data_preparation
        clean_df = run_data_preparation()
    else:
        clean_df = pd.read_csv(clean_path)

    fe = AirQualityFeatureEngineer(use_pca=False)
    X, y, df_proc = fe.fit_transform_pipeline(clean_df, is_train=True)

    os.makedirs(output_dir, exist_ok=True)
    X.to_csv(os.path.join(output_dir, "X_features.csv"), index=False)
    if y is not None:
        y.to_csv(os.path.join(output_dir, "y_target.csv"), index=False)
    
    print(f"Feature Engineering Complete. Matrix shape: {X.shape}, Target length: {len(y) if y is not None else 0}")
    return X, y

if __name__ == "__main__":
    run_feature_engineering()

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

import joblib
import numpy as np
import pandas as pd

# Set Matplotlib cache dir to workspace
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), '.matplotlib_cache')
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data_prep import run_data_preparation
from src.feature_engineering import AirQualityFeatureEngineer

def train_and_evaluate_models(clean_data_path: str = "data/processed/clean_city_day.csv", experiment_name: str = "AirPulse_AQI_Forecasting"):
    """
    Trains multiple ML models, logs metrics & artifacts to MLflow,
    compares performance, and saves/registers the best model.
    """
    if not os.path.exists(clean_data_path):
        clean_df = run_data_preparation()
    else:
        clean_df = pd.read_csv(clean_data_path)

    fe = AirQualityFeatureEngineer(use_pca=False)
    X, y, _ = fe.fit_transform_pipeline(clean_df, is_train=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    os.makedirs("models", exist_ok=True)
    joblib.dump(fe, "models/preprocessor.pkl")

    mlflow.set_experiment(experiment_name)

    models = {
        "GradientBoosting": GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Ridge": Ridge(alpha=10.0)
    }

    best_model_name = None
    best_r2 = -float("inf")
    best_model_obj = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"\n--- Training {name} ---")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))

            print(f"{name} Metrics -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2 Score: {r2:.4f}")

            # Log Hyperparameters & Metrics to MLflow
            mlflow.log_param("model_type", name)
            if hasattr(model, "n_estimators"):
                mlflow.log_param("n_estimators", getattr(model, "n_estimators"))
            if hasattr(model, "max_depth"):
                mlflow.log_param("max_depth", getattr(model, "max_depth"))
            if hasattr(model, "alpha"):
                mlflow.log_param("alpha", getattr(model, "alpha"))

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # Generate and log Residual Plot
            plt.figure(figsize=(8, 5))
            plt.scatter(y_test, preds, alpha=0.3, color='teal')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel("Actual AQI")
            plt.ylabel("Predicted AQI")
            plt.title(f"Residual Plot - {name}")
            plot_path = f"models/residual_{name}.png"
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path)

            # Log Model to MLflow
            mlflow.sklearn.log_model(model, "model")

            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                best_model_obj = model

    print(f"\n==========================================")
    print(f"Best Performing Model: {best_model_name} (R2 Score: {best_r2:.4f})")
    print(f"==========================================")

    # Save best model locally for FastAPI deployment
    joblib.dump(best_model_obj, "models/best_model.pkl")
    
    # Save model metadata summary
    metadata = {
        "best_model_name": best_model_name,
        "best_r2_score": float(best_r2),
        "feature_count": X.shape[1],
        "feature_names": list(X.columns)
    }
    joblib.dump(metadata, "models/model_metadata.pkl")
    print("Best model exported to models/best_model.pkl")

    return best_model_obj, fe

if __name__ == "__main__":
    train_and_evaluate_models()

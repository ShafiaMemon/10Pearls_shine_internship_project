# -*- coding: utf-8 -*-
"""
AQI_DS10pearls CI/CD-ready script

Supports two pipelines:
1. Feature pipeline: fetches weather & air-quality data, computes features, saves locally
2. Training pipeline: trains ML models (Random Forest & Ridge) and registers them in Hopsworks
"""

import os
import argparse
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import requests
from requests.exceptions import RequestException
import pandas as pd
import numpy as np
import joblib
import hopsworks
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# Configuration
# ----------------------------
latitude = 24.8607  # Karachi
longitude = 67.0011
start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 1, 1)

DATA_DIR = os.getenv("DATA_DIR", "./data")
os.makedirs(DATA_DIR, exist_ok=True)

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")  # store in GitHub Secrets

URL_WEATHER = "https://archive-api.open-meteo.com/v1/archive"
URL_AIR = "https://air-quality-api.open-meteo.com/v1/air-quality"

# ----------------------------
# Helper Functions
# ----------------------------
def fetch_open_meteo_chunk(lat, lon, start_dt, end_dt):
    """Fetch weather + air-quality data for a date range (1 month typical)"""
    params_weather = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_dt.date().isoformat(),
        "end_date": end_dt.date().isoformat(),
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "surface_pressure",
            "wind_speed_10m", "wind_direction_10m"
        ],
        "timezone": "auto"
    }

    params_air = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_dt.date().isoformat(),
        "end_date": end_dt.date().isoformat(),
        "hourly": [
            "pm10", "pm2_5", "carbon_monoxide",
            "nitrogen_dioxide", "sulphur_dioxide", "ozone"
        ],
        "timezone": "auto"
    }

    try:
        w_resp = requests.get(URL_WEATHER, params=params_weather, timeout=60)
        a_resp = requests.get(URL_AIR, params=params_air, timeout=60)
        w = w_resp.json()
        a = a_resp.json()

        if "hourly" not in w:
            print(f"⚠️ Weather API returned no hourly data for {start_dt} → {end_dt}")
            print(f"Response: {w}")
            return pd.DataFrame()

        if "hourly" not in a:
            print(f"⚠️ Air API returned no hourly data for {start_dt} → {end_dt}")
            print(f"Response: {a}")
            return pd.DataFrame()

        df_weather = pd.DataFrame(w["hourly"])
        df_air = pd.DataFrame(a["hourly"])
        df = pd.merge(df_weather, df_air, on="time", how="outer")
        df["time"] = pd.to_datetime(df["time"])
        return df.sort_values("time")

    except RequestException as e:
        print(f"❌ Request failed for {start_dt} → {end_dt}: {e}")
        return pd.DataFrame()
    except ValueError as e:
        print(f"⚠️ JSON decode error for {start_dt} → {end_dt}: {e}")
        return pd.DataFrame()


# ----------------------------
# Feature Engineering Pipeline
# ----------------------------
def run_feature_pipeline():
    """Fetch data, engineer features, and save to disk"""
    frames = []
    current = start_date

    while current < end_date:
        chunk_end = min(current + relativedelta(months=1), end_date)
        print(f"Fetching {current} → {chunk_end}")

        df_chunk = fetch_open_meteo_chunk(latitude, longitude, current, chunk_end)
        if not df_chunk.empty:
            frames.append(df_chunk)

        current = chunk_end
        time.sleep(2)  # prevent rate-limit

    if not frames:
        print("❌ No data fetched. Aborting feature pipeline.")
        return

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={
        "pm2_5": "pm25",
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity"
    })

    # Add simple features
    df["city"] = "Karachi"
    df["timestamp"] = df["time"]
    df = df.drop(columns=["time"])

    # Save
    out_path = os.path.join(DATA_DIR, "karachi_air_features_2024.parquet")
    df.to_parquet(out_path, index=False)
    print(f"✅ Feature pipeline complete. Saved to {out_path}")


# ----------------------------
# Training Pipeline
# ----------------------------
def run_training_pipeline():
    """Train models and register in Hopsworks"""
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY not set in environment variables!")

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    # Load features
    parquet_file = os.path.join(DATA_DIR, "karachi_air_features_2024.parquet")
    df = pd.read_parquet(parquet_file)
    df = df.dropna()

    target = "pm25"
    X = df.drop(columns=[target, "city", "timestamp"])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Random Forest ---
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    joblib.dump(rf_model, os.path.join(DATA_DIR, "pm25_rf_model.pkl"))

    print(f"Random Forest metrics → MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")

    # --- Ridge Regression ---
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    joblib.dump(ridge_model, os.path.join(DATA_DIR, "pm25_ridge_model.pkl"))

    # Register in Hopsworks
    mr = project.get_model_registry()

    rf_entry = mr.python.create_model(
        name="pm25_random_forest_model",
        metrics={"mae": mae, "r2": r2},
        description="Random Forest predicting PM2.5 using weather & pollutants"
    )
    rf_entry.save(os.path.join(DATA_DIR, "pm25_rf_model.pkl"))

    ridge_entry = mr.python.create_model(
        name="pm25_ridge_model",
        metrics={
            "mae": mean_absolute_error(y_test, ridge_model.predict(X_test)),
            "r2": r2_score(y_test, ridge_model.predict(X_test))
        },
        description="Ridge Regression baseline model"
    )
    ridge_entry.save(os.path.join(DATA_DIR, "pm25_ridge_model.pkl"))

    print("✅ Training pipeline complete. Models registered in Hopsworks.")


# ----------------------------
# CLI Execution
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["feature", "train"], required=True)
    args = parser.parse_args()

    if args.pipeline == "feature":
        run_feature_pipeline()
    elif args.pipeline == "train":
        run_training_pipeline()

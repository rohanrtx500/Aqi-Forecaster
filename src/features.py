"""
Phase 2: clean, merge, and feature-engineer the raw data saved in Phase 1.
Pandas/NumPy only. No model training here.
"""
import numpy as np
import pandas as pd

from src.utils import DATA_DIR

# Columns we'd like to keep if they exist (Phase 1 currently fetches pm25 only,
# but this stays flexible in case pm10/no2/o3 are added to data.py later).
POLLUTANT_COLS = ["pm25", "pm10", "no2", "o3"]
WEATHER_COLS = ["temperature", "humidity", "pressure", "wind_speed", "wind_direction", "rainfall"]

LAGS = [1, 3, 6, 12, 24, 48]
ROLLING_WINDOWS = [6, 12, 24]


def load_raw():
    pollution = pd.read_csv(f"{DATA_DIR}/raw_pollution.csv")
    weather = pd.read_csv(f"{DATA_DIR}/raw_weather.csv")
    pollution["timestamp"] = pd.to_datetime(pollution["timestamp"]).dt.floor("h")
    weather["timestamp"] = pd.to_datetime(weather["timestamp"]).dt.floor("h")
    
    pollution = pollution.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    weather = weather.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return pollution, weather


def merge_data(pollution: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(pollution, weather, on="timestamp", how="inner")
    return df.sort_values("timestamp").reset_index(drop=True)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate short numeric gaps (<=3 hours), then drop any rows still missing."""
    numeric_cols = [c for c in POLLUTANT_COLS + WEATHER_COLS if c in df.columns]
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=3, limit_direction="forward")
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """All lag/rolling values use only past rows (shift(1) before rolling) -> no leakage."""
    for lag in LAGS:
        df[f"pm25_lag_{lag}"] = df["pm25"].shift(lag)

    shifted = df["pm25"].shift(1)  # exclude current hour from its own rolling mean
    for window in ROLLING_WINDOWS:
        df[f"pm25_rolling_mean_{window}"] = shifted.rolling(window=window).mean()

    return df


def build_processed_dataset() -> pd.DataFrame:
    pollution, weather = load_raw()
    df = merge_data(pollution, weather)
    df = handle_missing(df)
    df = add_time_features(df)
    df = add_lag_and_rolling_features(df)

    # Drop rows with NaN caused by lag/rolling calculations (start of series)
    lag_roll_cols = [c for c in df.columns if "lag_" in c or "rolling_mean_" in c]
    df = df.dropna(subset=lag_roll_cols).reset_index(drop=True)

    return df


if __name__ == "__main__":
    processed = build_processed_dataset()
    out_path = f"{DATA_DIR}/processed_aqi.csv"
    processed.to_csv(out_path, index=False)
    print(f"Saved {len(processed)} rows, {len(processed.columns)} columns -> {out_path}")
    print(processed.head())

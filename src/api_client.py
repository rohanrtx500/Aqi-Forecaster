"""
Streamlit-facing client with seamless in-process fallback.
Allows the application to run either connected to a remote FastAPI backend
or 100% standalone for cloud deployments (Streamlit Cloud, Hugging Face, etc.).
"""
import os
from typing import Dict
import pandas as pd
import requests
import streamlit as st

from src.config import CITIES, city_models_dir, processed_csv_path, model_available, data_available
from src.predict import load_lstm_24h_artifacts, predict_next_24h
from src.live_api import fetch_live_air_quality
from src.utils import pm25_to_aqi

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
_local_artifact_cache: Dict[str, dict] = {}


class ApiError(Exception):
    pass


def _get_local_artifacts(city: str):
    if city not in _local_artifact_cache:
        if not model_available(city):
            raise ApiError(f"Forecast model unavailable for {city}")
        _local_artifact_cache[city] = load_lstm_24h_artifacts(city_models_dir(city))
    return _local_artifact_cache[city]


@st.cache_data(ttl=120)
def get_cities_status():
    try:
        resp = requests.get(f"{API_URL}/cities", timeout=2)
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return {
        city: {"model_available": model_available(city), "data_available": data_available(city)}
        for city in CITIES
    }


@st.cache_data(ttl=120)
def get_current(city: str):
    try:
        resp = requests.get(f"{API_URL}/current/{city}", timeout=2)
        if resp.ok:
            return resp.json()
    except Exception:
        pass

    if city not in CITIES:
        return None

    # Standalone fallback: query live real-time API
    live = fetch_live_air_quality(city)
    if live and "pm25" in live:
        pm25 = float(live["pm25"])
        aqi, category = pm25_to_aqi(pm25)
        return {
            "city": city,
            "timestamp": live.get("time", ""),
            "pm25": round(pm25, 1),
            "aqi": aqi,
            "category": category,
            "pollutants": live,
            "stream": "real-time",
        }

    if not data_available(city):
        return None

    try:
        df = pd.read_csv(processed_csv_path(city))
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        last = df.sort_values("timestamp").iloc[-1]
        pm25 = float(last["pm25"])
        aqi, category = pm25_to_aqi(pm25)
        return {
            "city": city,
            "timestamp": last["timestamp"].strftime("%Y-%m-%d %H:%M"),
            "pm25": round(pm25, 1),
            "aqi": aqi,
            "category": category,
            "pollutants": None,
            "stream": "cached",
        }
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_model_info(city: str):
    try:
        resp = requests.get(f"{API_URL}/model-info/{city}", timeout=2)
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    if not model_available(city):
        return None
    artifacts = _get_local_artifacts(city)
    return {
        "model_type": "PyTorch LSTM",
        "input_window_hours": artifacts["window"],
        "forecast_horizon_hours": artifacts["horizon"],
        "feature_count": len(artifacts["feature_cols"]),
        "metrics": artifacts.get("metrics"),
    }


def post_predict(city: str, records: list):
    try:
        resp = requests.post(f"{API_URL}/predict/{city}", json={"records": records}, timeout=5)
        if resp.ok:
            return resp.json()
    except Exception:
        pass

    # Standalone In-Process Model Prediction
    try:
        artifacts = _get_local_artifacts(city)
        pred_dict = predict_next_24h(records, artifacts)
        return {"city": city, "pm25_forecast": pred_dict["forecast"]}
    except Exception as e:
        raise ApiError(f"Prediction failed for {city}: {str(e)}")

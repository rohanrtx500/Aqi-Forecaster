"""
Multi-city FastAPI backend. Models are loaded lazily per city and cached —
never all at startup. No training happens here.
"""
import json
from typing import List, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from src.config import CITIES, city_models_dir, processed_csv_path, model_available, data_available
from src.predict import load_lstm_24h_artifacts, predict_next_24h
from src.utils import pm25_to_aqi
from src.live_api import fetch_live_air_quality

app = FastAPI(title="AirSense AI - Multi-City AQI Forecaster")

_artifact_cache: Dict[str, dict] = {}  # city -> loaded model artifacts


def _require_city(city: str):
    if city not in CITIES:
        raise HTTPException(status_code=404, detail=f"Unknown city: {city}")


def _get_artifacts(city: str):
    if city not in _artifact_cache:
        if not model_available(city):
            raise HTTPException(status_code=404, detail=f"Forecast model unavailable for {city}")
        _artifact_cache[city] = load_lstm_24h_artifacts(city_models_dir(city))
    return _artifact_cache[city]


class PredictRequest(BaseModel):
    records: List[Dict[str, float]]

    @field_validator("records")
    @classmethod
    def check_length(cls, v):
        if len(v) != 48:
            raise ValueError(f"records must contain exactly 48 hourly entries, got {len(v)}")
        return v


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/cities")
def list_cities():
    return {
        city: {"model_available": model_available(city), "data_available": data_available(city)}
        for city in CITIES
    }


@app.get("/current/{city}")
def current(city: str):
    _require_city(city)
    
    # Check for live real-time atmospheric stream
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
        raise HTTPException(status_code=404, detail=f"No pollution data available for {city}")
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


@app.get("/model-info/{city}")
def model_info(city: str):
    _require_city(city)
    if not model_available(city):
        raise HTTPException(status_code=404, detail=f"Forecast model unavailable for {city}")
    artifacts = _get_artifacts(city)
    info = {
        "model_type": "PyTorch LSTM",
        "input_window_hours": artifacts["window"],
        "forecast_horizon_hours": artifacts["horizon"],
        "feature_count": len(artifacts["features"]),
    }
    try:
        with open(f"{city_models_dir(city)}/metrics.json") as f:
            info["metrics"] = json.load(f)
    except FileNotFoundError:
        info["metrics"] = None
    return info


@app.post("/predict/{city}")
def predict(city: str, request: PredictRequest):
    _require_city(city)
    artifacts = _get_artifacts(city)
    try:
        forecast = predict_next_24h(request.records, artifacts)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "city": city,
        "forecast_hours": artifacts["horizon"],
        "pm25_forecast": forecast,
        "unit": "ug/m3",
    }

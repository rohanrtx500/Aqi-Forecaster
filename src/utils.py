"""Shared helpers used across the project."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY", "")
CITY_NAME = os.getenv("CITY_NAME", "Kolkata")
LATITUDE = float(os.getenv("LATITUDE", "22.5726"))
LONGITUDE = float(os.getenv("LONGITUDE", "88.3639"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def safe_get(url, headers=None, params=None, timeout=30):
    """GET request that never raises — returns None on any failure."""
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[safe_get] request failed: {url} -> {e}")
        return None


# ---- US EPA PM2.5 AQI breakpoints (24h avg, ug/m3) ----
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50, "Good"),
    (12.1, 35.4, 51, 100, "Moderate"),
    (35.5, 55.4, 101, 150, "Unhealthy for Sensitive Groups"),
    (55.5, 150.4, 151, 200, "Unhealthy"),
    (150.5, 250.4, 201, 300, "Very Unhealthy"),
    (250.5, 500.4, 301, 500, "Hazardous"),
]


def pm25_to_aqi(pm25: float):
    """Convert a PM2.5 concentration (ug/m3) to US AQI value + category."""
    if pm25 is None or pm25 != pm25:  # NaN check
        return None, "Unknown"
    pm25 = max(0.0, float(pm25))
    for c_lo, c_hi, i_lo, i_hi, category in PM25_BREAKPOINTS:
        if c_lo <= pm25 <= c_hi:
            aqi = (i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo) + i_lo
            return round(aqi), category
    # above scale -> cap at hazardous
    return 500, "Hazardous"

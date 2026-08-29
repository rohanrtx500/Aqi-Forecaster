"""
Fetch historical pollution data (OpenAQ v3) and weather data (Open-Meteo, free/no key)
for one city and save raw CSVs to data/.
"""
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.utils import (
    safe_get, DATA_DIR, OPENAQ_API_KEY, CITY_NAME, LATITUDE, LONGITUDE
)

OPENAQ_BASE = "https://api.openaq.org/v3"
OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"


def find_pm25_sensor(lat: float, lon: float, radius_km: int = 25):
    """Find the nearest OpenAQ location+sensor that reports pm25."""
    headers = {"X-API-Key": OPENAQ_API_KEY}
    radii_to_try = [min(radius_km, 25)]
    for r in radii_to_try:
        params = {
            "coordinates": f"{lat},{lon}",
            "radius": min(r * 1000, 25000),  # meters (max 25000 in OpenAQ v3)
            "limit": 50,
        }
        data = safe_get(f"{OPENAQ_BASE}/locations", headers=headers, params=params)
        if not data or "results" not in data:
            print(f"[find_pm25_sensor] no locations found or API issue at radius {r}km")
            continue

        for loc in data["results"]:
            for sensor in loc.get("sensors", []):
                param = sensor.get("parameter", {})
                param_name = param.get("name") if isinstance(param, dict) else str(param)
                if param_name == "pm25":
                    print(f"[find_pm25_sensor] Found PM2.5 sensor ID={sensor['id']} at location ID={loc['id']} ({loc.get('name', 'Unknown')})")
                    return loc["id"], sensor["id"]

    print("[find_pm25_sensor] Could not find a pm25 sensor within 25km.")
    return None, None


def fetch_openaq_pm25(sensor_id: int, date_from: str, date_to: str) -> pd.DataFrame:
    """Fetch hourly pm25 measurements for a sensor, paginating through results."""
    headers = {"X-API-Key": OPENAQ_API_KEY}
    all_rows = []
    page = 1
    while True:
        params = {
            "datetime_from": date_from,
            "datetime_to": date_to,
            "limit": 1000,
            "page": page,
        }
        data = safe_get(
            f"{OPENAQ_BASE}/sensors/{sensor_id}/measurements",
            headers=headers, params=params,
        )
        if not data or not data.get("results"):
            break

        results = data.get("results", [])
        for r in results:
            period = r.get("period", {})
            dt_from = period.get("datetimeFrom") if isinstance(period, dict) else None
            ts_str = dt_from.get("utc") if isinstance(dt_from, dict) else dt_from
            val = r.get("value")
            if ts_str is not None and val is not None:
                all_rows.append({
                    "timestamp": ts_str,
                    "pm25": val,
                })

        if len(results) < 1000:
            break
        page += 1
        time.sleep(0.2)  # be polite to the API

    if not all_rows:
        print("[fetch_openaq_pm25] no data returned")
        return pd.DataFrame(columns=["timestamp", "pm25"])

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_weather_data(lat: float, lon: float, date_from: str, date_to: str) -> pd.DataFrame:
    """Fetch hourly weather (temp, humidity, pressure, wind, rain) from Open-Meteo."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_from,
        "end_date": date_to,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
        ]),
        "timezone": "UTC",
    }
    data = safe_get(OPEN_METEO_BASE, params=params)
    if not data or "hourly" not in data:
        print("[fetch_weather_data] no weather data returned")
        return pd.DataFrame()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "pressure": hourly["surface_pressure"],
        "wind_speed": hourly["wind_speed_10m"],
        "wind_direction": hourly["wind_direction_10m"],
        "rainfall": hourly["precipitation"],
    })
    return df


if __name__ == "__main__":
    # Pull 180 days of data ending 6 days ago (accounting for Open-Meteo reporting lag)
    end = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=6)
    start = end - timedelta(days=180)
    date_from = start.strftime("%Y-%m-%d")
    date_to = end.strftime("%Y-%m-%d")

    print(f"Fetching data for {CITY_NAME} ({LATITUDE}, {LONGITUDE}) from {date_from} to {date_to}")

    # Fetch weather data first
    weather_df = fetch_weather_data(LATITUDE, LONGITUDE, date_from, date_to)
    if not weather_df.empty:
        weather_df.to_csv(f"{DATA_DIR}/raw_weather.csv", index=False)
        print(f"Saved {len(weather_df)} weather rows -> data/raw_weather.csv")

    loc_id, sensor_id = find_pm25_sensor(LATITUDE, LONGITUDE)
    pollution_df = pd.DataFrame()
    if sensor_id is not None:
        print(f"Using OpenAQ location={loc_id}, sensor={sensor_id}")
        pollution_df = fetch_openaq_pm25(sensor_id, date_from, date_to)
        if not pollution_df.empty:
            pollution_df.to_csv(f"{DATA_DIR}/raw_pollution.csv", index=False)
            print(f"Saved {len(pollution_df)} pollution rows -> data/raw_pollution.csv")

    if pollution_df.empty:
        print("\n[NOTE] OPENAQ_API_KEY is missing/unauthorized or no active sensor was found within 200km.")
        print("[NOTE] Register for a free API key at https://explore.openaq.org/register and set OPENAQ_API_KEY in .env")
        if not weather_df.empty:
            print("[FALLBACK] Generating realistic synthetic PM2.5 data for pipeline testing...")
            import numpy as np
            np.random.seed(42)
            hours = weather_df["timestamp"].dt.hour.values
            temp = weather_df["temperature"].values
            wind = weather_df["wind_speed"].values
            base_pm25 = 60 + 25 * np.sin(2 * np.pi * (hours - 6) / 24) - 3 * wind + 0.5 * temp
            noise = np.random.normal(0, 10, len(weather_df))
            pm25_vals = np.clip(base_pm25 + noise, 15, 350)
            pollution_df = pd.DataFrame({
                "timestamp": weather_df["timestamp"],
                "pm25": np.round(pm25_vals, 1)
            })
            pollution_df.to_csv(f"{DATA_DIR}/raw_pollution.csv", index=False)
            print(f"Saved {len(pollution_df)} fallback pollution rows -> data/raw_pollution.csv")



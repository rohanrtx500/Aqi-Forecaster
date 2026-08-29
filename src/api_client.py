"""
Streamlit-facing client for the FastAPI backend. All caching lives here so
app.py stays UI-only. Errors are raised as ApiError with a clean message -
no stack traces should ever reach the UI.
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


class ApiError(Exception):
    pass


def _get(path: str, timeout=10):
    try:
        resp = requests.get(f"{API_URL}{path}", timeout=timeout)
    except requests.exceptions.ConnectionError:
        raise ApiError(f"Cannot reach the API at {API_URL}. Start it with: uvicorn api:app --reload")
    except requests.exceptions.Timeout:
        raise ApiError("The API took too long to respond.")
    if resp.status_code == 404:
        return None  # caller decides how to present "not available"
    if not resp.ok:
        raise ApiError(resp.json().get("detail", f"API error ({resp.status_code})"))
    return resp.json()


def _post(path: str, json_body: dict, timeout=15):
    try:
        resp = requests.post(f"{API_URL}{path}", json=json_body, timeout=timeout)
    except requests.exceptions.ConnectionError:
        raise ApiError(f"Cannot reach the API at {API_URL}. Start it with: uvicorn api:app --reload")
    except requests.exceptions.Timeout:
        raise ApiError("The API took too long to respond.")
    if not resp.ok:
        try:
            detail = resp.json().get("detail", f"API error ({resp.status_code})")
        except Exception:
            detail = f"API error ({resp.status_code})"
        raise ApiError(str(detail))
    return resp.json()


@st.cache_data(ttl=120)
def get_cities_status():
    return _get("/cities") or {}


@st.cache_data(ttl=120)
def get_current(city: str):
    return _get(f"/current/{city}")


@st.cache_data(ttl=300)
def get_model_info(city: str):
    return _get(f"/model-info/{city}")


def post_predict(city: str, records: list):
    """Not cached - this is a POST tied to a user action (refresh)."""
    return _post(f"/predict/{city}", {"records": records})

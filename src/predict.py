"""
Phase 6: High-Speed Predictive Inference Engine for AirSense.
Includes:
1. Deterministic 24-Hour Forward Pass
2. Monte Carlo (MC) Dropout Uncertainty Estimation (P10, P50, P90)
3. Physically-Calibrated Interactive Scenario Simulator
4. Atmospheric Feature Sensitivity Attribution
5. Atmospheric Stability & Variance Index
"""
import numpy as np
import pandas as pd
import torch
import joblib

from src.utils import MODELS_DIR
from src.lstm_model import PM25LSTM, get_device


def load_lstm_24h_artifacts(model_dir: str = None):
    """Load model, scaler, and feature/window/horizon info once."""
    model_dir = model_dir or MODELS_DIR
    device = get_device()
    info = joblib.load(f"{model_dir}/lstm_24h_features.pkl")
    features, window, horizon = info["features"], info["window"], info["horizon"]

    model = PM25LSTM(n_features=len(features), output_size=horizon).to(device)
    model.load_state_dict(torch.load(f"{model_dir}/lstm_24h_model.pth", map_location=device))
    model.eval()

    scaler = joblib.load(f"{model_dir}/lstm_24h_scaler.pkl")

    return {
        "model": model,
        "scaler": scaler,
        "features": features,
        "window": window,
        "horizon": horizon,
        "device": device,
    }


def _prepare_tensor(records: list[dict], artifacts: dict):
    features = artifacts["features"]
    window = artifacts["window"]
    df = pd.DataFrame(records)

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    if len(df) != window:
        raise ValueError(f"Expected exactly {window} records, got {len(df)}")

    df = df[features]
    scaled = artifacts["scaler"].transform(df)
    x = torch.from_numpy(scaled.astype(np.float32)).unsqueeze(0).to(artifacts["device"])
    return x, df


def predict_next_24h(records: list[dict], artifacts: dict) -> list[float]:
    """Standard Deterministic 24-Hour Forward Pass."""
    x, _ = _prepare_tensor(records, artifacts)
    artifacts["model"].eval()
    with torch.no_grad():
        pred = artifacts["model"](x).cpu().numpy().flatten()
    return [round(float(max(0.0, v)), 2) for v in pred]


def predict_next_24h_mc_dropout(records: list[dict], artifacts: dict, n_samples: int = 20) -> dict:
    """
    Probabilistic Forecasting via Monte Carlo Dropout.
    Runs n_samples stochastic forward passes to calculate P10, P50, and P90 confidence ranges.
    """
    x, _ = _prepare_tensor(records, artifacts)
    model = artifacts["model"]
    
    model.train()
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x).cpu().numpy().flatten()
            samples.append(np.maximum(0.0, out))
    
    model.eval()
    samples = np.array(samples)

    p10 = np.percentile(samples, 10, axis=0)
    p50 = np.percentile(samples, 50, axis=0)
    p90 = np.percentile(samples, 90, axis=0)
    std = np.std(samples, axis=0)

    return {
        "p10": [round(float(v), 1) for v in p10],
        "p50": [round(float(v), 1) for v in p50],
        "p90": [round(float(v), 1) for v in p90],
        "std": [round(float(v), 2) for v in std],
    }


def predict_what_if_scenario(
    records: list[dict],
    artifacts: dict,
    wind_delta: float = 0.0,
    temp_delta: float = 0.0,
    rain_val: float = 0.0,
    emission_reduction_pct: float = 0.0,
) -> list[float]:
    """
    High-Speed Physically-Calibrated Interactive Scenario Simulation:
    Combines atmospheric ventilation, wet scavenging, thermal inversion dynamics,
    and municipal emission controls with neural model inference.
    """
    df = pd.DataFrame(records).copy()
    features = artifacts["features"]

    # Atmospheric physics response multipliers
    delta_wind = wind_delta * -0.028
    delta_temp = -temp_delta * 0.025
    delta_rain = -min(0.55, rain_val * 0.038)
    delta_curb = -(emission_reduction_pct / 100.0) * 0.75

    m_atmo = float(np.clip(1.0 + delta_wind + delta_temp + delta_rain + delta_curb, 0.25, 2.2))

    for col in df.columns:
        if "wind_speed" in col:
            df[col] = np.maximum(0.5, df[col] + wind_delta)
        elif "temperature" in col:
            df[col] = df[col] + temp_delta
        elif "rainfall" in col or "precipitation" in col:
            df[col] = np.maximum(0.0, df[col] + rain_val)
        elif "lag_" in col or "rolling_" in col or "pm25" in col:
            df[col] = df[col] * m_atmo

    scaled = artifacts["scaler"].transform(df[features])
    x = torch.from_numpy(scaled.astype(np.float32)).unsqueeze(0).to(artifacts["device"])

    artifacts["model"].eval()
    with torch.no_grad():
        neural_out = artifacts["model"](x).cpu().numpy().flatten()

    sim = np.maximum(5.0, neural_out * (0.65 + 0.35 * m_atmo))
    return [round(float(v), 2) for v in sim]


def compute_xai_feature_attributions(records: list[dict], artifacts: dict) -> dict:
    """
    Atmospheric Feature Attribution:
    Calculates gradient-based sensitivity contributions for the primary atmospheric drivers.
    """
    features = artifacts["features"]
    df = pd.DataFrame(records)[features]
    scaled = artifacts["scaler"].transform(df)
    x = torch.from_numpy(scaled.astype(np.float32)).unsqueeze(0).to(artifacts["device"])
    x.requires_grad = True

    model = artifacts["model"]
    model.eval()

    out = model(x)
    target = out.sum()
    target.backward()

    grads = x.grad.detach().cpu().numpy().squeeze(0)
    mean_abs_grads = np.mean(np.abs(grads), axis=0)

    pillars = {
        "💨 Wind Speed & Atmospheric Dispersion": 0.0,
        "🌡️ Thermal Inversion & Temperature Delta": 0.0,
        "💧 Relative Humidity & Moisture Trapping": 0.0,
        "⏳ Particulate Momentum (Lags)": 0.0,
        "⏰ Diurnal Traffic & Rush-Hour Cycles": 0.0,
    }

    for idx, f_name in enumerate(features):
        weight = float(mean_abs_grads[idx])
        if "wind" in f_name:
            pillars["💨 Wind Speed & Atmospheric Dispersion"] += weight
        elif "temperature" in f_name or "pressure" in f_name:
            pillars["🌡️ Thermal Inversion & Temperature Delta"] += weight
        elif "humidity" in f_name or "rainfall" in f_name or "precipitation" in f_name:
            pillars["💧 Relative Humidity & Moisture Trapping"] += weight
        elif "lag_" in f_name or "rolling_" in f_name:
            pillars["⏳ Particulate Momentum (Lags)"] += weight
        elif "hour_" in f_name or "day_" in f_name or "month" in f_name:
            pillars["⏰ Diurnal Traffic & Rush-Hour Cycles"] += weight

    total = sum(pillars.values()) or 1.0
    percentages = {k: round((v / total) * 100.0, 1) for k, v in pillars.items()}
    return percentages


def compute_atmospheric_anomaly_index(records: list[dict], current_pm: float) -> dict:
    """
    Atmospheric Stability & Variance Index.
    Calculates Z-score and assigns an environmental stability score.
    """
    if not records or current_pm is None:
        return {"anomaly_score": 5.0, "status": "Stable Atmospheric Baseline", "severity": "Normal", "color": "#10b981"}

    df = pd.DataFrame(records)
    pm_series = df["pm25"] if "pm25" in df.columns else pd.Series([current_pm])
    
    mean = pm_series.mean()
    std = pm_series.std() if pm_series.std() > 0 else 5.0
    z_score = abs((current_pm - mean) / std)

    score = min(100.0, round(float(z_score * 28.0), 1))

    if score > 70.0:
        status = "🚨 Significant Air Quality Surge Detected"
        severity = "Elevated"
        color = "#ef4444"
    elif score > 45.0:
        status = "⚠️ Moderate Atmospheric Variation"
        severity = "Moderate"
        color = "#f59e0b"
    else:
        status = "🟢 Stable Atmospheric Baseline"
        severity = "Normal"
        color = "#10b981"

    return {
        "anomaly_score": score,
        "z_score": round(float(z_score), 2),
        "status": status,
        "severity": severity,
        "color": color,
    }

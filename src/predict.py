"""
Phase 6: reusable prediction logic for the trained 24h LSTM.
No training happens here — only loading saved artifacts and running inference.
"""
import numpy as np
import pandas as pd
import torch
import joblib

from src.utils import MODELS_DIR
from src.lstm_model import PM25LSTM, get_device


def load_lstm_24h_artifacts(model_dir: str = None):
    """Load model, scaler, and feature/window/horizon info once. Defaults to
    the original flat models/ folder for backward compatibility."""
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


def predict_next_24h(records: list[dict], artifacts: dict) -> list[float]:
    """
    records: list of 48 dicts, each containing all required feature values
              (in any key order — columns are reordered to match training).
    Returns: list of 24 predicted PM2.5 values (raw units, no inverse-transform
             needed since the target was never scaled during training).
    """
    features = artifacts["features"]
    window = artifacts["window"]

    df = pd.DataFrame(records)

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    if len(df) != window:
        raise ValueError(f"Expected exactly {window} records, got {len(df)}")

    df = df[features]  # enforce training column order

    if not np.issubdtype(df.to_numpy().dtype, np.number):
        raise ValueError("All feature values must be numeric")
    if df.isnull().any().any():
        raise ValueError("Input contains missing/NaN values")

    scaled = artifacts["scaler"].transform(df)
    x = torch.from_numpy(scaled.astype(np.float32)).unsqueeze(0).to(artifacts["device"])  # (1, window, n_features)

    with torch.no_grad():
        pred = artifacts["model"](x).cpu().numpy().flatten()

    return [round(float(v), 2) for v in pred]

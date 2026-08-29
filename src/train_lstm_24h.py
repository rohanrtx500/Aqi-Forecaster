"""
Phase 5: extend LSTM from 1-hour to 24-hour PM2.5 forecasting.
Reuses Phase 4's model class, split logic, and preprocessing.
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.utils import MODELS_DIR
from src.train import load_processed, chronological_split, get_feature_list, evaluate, TARGET
from src.lstm_model import PM25LSTM, create_sequences, create_multistep_sequences, get_device, WINDOW
from src.train_lstm import EPOCHS, BATCH_SIZE, PATIENCE, LR

HORIZON = 24
CHECK_HOURS = [1, 6, 12, 24]  # horizons to report MAE for


def make_loader_24h(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_lstm_24h(model, train_loader, val_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_state = None
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(criterion(model(xb), yb).item())
        val_loss = float(np.mean(val_losses))
        print(f"Epoch {epoch:02d}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    df = load_processed()
    train_df, val_df, test_df = chronological_split(df)
    features = get_feature_list(df)

    # Scaler fit on TRAIN ONLY (same rule as Phase 4)
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_df[features])
    val_X = scaler.transform(val_df[features])
    test_X = scaler.transform(test_df[features])

    train_y = train_df[TARGET].values
    val_y = val_df[TARGET].values
    test_y = test_df[TARGET].values

    X_train, y_train = create_multistep_sequences(train_X, train_y, WINDOW, HORIZON)
    X_val, y_val = create_multistep_sequences(val_X, val_y, WINDOW, HORIZON)
    X_test, y_test = create_multistep_sequences(test_X, test_y, WINDOW, HORIZON)
    print(f"Sequences -> train:{X_train.shape} val:{X_val.shape} test:{X_test.shape} (y horizon={HORIZON})")

    train_loader = make_loader_24h(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader_24h(X_val, y_val, BATCH_SIZE, shuffle=False)
    test_loader = make_loader_24h(X_test, y_test, BATCH_SIZE, shuffle=False)

    model = PM25LSTM(n_features=len(features), output_size=HORIZON).to(device)
    model = train_lstm_24h(model, train_loader, val_loader, device)

    # Evaluate on test set
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)  # (n_samples, 24)

    overall_mae, overall_rmse = evaluate(y_test.flatten(), y_pred.flatten())
    horizon_mae = {h: float(np.mean(np.abs(y_test[:, h - 1] - y_pred[:, h - 1]))) for h in CHECK_HOURS}

    # --- Recompute Phase 3/4 models at the +1h horizon for comparison ---
    persist_pred_1h = test_df["pm25_lag_1"].values[WINDOW:]
    persist_true_1h = test_df[TARGET].values[WINDOW:]
    persist_mae, persist_rmse = evaluate(persist_true_1h, persist_pred_1h)

    xgb_mae = xgb_rmse = None
    try:
        xgb_model = joblib.load(f"{MODELS_DIR}/xgboost_model.pkl")
        xgb_features = joblib.load(f"{MODELS_DIR}/feature_list.pkl")
        xgb_pred = xgb_model.predict(test_df[xgb_features])[WINDOW:]
        xgb_mae, xgb_rmse = evaluate(persist_true_1h, xgb_pred)
    except FileNotFoundError:
        pass

    lstm1h_mae = lstm1h_rmse = None
    try:
        lstm1h_scaler = joblib.load(f"{MODELS_DIR}/lstm_scaler.pkl")
        lstm1h_info = joblib.load(f"{MODELS_DIR}/lstm_features.pkl")
        lstm1h_model = PM25LSTM(n_features=len(lstm1h_info["features"])).to(device)
        lstm1h_model.load_state_dict(torch.load(f"{MODELS_DIR}/lstm_model.pth", map_location=device))
        lstm1h_model.eval()
        test_X_1h = lstm1h_scaler.transform(test_df[lstm1h_info["features"]])
        Xs, ys = create_sequences(test_X_1h, test_df[TARGET].values, lstm1h_info["window"], horizon=1)
        with torch.no_grad():
            p = lstm1h_model(torch.from_numpy(Xs).to(device)).cpu().numpy().flatten()
        lstm1h_mae, lstm1h_rmse = evaluate(ys, p)
    except FileNotFoundError:
        pass

    print("\nModel              MAE      RMSE")
    print(f"Persistence        {persist_mae:.3f}    {persist_rmse:.3f}")
    if xgb_mae is not None:
        print(f"XGBoost            {xgb_mae:.3f}    {xgb_rmse:.3f}")
    if lstm1h_mae is not None:
        print(f"LSTM 1H            {lstm1h_mae:.3f}    {lstm1h_rmse:.3f}")
    print(f"LSTM 24H (+1h)     {horizon_mae[1]:.3f}    -")
    print(f"LSTM 24H (overall) {overall_mae:.3f}    {overall_rmse:.3f}")
    print("\nLSTM 24H MAE by horizon:")
    for h in CHECK_HOURS:
        print(f"  +{h}h: {horizon_mae[h]:.3f}")

    # --- Plot 1: actual vs predicted for one test window ---
    sample_idx = len(y_test) // 2  # pick a period in the middle of the test set
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, HORIZON + 1), y_test[sample_idx], marker="o", label="Actual")
    plt.plot(range(1, HORIZON + 1), y_pred[sample_idx], marker="x", label="Predicted")
    plt.xlabel("Hours ahead")
    plt.ylabel("PM2.5")
    plt.title("Actual vs Predicted PM2.5 (sample 24h forecast)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{MODELS_DIR}/lstm_24h_actual_vs_pred.png")
    plt.close()

    # --- Plot 2: error by horizon ---
    all_horizon_mae = [float(np.mean(np.abs(y_test[:, h] - y_pred[:, h]))) for h in range(HORIZON)]
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, HORIZON + 1), all_horizon_mae, marker="o")
    plt.xlabel("Forecast horizon (hours ahead)")
    plt.ylabel("MAE")
    plt.title("Forecast Error by Horizon")
    plt.tight_layout()
    plt.savefig(f"{MODELS_DIR}/lstm_24h_error_by_horizon.png")
    plt.close()

    # --- Save model artifacts ---
    torch.save(model.state_dict(), f"{MODELS_DIR}/lstm_24h_model.pth")
    joblib.dump(scaler, f"{MODELS_DIR}/lstm_24h_scaler.pkl")
    joblib.dump({"features": features, "window": WINDOW, "horizon": HORIZON}, f"{MODELS_DIR}/lstm_24h_features.pkl")

    # --- Save metrics ---
    metrics = {
        "persistence": {"mae": persist_mae, "rmse": persist_rmse},
        "xgboost": {"mae": xgb_mae, "rmse": xgb_rmse},
        "lstm_1h": {"mae": lstm1h_mae, "rmse": lstm1h_rmse},
        "lstm_24h_overall": {"mae": overall_mae, "rmse": overall_rmse},
        "lstm_24h_by_horizon_mae": horizon_mae,
    }
    with open(f"{MODELS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved -> lstm_24h_model.pth, lstm_24h_scaler.pkl, lstm_24h_features.pkl, "
          f"metrics.json, lstm_24h_actual_vs_pred.png, lstm_24h_error_by_horizon.png")

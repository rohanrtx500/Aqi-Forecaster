"""
Phase 4: train PyTorch LSTM for next-hour PM2.5, compare against Phase 3 baselines.
"""
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from src.utils import MODELS_DIR
from src.train import load_processed, chronological_split, get_feature_list, evaluate, TARGET
from src.lstm_model import PM25LSTM, create_sequences, get_device, WINDOW

EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 5
LR = 1e-3


def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_lstm(model, train_loader, val_loader, device):
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

    # Fit scaler on TRAIN ONLY, then transform all splits
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_df[features])
    val_X = scaler.transform(val_df[features])
    test_X = scaler.transform(test_df[features])

    train_y = train_df[TARGET].values
    val_y = val_df[TARGET].values
    test_y = test_df[TARGET].values

    # Build sequences separately per split (no window crossing split boundaries)
    X_train, y_train = create_sequences(train_X, train_y, window=WINDOW, horizon=1)
    X_val, y_val = create_sequences(val_X, val_y, window=WINDOW, horizon=1)
    X_test, y_test = create_sequences(test_X, test_y, window=WINDOW, horizon=1)
    print(f"Sequences -> train:{X_train.shape} val:{X_val.shape} test:{X_test.shape}")

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)
    test_loader = make_loader(X_test, y_test, BATCH_SIZE, shuffle=False)

    model = PM25LSTM(n_features=len(features)).to(device)
    model = train_lstm(model, train_loader, val_loader, device)

    # Evaluate on test set
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    y_pred = np.concatenate(preds).flatten()
    lstm_mae, lstm_rmse = evaluate(y_test, y_pred)

    # Recompute baselines on the same test rows for a fair comparison
    persist_pred = test_df["pm25_lag_1"].values[WINDOW:]  # align with sequence targets
    persist_mae, persist_rmse = evaluate(y_test, persist_pred)

    try:
        xgb_model = joblib.load(f"{MODELS_DIR}/xgboost_model.pkl")
        xgb_features = joblib.load(f"{MODELS_DIR}/feature_list.pkl")
        xgb_pred = xgb_model.predict(test_df[xgb_features])[WINDOW:]
        xgb_mae, xgb_rmse = evaluate(y_test, xgb_pred)
    except FileNotFoundError:
        xgb_mae, xgb_rmse = None, None

    print("\nModel          MAE       RMSE")
    print(f"Persistence    {persist_mae:.3f}    {persist_rmse:.3f}")
    if xgb_mae is not None:
        print(f"XGBoost        {xgb_mae:.3f}    {xgb_rmse:.3f}")
    print(f"LSTM           {lstm_mae:.3f}    {lstm_rmse:.3f}")

    torch.save(model.state_dict(), f"{MODELS_DIR}/lstm_model.pth")
    joblib.dump(scaler, f"{MODELS_DIR}/lstm_scaler.pkl")
    joblib.dump({"features": features, "window": WINDOW}, f"{MODELS_DIR}/lstm_features.pkl")
    print(f"\nSaved -> {MODELS_DIR}/lstm_model.pth, lstm_scaler.pkl, lstm_features.pkl")

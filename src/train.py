"""
Phase 3: baseline models (Persistence, XGBoost) with chronological split.
No LSTM here. No random splitting. No leakage.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from src.utils import DATA_DIR, MODELS_DIR

TARGET = "pm25"
# Columns that must not be used as model inputs (target, timestamp, or raw string cols)
NON_FEATURE_COLS = ["timestamp", TARGET]


def load_processed():
    df = pd.read_csv(f"{DATA_DIR}/processed_aqi.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def chronological_split(df: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test


def get_feature_list(df: pd.DataFrame):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def persistence_baseline(test_df: pd.DataFrame):
    """Predict next PM2.5 = previous hour's PM2.5 (pm25_lag_1, already leak-free)."""
    y_true = test_df[TARGET].values
    y_pred = test_df["pm25_lag_1"].values
    return evaluate(y_true, y_pred)


def train_xgboost(train_df, val_df, test_df, features):
    X_train, y_train = train_df[features], train_df[TARGET]
    X_val, y_val = val_df[features], val_df[TARGET]
    X_test, y_test = test_df[features], test_df[TARGET]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mae",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    mae, rmse = evaluate(y_test, model.predict(X_test))
    return model, mae, rmse


if __name__ == "__main__":
    df = load_processed()
    train_df, val_df, test_df = chronological_split(df)
    features = get_feature_list(df)

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}  Features: {len(features)}")

    persist_mae, persist_rmse = persistence_baseline(test_df)
    xgb_model, xgb_mae, xgb_rmse = train_xgboost(train_df, val_df, test_df, features)

    print("\nModel          MAE       RMSE")
    print(f"Persistence    {persist_mae:.3f}    {persist_rmse:.3f}")
    print(f"XGBoost        {xgb_mae:.3f}    {xgb_rmse:.3f}")

    joblib.dump(xgb_model, f"{MODELS_DIR}/xgboost_model.pkl")
    joblib.dump(features, f"{MODELS_DIR}/feature_list.pkl")
    print(f"\nSaved model -> {MODELS_DIR}/xgboost_model.pkl")
    print(f"Saved feature list -> {MODELS_DIR}/feature_list.pkl")

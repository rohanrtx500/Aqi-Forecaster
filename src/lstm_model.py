"""
Phase 4: PyTorch LSTM model + sequence builder.
Kept reusable so a later phase can extend target from 1-hour to 24-hour forecasts.
"""
import numpy as np
import torch
import torch.nn as nn

WINDOW = 48  # hours of history used as input


class PM25LSTM(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)          # out: (batch, seq_len, hidden)
        last_step = out[:, -1, :]       # take final timestep's hidden state
        return self.fc(last_step)       # (batch, output_size)


def create_sequences(features: np.ndarray, target: np.ndarray, window: int = WINDOW, horizon: int = 1):
    """
    Slide a window of length `window` over `features`; label = target `horizon`
    steps after the window ends. Only uses data within the array passed in,
    so call separately per split (train/val/test) to avoid cross-split leakage.
    """
    X, y = [], []
    n = len(features)
    for i in range(n - window - horizon + 1):
        X.append(features[i:i + window])
        y.append(target[i + window + horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def create_multistep_sequences(features: np.ndarray, target: np.ndarray, window: int = WINDOW, horizon: int = 24):
    """
    Same idea as create_sequences, but label = the next `horizon` future values
    (t+1 ... t+horizon), so a single window predicts a full multi-step forecast.
    """
    X, y = [], []
    n = len(features)
    for i in range(n - window - horizon + 1):
        X.append(features[i:i + window])
        y.append(target[i + window: i + window + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

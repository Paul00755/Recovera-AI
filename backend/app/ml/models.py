"""
Recovera AI 2.0: Deep Learning & Baseline Architectures
Implements the multi-model comparison shortlist from Section 4 of the proposal:
- Persistence Baseline
- Historical Mean / Trend Baseline
- Random Forest (Classical ML)
- GRU (Recurrent)
- LSTM (Recurrent)
- 1D Temporal CNN (Convolutional)
"""
import time
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestRegressor
import paddle
import paddle.nn as nn
import paddle.optimizer as optim

# ----------------------------------------------------
# 1. Non-Deep Learning Baselines
# ----------------------------------------------------
class PersistenceBaseline:
    """Predicts next day's pain from the latest observed pain in window."""
    def __init__(self):
        self.name = "Persistence"
        self.params_count = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass # Zero-parameter heuristic

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X shape: (N, window_size, 5) -> feature 0 is normalized pain (0-1)
        latest_pain_norm = X[:, -1, 0]
        return np.clip(latest_pain_norm * 10.0, 0.0, 10.0)


class HistoricalMeanTrendBaseline:
    """Predicts next day's pain by extrapolating the empirical velocity over the window."""
    def __init__(self):
        self.name = "Historical Mean / Trend"
        self.params_count = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        latest = X[:, -1, 0]
        prior = X[:, -2, 0]
        slope = latest - prior
        pred = (latest + (0.5 * slope)) * 10.0
        return np.clip(pred, 0.0, 10.0)


class RandomForestBaseline:
    """Classical tabular machine-learning baseline using flattened time-lag features."""
    def __init__(self, n_estimators: int = 100, max_depth: int = 6):
        self.name = "Random Forest (Classical ML)"
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        self.params_count = 0

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        # Flatten (N, window_size, 5) -> (N, window_size * 5)
        N, W, F = X.shape
        return X.reshape(N, W * F)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_flat = self._flatten(X)
        self.model.fit(X_flat, y)
        self.params_count = sum(tree.tree_.node_count for tree in self.model.estimators_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._flatten(X)
        pred = self.model.predict(X_flat)
        return np.clip(pred, 0.0, 10.0)


# ----------------------------------------------------
# 2. Deep Learning Candidates (PaddlePaddle 3.0)
# ----------------------------------------------------
class GRUForecaster(nn.Layer):
    """Gated Recurrent Unit sequence forecaster for longitudinal recovery."""
    def __init__(self, input_dim: int = 5, hidden_dim: int = 24, num_layers: int = 1):
        super().__init__()
        self.name = "GRU (Gated Recurrent Unit)"
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, time_major=False)
        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (N, seq_len, input_dim)
        out, _ = self.gru(x)
        last_step = out[:, -1, :] # Take final temporal embedding
        drop = self.dropout(last_step)
        pred = self.fc(drop)
        return pred.squeeze(-1) * 10.0 # Scale back to 0-10 pain score


class LSTMForecaster(nn.Layer):
    """Long Short-Term Memory network with separate cell and hidden state gates."""
    def __init__(self, input_dim: int = 5, hidden_dim: int = 24, num_layers: int = 1):
        super().__init__()
        self.name = "LSTM (Long Short-Term Memory)"
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, time_major=False)
        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        drop = self.dropout(last_step)
        pred = self.fc(drop)
        return pred.squeeze(-1) * 10.0


class TemporalCNNForecaster(nn.Layer):
    """1D Temporal Convolutional Network with causal convolutions."""
    def __init__(self, input_dim: int = 5, num_channels: int = 20):
        super().__init__()
        self.name = "1D Temporal CNN"
        # Conv1D expects shape (N, channels, length)
        self.conv1 = nn.Conv1D(in_channels=input_dim, out_channels=num_channels, kernel_size=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1D(in_channels=num_channels, out_channels=num_channels, kernel_size=2, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1D(1)
        self.fc = nn.Linear(num_channels, 1)

    def forward(self, x):
        # x is (N, length, input_dim) -> transpose to (N, input_dim, length)
        x_trans = paddle.transpose(x, [0, 2, 1])
        c1 = self.relu1(self.conv1(x_trans))
        c2 = self.relu2(self.conv2(c1))
        p = self.pool(c2).squeeze(-1)
        pred = self.fc(p)
        return pred.squeeze(-1) * 10.0


def count_parameters(layer: nn.Layer) -> int:
    return sum(p.numel().item() if hasattr(p.numel(), "item") else int(np.prod(p.shape)) for p in layer.parameters())


def train_deep_model(
    model: nn.Layer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 0.005
) -> Dict[str, Any]:
    """Trains a candidate deep learning model using Adam and L1/MSE loss with early stopping."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(learning_rate=lr, parameters=model.parameters())

    X_tr_t = paddle.to_tensor(X_train, dtype="float32")
    y_tr_t = paddle.to_tensor(y_train, dtype="float32")
    X_va_t = paddle.to_tensor(X_val, dtype="float32")
    y_va_t = paddle.to_tensor(y_val, dtype="float32")

    n_samples = len(X_train)
    indices = np.arange(n_samples)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        np.random.shuffle(indices)
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            bx = X_tr_t[batch_idx]
            by = y_tr_t[batch_idx]

            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        # Validation step
        model.eval()
        with paddle.no_grad():
            val_out = model(X_va_t)
            v_loss = criterion(val_out, y_va_t).item()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}

    if best_weights:
        model.set_state_dict(best_weights)

    return {"best_val_loss": best_val_loss}

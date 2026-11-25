"""Optional LSTM extension matching the report's future-work direction.

TensorFlow is intentionally imported inside the builder so the main Streamlit
application stays lightweight. Install requirements-optional.txt to use it.
"""

from __future__ import annotations

import numpy as np


def make_sequences(values: np.ndarray, window: int = 24) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) <= window:
        raise ValueError("values must contain more observations than the sequence window")
    X = np.stack([values[index - window:index] for index in range(window, len(values))])
    y = values[window:]
    return X[..., None], y


def build_lstm_classifier(window: int, feature_count: int = 1):
    """Build an optional Keras LSTM classifier for sequential fraud signals."""
    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install requirements-optional.txt to enable the LSTM extension.") from exc

    model = Sequential([
        LSTM(64, input_shape=(window, feature_count), return_sequences=False),
        Dropout(0.25),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "AUC", "Recall"])
    return model

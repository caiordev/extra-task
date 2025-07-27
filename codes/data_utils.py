"""Utility functions for loading the vibration dataset and generating windows
for LSTM shared-sensor architecture (Option B of the assignment).

This module DOES NOT depend on TensorFlow and can be reused in notebooks or
stand-alone scripts.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

WINDOW_COL_PREFIX = "sensor_"
LABEL_COL = "label"
TIMESTAMP_COL = "Timestamp"

def load_csv_parts(dataset_dir: Path | str) -> pd.DataFrame:
    """Load all `dataset_parte_*.csv` files and concatenate them.

    The files are expected to reside in *dataset_dir* and share identical
    columns.
    """
    dataset_dir = Path(dataset_dir)
    csv_paths = sorted(dataset_dir.glob("dataset_parte_*.csv"))
    if not csv_paths:
        raise FileNotFoundError("No CSV parts found in directory " f"{dataset_dir!s}")

    # Use low-memory=False to avoid dtype inference issues and preserve speed.
    dfs = [pd.read_csv(p, low_memory=False) for p in csv_paths]
    data = pd.concat(dfs, ignore_index=True)
    return data

def get_sensor_columns(df: pd.DataFrame) -> List[str]:
    """Return list of sensor_* columns in the correct order."""
    return [c for c in df.columns if c.startswith(WINDOW_COL_PREFIX)]

def window_generator(
    df: pd.DataFrame,
    window_size: int = 240,
    step_size: int = 1,
    use_diff_mean: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:

    
    """Create windows per sensor.

    For each sensor column `sensor_i`, creates rolling windows of *window_size*
    with stride *step_size*. The resulting shape is (n_windows, window_size).

    Labeling strategy (binary):
    1 for a window if the *label* column equals the sensor name for ANY row in
    that window (i.e. failure reported for that sensor). Otherwise 0.
    This follows the assignment guideline of comparing a sensor against the
    mean of others and detecting failures.
    """
    sensor_cols = get_sensor_columns(df)

    # Pre-allocate lists to collect windows/labels across sensors
    all_windows: List[np.ndarray] = []
    all_labels: List[int] = []

    for sensor_col in sensor_cols:
        if use_diff_mean:
            # subtract mean of all sensors row-wise to emphasize anomaly
            mean_all = df[sensor_cols].mean(axis=1).to_numpy()
            sensor_series = (df[sensor_col] - mean_all).to_numpy()
        else:
            sensor_series = df[sensor_col].to_numpy()
        label_matches = (df[LABEL_COL] == sensor_col).to_numpy()

        # Generate indices for rolling windows
        max_start = len(sensor_series) - window_size + 1
        for start in range(0, max_start, step_size):
            end = start + window_size
            window = sensor_series[start:end]
            if np.isnan(window).any():
                # Skip windows with NaNs
                continue
            label_window = 1 if label_matches[start:end].any() else 0
            all_windows.append(window.astype(np.float32))
            all_labels.append(label_window)

    X = np.stack(all_windows)  # shape (n_windows, window_size)
    y = np.array(all_labels, dtype=np.int32)
    return X, y

def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    random_state: int | None = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split of X/y into train/val/test sets."""
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=1 - train_frac, stratify=y, random_state=random_state
    )
    val_size_adjusted = val_frac / (1 - train_frac)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_size_adjusted, stratify=y_temp, random_state=random_state
    )
    return X_train, y_train, X_val, y_val, X_test, y_test

__all__ = [
    "load_csv_parts",
    "window_generator",
    "train_val_test_split",
    "get_sensor_columns",
]

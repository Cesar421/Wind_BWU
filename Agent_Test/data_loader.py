"""
Shared Data Loading Utilities for Agent_Test models.

Loads face-averaged Cp time series from:
    Data/Data_All_The_BDH_PostProcess/<alpha>/<building_ratio>/Data/
        windward_avg_angle_<A>.npy   shape (32768,) float32
        leeward_avg_angle_<A>.npy
        sideleft_avg_angle_<A>.npy
        sideright_avg_angle_<A>.npy

Key constraints (from repo notes):
  - Always use float32 to avoid OOM.
  - Use sliding-window step >= 10 for multi-building loading (~1.8 GB).
  - Chronological 70/15/15 split — no shuffling.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Project root (Agent_Test/ sits inside the project root)
PROJECT_ROOT = Path(__file__).parent.parent
POSTPROCESS_DIR = PROJECT_ROOT / "Data" / "Data_All_The_BDH_PostProcess"

FACES = ["windward", "leeward", "sideleft", "sideright"]

ANGLES_1XX = list(range(0, 55, 5))    # 11 angles for 1:1:x buildings
ANGLES_23XX = list(range(0, 105, 5))  # 21 angles for 2:1:x / 3:1:x


# ──────────────────────────────────────────────────────────────────────────────
# Low-level loading
# ──────────────────────────────────────────────────────────────────────────────

def load_faces(alpha: str, ratio: str, angle: int,
               data_dir: Optional[Path] = None) -> np.ndarray:
    """
    Load the 4 face-averaged Cp arrays for a single (alpha, ratio, angle).

    Returns
    -------
    np.ndarray  shape (32768, 4)  dtype float32
        Columns: windward, leeward, sideleft, sideright
    """
    base = (data_dir or POSTPROCESS_DIR) / alpha / ratio / "Data"
    arrays = []
    for face in FACES:
        p = base / f"{face}_avg_angle_{angle}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        arrays.append(np.load(str(p)).astype(np.float32))
    return np.stack(arrays, axis=1)  # (32768, 4)


def available_angles(alpha: str, ratio: str,
                     data_dir: Optional[Path] = None) -> List[int]:
    """Return all angles that have windward .npy files on disk."""
    base = (data_dir or POSTPROCESS_DIR) / alpha / ratio / "Data"
    angles = sorted({
        int(m.group(1))
        for f in os.listdir(str(base))
        if (m := re.match(r"windward_avg_angle_(\d+)\.npy$", f))
    })
    return angles


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation
# ──────────────────────────────────────────────────────────────────────────────

def zscore_fit(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean/std on a (T, n_features) array."""
    mu = data.mean(axis=0).astype(np.float32)
    sigma = data.std(axis=0).astype(np.float32)
    sigma[sigma == 0] = 1.0
    return mu, sigma


def zscore_transform(data: np.ndarray, mu: np.ndarray,
                     sigma: np.ndarray) -> np.ndarray:
    return ((data - mu) / sigma).astype(np.float32)


def zscore_inverse(data: np.ndarray, mu: np.ndarray,
                   sigma: np.ndarray) -> np.ndarray:
    return (data * sigma + mu).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Sequence window builder
# ──────────────────────────────────────────────────────────────────────────────

def build_sequences(
    ts: np.ndarray,           # (T, n_features)
    seq_length: int = 100,
    step: int = 1,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window sequence builder.

    Parameters
    ----------
    ts         : (T, n_features) float32
    seq_length : look-back window
    step       : stride (≥10 recommended for large datasets)
    horizon    : forecast horizon (1 = single-step)

    Returns
    -------
    X : (N, seq_length, n_features)  float32
    y : (N, horizon)                 float32  — windward face (col 0)
    """
    T = len(ts)
    indices = range(0, T - seq_length - horizon + 1, step)
    X = np.stack([ts[i: i + seq_length] for i in indices], axis=0)
    y = np.stack([ts[i + seq_length: i + seq_length + horizon, 0] for i in indices], axis=0)
    return X.astype(np.float32), y.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Single-building dataset
# ──────────────────────────────────────────────────────────────────────────────

def load_single_building(
    alpha: str = "Alpha1_4",
    ratio: str = "2_1_3",
    angles: Optional[List[int]] = None,
    seq_length: int = 100,
    step: int = 10,
    horizon: int = 1,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    data_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """
    Load and prepare training/validation/test data for one building.

    All angles for the building are concatenated chronologically, then
    split 70/15/15 and windowed.

    Returns
    -------
    dict with keys:
        X_train, y_train, X_val, y_val, X_test, y_test  — np.ndarray
        mu, sigma           — normalisation parameters (per face)
        test_seed           — (seq_length, n_features)  raw normalised ts
        y_future            — (500,)  ground-truth next 500 windward steps
    """
    if angles is None:
        angles = available_angles(alpha, ratio, data_dir)

    # Concatenate all angles into one time series (chronological)
    segments = []
    for angle in sorted(angles):
        seg = load_faces(alpha, ratio, angle, data_dir)  # (32768, 4)
        segments.append(seg)
    ts_full = np.concatenate(segments, axis=0)  # (T_total, 4)

    # Chronological split on RAW data (before windowing) to avoid leakage
    T = len(ts_full)
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    ts_train = ts_full[:train_end]
    ts_val = ts_full[train_end:val_end]
    ts_test = ts_full[val_end:]

    # Fit normalisation ONLY on training data
    mu, sigma = zscore_fit(ts_train)
    ts_train = zscore_transform(ts_train, mu, sigma)
    ts_val = zscore_transform(ts_val, mu, sigma)
    ts_test = zscore_transform(ts_test, mu, sigma)

    # Build sequences
    X_train, y_train = build_sequences(ts_train, seq_length, step, horizon)
    X_val, y_val = build_sequences(ts_val, seq_length, max(step, 5), horizon)
    X_test, y_test = build_sequences(ts_test, seq_length, max(step, 5), horizon)

    # Seed for autoregressive forecasting
    test_seed = ts_test[:seq_length]  # (seq_length, 4)
    y_future = ts_test[seq_length: seq_length + 500, 0]  # windward, denorm later

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val,     "y_val": y_val,
        "X_test": X_test,   "y_test": y_test,
        "mu": mu, "sigma": sigma,
        "test_seed": test_seed,
        "y_future": y_future,
        "n_features": 4,
        "seq_length": seq_length,
        "horizon": horizon,
        "alpha": alpha, "ratio": ratio,
    }

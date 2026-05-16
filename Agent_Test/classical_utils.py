"""
Shared utilities for classical baselines (Ridge, RandomForest, XGBoost).

These models do NOT use the PyTorch training loop. Instead:
  * Sliding windows (N, seq_length, n_features) are flattened to (N, seq_length * n_features)
  * .fit(X_flat, y) / .predict(X_flat) interface (sklearn / xgboost)
  * Same evaluation metrics, same multi-step autoregressive forecast,
    same CSV output format as the DL models.

Round 1: single building (Alpha1_4 / 2_1_3) via --building flag.
"""

import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from data_loader import load_all_buildings, load_single_building
from train_utils import (compute_metrics, save_metrics_csv,
                         save_pred_vs_actual, save_forecasts, get_data)


def flatten(X: np.ndarray) -> np.ndarray:
    """(N, seq_length, n_features) → (N, seq_length * n_features)."""
    return X.reshape(X.shape[0], -1).astype(np.float32)


def classical_multistep_forecast(predict_fn: Callable[[np.ndarray], np.ndarray],
                                 test_seed: np.ndarray,
                                 horizons: List[int],
                                 mu_w: float, sigma_w: float,
                                 seq_length: int) -> Dict[int, np.ndarray]:
    """
    Autoregressive multi-step forecast for classical models.

    `predict_fn` accepts a (1, seq_length * n_features) array and returns
    a scalar (or shape-(1,)) prediction for the next windward step.

    Other facades are kept frozen at their last observed value (same convention
    as the DL `multistep_forecast` in train_utils).
    """
    max_h = max(horizons)
    window = test_seed.copy()  # (seq_length, n_features)
    preds_norm: List[float] = []

    for _ in range(max_h):
        x_flat = window[-seq_length:].reshape(1, -1).astype(np.float32)
        out = predict_fn(x_flat)
        val = float(np.asarray(out).squeeze())
        new_step = window[-1].copy()
        new_step[0] = val
        window = np.vstack([window, new_step[np.newaxis]])
        preds_norm.append(val)

    preds = np.array(preds_norm, dtype=np.float32) * sigma_w + mu_w
    return {h: preds[:h].copy() for h in horizons}


def run_classical(
    model_name: str,
    build_estimator: Callable,
    root: Path,
    seq_length: int = 100,
    step: int = 10,
    horizon: int = 1,
    horizons: List[int] = (1, 10, 50, 100, 500),
    scope: str = "Alpha1_4/2_1_3",
):
    """
    Generic train/evaluate/save pipeline for sklearn-style estimators.

    `build_estimator()` must return an unfitted estimator implementing
    .fit(X, y) and .predict(X).
    """
    print(f"\n{'='*60}\n  {model_name.upper()}\n{'='*60}")
    data = get_data(scope, seq_length, step, horizon)

    X_train = flatten(data["X_train"])
    X_val   = flatten(data["X_val"])
    X_test  = flatten(data["X_test"])
    y_train = data["y_train"].squeeze().astype(np.float32)
    y_val   = data["y_val"].squeeze().astype(np.float32)
    y_test  = data["y_test"].squeeze().astype(np.float32)

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    estimator = build_estimator()
    t0 = time.time()
    estimator.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f} s")

    # validation loss for the consolidated CSV (MSE on normalised target)
    val_pred = estimator.predict(X_val)
    best_val = float(np.mean((y_val - val_pred) ** 2))

    # denormalise windward predictions
    mu_w, sigma_w = float(data["mu"][0]), float(data["sigma"][0])
    y_pred_norm = estimator.predict(X_test)
    y_pred = y_pred_norm.squeeze() * sigma_w + mu_w
    y_true = y_test * sigma_w + mu_w
    metrics = compute_metrics(y_true, y_pred)

    print("\n-- Test Metrics --")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Parameter count: best-effort
    if hasattr(estimator, "coef_"):
        n_params = int(estimator.coef_.size + (1 if hasattr(estimator, "intercept_") else 0))
    elif hasattr(estimator, "n_estimators"):
        n_params = int(getattr(estimator, "n_estimators", 0))
    else:
        n_params = 0

    fc = classical_multistep_forecast(
        lambda x: estimator.predict(x),
        data["test_seed"], list(horizons), mu_w, sigma_w, seq_length,
    )

    res = root / "results"
    save_metrics_csv(res / "model_comparison.csv", model_name, metrics,
                     n_params, train_time, epochs=1, best_val=best_val)
    save_pred_vs_actual(res / "plots", model_name, y_true, y_pred, metrics)
    save_forecasts(res / "forecasts", model_name, fc,
                   data["y_future"], mu_w, sigma_w)
    print(f"\nResults saved to {res}")

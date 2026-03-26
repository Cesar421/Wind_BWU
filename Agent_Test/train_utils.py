"""
Shared training, evaluation and visualisation utilities for Agent_Test models.
All per-model training scripts import from this module.
"""

import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Pin random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── DataLoaders ──────────────────────────────────────────────────────────────

def make_loaders(data: dict, batch_size: int = 256
                 ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train / val / test DataLoaders from a numpy data dict."""
    def _ds(X, y):
        return TensorDataset(torch.from_numpy(X).float(),
                             torch.from_numpy(y).float())
    return (
        DataLoader(_ds(data["X_train"], data["y_train"]),
                   batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(_ds(data["X_val"], data["y_val"]),
                   batch_size=batch_size * 2, shuffle=False, num_workers=0),
        DataLoader(_ds(data["X_test"], data["y_test"]),
                   batch_size=batch_size * 2, shuffle=False, num_workers=0),
    )


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """RMSE, MAE, R², MAPE, directional accuracy."""
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mape = float(np.mean(np.abs(err / (np.abs(y_true) + 1e-8))) * 100)
    if len(y_true) > 1:
        da = float(np.mean(np.sign(np.diff(y_true)) ==
                           np.sign(np.diff(y_pred))) * 100)
    else:
        da = float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape,
            "directional_accuracy": da}


# ── Training Loop ────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    ckpt_path: Path,
    max_epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    lr_patience: int = 5,
    lr_factor: float = 0.5,
) -> Tuple[List[float], List[float], float, float]:
    """
    Train with early stopping and ReduceLROnPlateau.

    Returns
    -------
    (train_losses, val_losses, training_time_sec, best_val_loss)
    Best checkpoint is loaded back into *model* before returning.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience)

    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    wait = 0
    train_losses: List[float] = []
    val_losses: List[float] = []

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        # ── train ──
        model.train()
        b_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            b_losses.append(loss.item())
        t_loss = float(np.mean(b_losses))

        # ── validate ──
        model.eval()
        v_batch = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                v_batch.append(criterion(model(xb), yb).item())
        v_loss = float(np.mean(v_batch))

        scheduler.step(v_loss)
        train_losses.append(t_loss)
        val_losses.append(v_loss)

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:>3d} | train={t_loss:.6f}"
                  f" | val={v_loss:.6f} | lr={lr_now:.2e}")

        if v_loss < best_val:
            best_val = v_loss
            wait = 0
            torch.save({"state_dict": model.state_dict(),
                         "epoch": epoch,
                         "best_val_loss": best_val}, str(ckpt_path))
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    train_time = time.time() - t0

    # reload best weights
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    return train_losses, val_losses, train_time, best_val


# ── Test Evaluation ──────────────────────────────────────────────────────────

def evaluate_model(model: nn.Module, test_loader: DataLoader,
                   mu_w: float, sigma_w: float
                   ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Run model on test set, denormalise windward Cp, compute metrics."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
            trues.append(yb.numpy())
    y_pred = np.concatenate(preds).squeeze() * sigma_w + mu_w
    y_true = np.concatenate(trues).squeeze() * sigma_w + mu_w
    return y_true, y_pred, compute_metrics(y_true, y_pred)


# ── Multi-step Forecasting ──────────────────────────────────────────────────

def multistep_forecast(model: nn.Module, test_seed: np.ndarray,
                       horizons: List[int], mu_w: float, sigma_w: float,
                       seq_length: int) -> Dict[int, np.ndarray]:
    """
    Autoregressive multi-step forecast from *test_seed* (normalised).

    Returns {horizon: denormalised_predictions}.
    """
    model.eval()
    max_h = max(horizons)
    window = test_seed.copy()  # (seq_length, n_features)
    preds_norm: List[float] = []

    with torch.no_grad():
        for _ in range(max_h):
            x = torch.from_numpy(
                window[-seq_length:]).float().unsqueeze(0).to(DEVICE)
            out = model(x).cpu().numpy().squeeze()
            val = float(out) if out.ndim == 0 else float(out[0])
            new_step = window[-1].copy()
            new_step[0] = val
            window = np.vstack([window, new_step[np.newaxis]])
            preds_norm.append(val)

    preds = np.array(preds_norm, dtype=np.float32) * sigma_w + mu_w
    return {h: preds[:h].copy() for h in horizons}


# ── CSV I/O ──────────────────────────────────────────────────────────────────

CSV_FIELDS = ["model", "rmse", "mae", "r2", "mape", "directional_accuracy",
              "parameters", "train_time_s", "epochs_run", "best_val_loss"]


def save_metrics_csv(csv_path: Path, model_name: str, metrics: dict,
                     param_count: int, train_time: float,
                     epochs: int, best_val: float):
    """Write one row to the per-model model_comparison.csv (overwrite)."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"model": model_name,
           "parameters": param_count,
           "train_time_s": round(train_time, 1),
           "epochs_run": epochs,
           "best_val_loss": round(best_val, 8),
           **{k: round(v, 6) for k, v in metrics.items()}}
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerow(row)


# ── Plotting ─────────────────────────────────────────────────────────────────

def save_training_curves(plots_dir: Path, tag: str,
                         train_losses: list, val_losses: list):
    """Save train/val loss curves."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label="Train", color="steelblue")
    ax.plot(val_losses, label="Val", color="darkorange")
    ax.set(xlabel="Epoch", ylabel="MSE Loss",
           title=f"{tag} — Training Curves")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{tag}_training_curves.png", dpi=150)
    plt.close(fig)


def save_pred_vs_actual(plots_dir: Path, tag: str,
                        y_true: np.ndarray, y_pred: np.ndarray,
                        metrics: dict, n_show: int = 500):
    """Save prediction-vs-actual overlay and scatter plot."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    n = min(n_show, len(y_true))

    # time-series overlay
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_true[:n], label="Actual", color="royalblue", lw=1.2)
    ax.plot(y_pred[:n], label="Predicted", color="tomato", lw=1.2, ls="--")
    ax.set(xlabel="Index", ylabel="Cp (windward)",
           title=f"{tag} | RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:.4f}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{tag}_pred_vs_actual.png", dpi=150)
    plt.close(fig)

    # scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.scatter(y_true, y_pred, alpha=0.15, s=3, color="steelblue")
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Ideal")
    ax.set(xlabel="Actual Cp", ylabel="Predicted Cp",
           title=f"{tag} — R²={metrics['r2']:.4f}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{tag}_scatter.png", dpi=150)
    plt.close(fig)


def save_forecasts(fc_dir: Path, tag: str,
                   forecasts: Dict[int, np.ndarray],
                   y_future: np.ndarray,
                   mu_w: float, sigma_w: float):
    """Save multi-step forecast .npy files and plot longest horizon."""
    fc_dir = Path(fc_dir)
    fc_dir.mkdir(parents=True, exist_ok=True)
    gt = y_future * sigma_w + mu_w  # denormalise ground truth

    for h, pred in forecasts.items():
        np.save(fc_dir / f"{tag}_h{h}.npy", pred)

    # plot longest horizon vs ground truth
    max_h = max(forecasts)
    pred = forecasts[max_h]
    n = min(len(gt), len(pred))
    if n > 0:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(gt[:n], label="Ground Truth", color="royalblue", lw=1.2)
        ax.plot(pred[:n], label=f"Forecast (h={max_h})",
                color="tomato", lw=1.2, ls="--")
        ax.set(xlabel="Step", ylabel="Cp (windward)",
               title=f"{tag} — Multi-step Forecast (h={max_h})")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fc_dir / f"{tag}_forecast_h{max_h}.png", dpi=150)
        plt.close(fig)

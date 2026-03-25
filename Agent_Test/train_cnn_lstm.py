"""
Training Script — CNN-LSTM on Alpha1_4 / 2_1_3
================================================

Trains the hybrid CNN-LSTM model (MICE 2025) on all wind angles for
building 2_1_3 under terrain roughness Alpha1_4.

Usage
-----
    cd Agent_Test
    python train_cnn_lstm.py

Outputs
-------
    checkpoints/cnn_lstm_alpha1_4_2_1_3.pt   — best model weights
    results/model_comparison.csv             — appended with this run's metrics
    results/plots/cnn_lstm_alpha1_4_2_1_3_*.png
"""

import os
import sys
import time
import csv
from pathlib import Path

# ── make Agent_Test importable ───────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_single_building
from models.cnn_lstm import CNNLSTM

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Hyperparameters ──────────────────────────────────────────────────────────
ALPHA         = "Alpha1_4"
RATIO         = "2_1_3"
SEQ_LENGTH    = 100
STEP          = 10       # sliding-window stride (≥10 avoids OOM)
HORIZON       = 1        # single-step forecast
BATCH_SIZE    = 256
MAX_EPOCHS    = 200
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 15       # early stopping
LR_PATIENCE   = 5        # ReduceLROnPlateau
LR_FACTOR     = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── helpers ──────────────────────────────────────────────────────────────────

def metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute RMSE, MAE, R², MAPE, directional accuracy."""
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    mape = float(np.mean(np.abs(err / (np.abs(y_true) + 1e-8))) * 100)
    # Directional accuracy (only for horizon=1 and step ≥ 2)
    if len(y_true) > 1:
        dir_true = np.sign(np.diff(y_true))
        dir_pred = np.sign(np.diff(y_pred))
        da = float(np.mean(dir_true == dir_pred) * 100)
    else:
        da = float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape,
            "directional_accuracy": da}


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).float()


# ── 1. Load data ─────────────────────────────────────────────────────────────
print(f"\nLoading data: {ALPHA}/{RATIO} ...")
t0 = time.time()
data = load_single_building(
    alpha=ALPHA, ratio=RATIO,
    seq_length=SEQ_LENGTH, step=STEP, horizon=HORIZON,
)
print(f"  X_train: {data['X_train'].shape}  y_train: {data['y_train'].shape}")
print(f"  X_val:   {data['X_val'].shape}    y_val:   {data['y_val'].shape}")
print(f"  X_test:  {data['X_test'].shape}   y_test:  {data['y_test'].shape}")
print(f"  Loaded in {time.time()-t0:.1f}s")

# Create PyTorch datasets
train_ds = TensorDataset(to_tensor(data["X_train"]),
                         to_tensor(data["y_train"]))
val_ds   = TensorDataset(to_tensor(data["X_val"]),
                         to_tensor(data["y_val"]))
test_ds  = TensorDataset(to_tensor(data["X_test"]),
                         to_tensor(data["y_test"]))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                          num_workers=0)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                          num_workers=0)


# ── 2. Build model ───────────────────────────────────────────────────────────
model = CNNLSTM(
    n_features=4,
    seq_length=SEQ_LENGTH,
    horizon=HORIZON,
    cnn_channels=(64, 128, 256),
    lstm_hidden=256,
    lstm_layers=2,
    lstm_dropout=0.2,
    fc_hidden=128,
    dropout=0.3,
).to(DEVICE)

print(f"\nModel: {model.count_parameters():,} trainable parameters")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                             weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE
)


# ── 3. Training loop ─────────────────────────────────────────────────────────
ckpt_dir  = ROOT / "checkpoints"
ckpt_path = ckpt_dir / f"cnn_lstm_{ALPHA.lower()}_{RATIO}.pt"
ckpt_dir.mkdir(exist_ok=True)

best_val_loss = float("inf")
patience_counter = 0
train_losses, val_losses = [], []
train_start = time.time()

print(f"\nTraining for up to {MAX_EPOCHS} epochs ...")
for epoch in range(1, MAX_EPOCHS + 1):
    # ── train ──
    model.train()
    batch_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)                     # (batch, horizon)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_losses.append(loss.item())
    train_loss = float(np.mean(batch_losses))

    # ── validate ──
    model.eval()
    vbatch = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            vbatch.append(criterion(pred, yb).item())
    val_loss = float(np.mean(vbatch))

    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>3} | train_loss={train_loss:.6f}"
              f" | val_loss={val_loss:.6f}"
              f" | lr={optimizer.param_groups[0]['lr']:.2e}")

    # ── early stopping ──
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        model.save(str(ckpt_path), extra={
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "mu": data["mu"].tolist(),
            "sigma": data["sigma"].tolist(),
        })
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

train_time = time.time() - train_start
print(f"\nTraining complete in {train_time:.1f}s  | best val_loss={best_val_loss:.6f}")
print(f"Checkpoint saved: {ckpt_path}")


# ── 4. Test evaluation ───────────────────────────────────────────────────────
# Reload best weights
ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        pred = model(xb).cpu().numpy()
        all_preds.append(pred)
        all_true.append(yb.numpy())

y_pred_norm = np.concatenate(all_preds, axis=0).squeeze()   # (N,)
y_true_norm = np.concatenate(all_true, axis=0).squeeze()    # (N,)

# Denormalise (windward = feature index 0)
mu_w    = data["mu"][0]
sigma_w = data["sigma"][0]
y_pred = y_pred_norm * sigma_w + mu_w
y_true = y_true_norm * sigma_w + mu_w

test_metrics = metrics(y_true, y_pred)
print("\n── Test Metrics ──────────────────────────────────────")
for k, v in test_metrics.items():
    print(f"  {k:<25}: {v:.4f}")


# ── 5. Save results CSV ──────────────────────────────────────────────────────
csv_path = ROOT / "results" / "model_comparison.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)
fieldnames = ["model", "alpha", "ratio", "seq_length", "horizon",
              "rmse", "mae", "r2", "mape", "directional_accuracy",
              "parameters", "train_time_s", "epochs_run", "best_val_loss"]

row = {
    "model": "CNNLSTM",
    "alpha": ALPHA,
    "ratio": RATIO,
    "seq_length": SEQ_LENGTH,
    "horizon": HORIZON,
    "parameters": model.count_parameters(),
    "train_time_s": round(train_time, 1),
    "epochs_run": len(train_losses),
    "best_val_loss": round(best_val_loss, 8),
    **{k: round(v, 6) for k, v in test_metrics.items()},
}

write_header = not csv_path.exists()
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerow(row)
print(f"\nMetrics appended to {csv_path}")


# ── 6. Plots ─────────────────────────────────────────────────────────────────
plots_dir = ROOT / "results" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
run_tag = f"cnn_lstm_{ALPHA.lower()}_{RATIO}"

# 6a. Training curves
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train_losses, label="Train Loss", color="steelblue")
ax.plot(val_losses, label="Val Loss",   color="darkorange")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.set_title(f"CNN-LSTM Training Curves — {ALPHA}/{RATIO}")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(plots_dir / f"{run_tag}_training_curves.png", dpi=150)
plt.close(fig)

# 6b. Predictions vs actual (first 500 test samples)
n_show = min(500, len(y_true))
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(y_true[:n_show], label="Actual Cp (windward)", color="royalblue",
        linewidth=1.2)
ax.plot(y_pred[:n_show], label="Predicted Cp", color="tomato",
        linewidth=1.2, linestyle="--")
ax.set_xlabel("Test Sample Index"); ax.set_ylabel("Cp (windward)")
ax.set_title(f"CNN-LSTM Predictions vs Actual — {ALPHA}/{RATIO}\n"
             f"RMSE={test_metrics['rmse']:.4f}  R²={test_metrics['r2']:.4f}")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(plots_dir / f"{run_tag}_pred_vs_actual.png", dpi=150)
plt.close(fig)

# 6c. Scatter plot
fig, ax = plt.subplots(figsize=(6, 6))
mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
ax.scatter(y_true, y_pred, alpha=0.2, s=5, color="steelblue")
ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Cp"); ax.set_ylabel("Predicted Cp")
ax.set_title(f"CNN-LSTM Scatter — {ALPHA}/{RATIO}\nR²={test_metrics['r2']:.4f}")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(plots_dir / f"{run_tag}_scatter.png", dpi=150)
plt.close(fig)

print(f"Plots saved to {plots_dir}")
print("\nDone!")

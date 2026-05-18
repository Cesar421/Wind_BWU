"""
PatchTST training: channel-mixed PatchTST + direct multi-step head.

Designed to be a 1:1 functional replacement for train_lstm_direct.py so the
results land in the same comparison CSVs.

Resource-aware defaults (conservative mode):
  - torch.cuda.set_per_process_memory_fraction(0.85)  # leave 15% GPU free
  - torch.set_num_threads(cpu_count - 4)              # leave 4 CPU cores free
  - batch=128                                          # half of LSTM_direct, lowers TDR risk

Usage
-----
    python train_patchtst.py --scope Alpha1_4/2_1_3 --horizon 10   # smoke
    python train_patchtst.py --scope round2 --horizon 500
    python train_patchtst.py --scope all    --horizon 500          # R3

Outputs
-------
    checkpoints/patchtst_h{H}.pt
    ../results/patchtst_metrics.csv
    ../results/patchtst_h{H}_rmse_curve.npy
    ../results/patchtst_h{H}_preds.npy
    ../results/patchtst_h{H}_trues.npy
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # Agent_Test/patchtst/
sys.path.append(str(ROOT.parent))               # Agent_Test/
sys.path.append(str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import pandas as pd
import torch

# ─── Resource limits (must run BEFORE any CUDA op) ────────────────────────
def apply_resource_limits(gpu_fraction=0.85, free_cpu_cores=4):
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(gpu_fraction, device=0)
            print(f"GPU memory cap: {gpu_fraction*100:.0f}%")
        except Exception as e:
            print(f"[warn] could not set GPU memory fraction: {e}")
    n_cpu = os.cpu_count() or 8
    n_use = max(1, n_cpu - free_cpu_cores)
    torch.set_num_threads(n_use)
    torch.set_num_interop_threads(max(1, n_use // 2))
    os.environ["OMP_NUM_THREADS"] = str(n_use)
    print(f"CPU threads: {n_use}/{n_cpu}  (left {free_cpu_cores} free)")


apply_resource_limits(gpu_fraction=0.85, free_cpu_cores=4)

from train_utils import (set_seed, DEVICE, make_loaders, train_model,
                         get_data, compute_metrics)
from models.patchtst import PatchTST


SEQ_LENGTH   = 100
STEP         = 10
BATCH_SIZE   = 128            # half of LSTM_direct → less TDR risk on wide head
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 200
PATIENCE     = 15
EVAL_HS      = [1, 10, 50, 100, 500]

# PatchTST architecture
PATCH_LEN = 16
STRIDE    = 8
D_MODEL   = 128
N_HEADS   = 4
D_FF      = 256
N_LAYERS  = 3
DROPOUT   = 0.1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope",   type=str, default="Alpha1_4/2_1_3")
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--batch",   type=int, default=BATCH_SIZE,
                        help="Override batch size (default 128 conservative)")
    args = parser.parse_args()

    H = args.horizon
    set_seed(42)

    print(f"\n{'='*60}\n  PatchTST DIRECT  (scope={args.scope}, H={H})\n{'='*60}")
    print(f"Device: {DEVICE}")

    data = get_data(args.scope, SEQ_LENGTH, STEP, H)
    print(f"X_train: {data['X_train'].shape}  y_train: {data['y_train'].shape}")
    print(f"X_val:   {data['X_val'].shape}  y_val:   {data['y_val'].shape}")
    print(f"X_test:  {data['X_test'].shape}  y_test:  {data['y_test'].shape}")

    train_loader, val_loader, test_loader = make_loaders(data, args.batch)

    model = PatchTST(
        n_features=4, seq_len=SEQ_LENGTH, horizon=H,
        patch_len=PATCH_LEN, stride=STRIDE,
        d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
        n_layers=N_LAYERS, dropout=DROPOUT,
    ).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_par:,}")

    ckpt = ROOT / "checkpoints" / f"patchtst_h{H}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tl, vl, tt, bv = train_model(
        model, train_loader, val_loader, ckpt,
        max_epochs=MAX_EPOCHS, lr=LR,
        weight_decay=WEIGHT_DECAY, patience=PATIENCE,
    )
    print(f"\nTraining time: {tt:.1f} s  ({tt/60:.1f} min)  best_val={bv:.6f}")

    # ── Test inference ────────────────────────────────────────────────
    mu_w, sigma_w = float(data["mu"][0]), float(data["sigma"][0])
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
            trues.append(yb.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    # Reshape (N,) → (N,1) for H=1 to keep indexing uniform
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
        y_true = y_true[:, None]
    y_pred = y_pred * sigma_w + mu_w
    y_true = y_true * sigma_w + mu_w
    print(f"Test shape: {y_pred.shape}")

    # ── Per-step metrics ──────────────────────────────────────────────
    eval_hs = [h for h in EVAL_HS if h <= H]
    rows = []
    print("\n-- Per-step Metrics --")
    for h in eval_hs:
        m = compute_metrics(y_true[:, h-1], y_pred[:, h-1])
        rows.append({"model": f"patchtst_h{H}", "horizon": h, **m,
                     "train_time_s": tt, "scope": args.scope})
        print(f"  h={h:>3d}: rmse={m['rmse']:.4f}  mae={m['mae']:.4f}  r2={m['r2']:.4f}")

    out_csv = ROOT.parent / "results" / "patchtst_metrics.csv"
    df_new = pd.DataFrame(rows)
    if out_csv.exists():
        df_new.to_csv(out_csv, mode="a", header=False, index=False)
    else:
        df_new.to_csv(out_csv, index=False)
    print(f"\nMetrics appended to {out_csv}")

    # ── RMSE curve + raw preds/trues ──────────────────────────────────
    rmse_curve = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    np.save(ROOT.parent / "results" / f"patchtst_h{H}_rmse_curve.npy", rmse_curve)
    np.save(ROOT.parent / "results" / f"patchtst_h{H}_preds.npy", y_pred.astype(np.float32))
    np.save(ROOT.parent / "results" / f"patchtst_h{H}_trues.npy", y_true.astype(np.float32))
    print(f"RMSE curve: first={rmse_curve[0]:.4f}  last={rmse_curve[-1]:.4f}  mean={rmse_curve.mean():.4f}")


if __name__ == "__main__":
    main()

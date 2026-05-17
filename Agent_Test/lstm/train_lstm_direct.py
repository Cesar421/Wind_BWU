"""
LSTM with DIRECT multi-step forecasting head.

Difference vs. train_lstm.py:
  * Trains a single LSTM whose final Linear outputs `HORIZON` values directly.
  * No autoregressive rollout at test time — the H-step trajectory comes from
    one forward pass.
  * Eliminates exposure bias: every training target is the true future, so the
    model never sees its own predictions as inputs.

Usage
-----
    python train_lstm_direct.py --scope round3 --horizon 500

Outputs
-------
    checkpoints/lstm_direct_h{H}.pt
    ../results/lstm_direct_metrics.csv          (per-step metrics row)
    ../results/lstm_direct_h{H}_rmse_curve.npy  (RMSE for every k=1..H)
    ../results/lstm_direct_h{H}_preds.npy       (test predictions, denormalised)
    ../results/lstm_direct_h{H}_trues.npy       (test ground truth, denormalised)
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # Agent_Test/lstm/
sys.path.append(str(ROOT.parent))               # Agent_Test/
sys.path.append(str(ROOT))                      # lstm/

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

from train_utils import (set_seed, DEVICE, make_loaders, train_model,
                         get_data, compute_metrics)
from models.lstm import PureLSTM


SEQ_LENGTH   = 100
STEP         = 10
BATCH_SIZE   = 256
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 200
PATIENCE     = 15
EVAL_HS      = [1, 10, 50, 100, 500]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope",   type=str, default="Alpha1_4/2_1_3")
    parser.add_argument("--horizon", type=int, default=500)
    args = parser.parse_args()

    H = args.horizon
    set_seed(42)

    print(f"\n{'='*60}\n  LSTM DIRECT  (scope={args.scope}, H={H})\n{'='*60}")
    print(f"Device: {DEVICE}")

    data = get_data(args.scope, SEQ_LENGTH, STEP, H)
    print(f"X_train: {data['X_train'].shape}  y_train: {data['y_train'].shape}")
    print(f"X_val:   {data['X_val'].shape}  y_val:   {data['y_val'].shape}")
    print(f"X_test:  {data['X_test'].shape}  y_test:  {data['y_test'].shape}")

    train_loader, val_loader, test_loader = make_loaders(data, BATCH_SIZE)

    model = PureLSTM(
        n_features=4, hidden_size=256, num_layers=2,
        dropout=0.2, fc_hidden=128, fc_dropout=0.3, horizon=H,
    ).to(DEVICE)
    print(f"Parameters: {model.count_parameters():,}")

    ckpt = ROOT / "checkpoints" / f"lstm_direct_h{H}.pt"
    t0 = time.time()
    tl, vl, tt, bv = train_model(
        model, train_loader, val_loader, ckpt,
        max_epochs=MAX_EPOCHS, lr=LR,
        weight_decay=WEIGHT_DECAY, patience=PATIENCE,
    )
    print(f"\nTraining time: {tt:.1f} s  ({tt/60:.1f} min)  best_val={bv:.6f}")

    # ── Test inference ───────────────────────────────────────────────────
    mu_w, sigma_w = float(data["mu"][0]), float(data["sigma"][0])
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
            trues.append(yb.numpy())
    y_pred = np.concatenate(preds) * sigma_w + mu_w   # (N, H)
    y_true = np.concatenate(trues) * sigma_w + mu_w   # (N, H)
    print(f"Test shape: {y_pred.shape}")

    # ── Per-step (k-th forecast horizon) metrics ─────────────────────────
    eval_hs = [h for h in EVAL_HS if h <= H]
    rows = []
    print("\n-- Per-step Metrics --")
    for h in eval_hs:
        m = compute_metrics(y_true[:, h-1], y_pred[:, h-1])
        row = {"model": f"lstm_direct_h{H}", "horizon": h, **m,
                "train_time_s": tt, "scope": args.scope}
        rows.append(row)
        print(f"  h={h:>3d}: rmse={m['rmse']:.4f}  mae={m['mae']:.4f}  r2={m['r2']:.4f}")

    out_csv = ROOT.parent / "results" / "lstm_direct_metrics.csv"
    df_new = pd.DataFrame(rows)
    if out_csv.exists():
        df_new.to_csv(out_csv, mode="a", header=False, index=False)
    else:
        df_new.to_csv(out_csv, index=False)
    print(f"\nMetrics appended to {out_csv}")

    # ── Full per-step RMSE curve and raw preds ───────────────────────────
    rmse_curve = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    np.save(ROOT.parent / "results" / f"lstm_direct_h{H}_rmse_curve.npy", rmse_curve)
    np.save(ROOT.parent / "results" / f"lstm_direct_h{H}_preds.npy", y_pred.astype(np.float32))
    np.save(ROOT.parent / "results" / f"lstm_direct_h{H}_trues.npy", y_true.astype(np.float32))
    print(f"RMSE curve length: {len(rmse_curve)}  "
          f"(first step {rmse_curve[0]:.4f}, last step {rmse_curve[-1]:.4f})")


if __name__ == "__main__":
    main()

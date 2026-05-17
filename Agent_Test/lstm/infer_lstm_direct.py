"""
Inference-only: load `lstm_direct_h{H}.pt` and compute test metrics.

Used after a crash interrupted training but a usable checkpoint survived.
Loads data, instantiates the model with horizon=H, restores the state_dict,
runs forward over the test set, and saves the same artefacts as train_lstm_direct.py.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))
sys.path.append(str(ROOT))

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

from train_utils import set_seed, DEVICE, get_data, compute_metrics
from models.lstm import PureLSTM


SEQ_LENGTH = 100
STEP = 10
EVAL_HS = [1, 10, 50, 100, 500]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scope",   type=str, default="all")
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--ckpt",    type=str, default="")
    args = p.parse_args()
    H = args.horizon
    set_seed(42)

    ckpt_path = Path(args.ckpt) if args.ckpt else ROOT / "checkpoints" / f"lstm_direct_h{H}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device:     {DEVICE}")

    data = get_data(args.scope, SEQ_LENGTH, STEP, H)
    print(f"X_test: {data['X_test'].shape}  y_test: {data['y_test'].shape}")

    model = PureLSTM(
        n_features=4, hidden_size=256, num_layers=2,
        dropout=0.2, fc_hidden=128, fc_dropout=0.3, horizon=H,
    ).to(DEVICE)
    ck = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    print(f"Loaded epoch={ck.get('epoch','?')}  best_val={ck.get('best_val_loss','?')}")

    model.eval()
    mu_w, sigma_w = float(data["mu"][0]), float(data["sigma"][0])

    # Stream the test set in batches to avoid OOM
    BATCH = 256
    X_test = torch.from_numpy(data["X_test"]).float()
    y_test = data["y_test"]
    n = X_test.shape[0]
    preds_chunks = []
    with torch.no_grad():
        for i in range(0, n, BATCH):
            xb = X_test[i:i+BATCH].to(DEVICE)
            preds_chunks.append(model(xb).cpu().numpy())
    y_pred = np.concatenate(preds_chunks) * sigma_w + mu_w
    y_true = y_test * sigma_w + mu_w
    print(f"y_pred: {y_pred.shape}")

    eval_hs = [h for h in EVAL_HS if h <= H]
    rows = []
    print("\n-- Per-step Metrics --")
    for h in eval_hs:
        m = compute_metrics(y_true[:, h-1], y_pred[:, h-1])
        rows.append({"model": f"lstm_direct_h{H}", "horizon": h, **m,
                     "train_time_s": float(ck.get("best_val_loss", float("nan"))),
                     "scope": args.scope, "source": "inference_only"})
        print(f"  h={h:>3d}: rmse={m['rmse']:.4f}  mae={m['mae']:.4f}  r2={m['r2']:.4f}")

    out_csv = ROOT.parent / "results" / "lstm_direct_metrics.csv"
    df = pd.DataFrame(rows)
    if out_csv.exists():
        df.to_csv(out_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(out_csv, index=False)
    print(f"\nAppended to {out_csv}")

    rmse_curve = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    np.save(ROOT.parent / "results" / f"lstm_direct_h{H}_rmse_curve.npy", rmse_curve)
    np.save(ROOT.parent / "results" / f"lstm_direct_h{H}_preds.npy", y_pred.astype(np.float32))
    np.save(ROOT.parent / "results" / f"lstm_direct_h{H}_trues.npy", y_true.astype(np.float32))
    print(f"RMSE curve: first={rmse_curve[0]:.4f}  last={rmse_curve[-1]:.4f}  mean={rmse_curve.mean():.4f}")


if __name__ == "__main__":
    main()

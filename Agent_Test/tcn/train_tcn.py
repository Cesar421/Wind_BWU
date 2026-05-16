"""
Training Script — TCN (Kareem JWEIA 2024)
==========================================
Trains on ALL buildings from Alpha1_4 and Alpha1_6.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent       # Agent_Test/tcn/
sys.path.insert(0, str(ROOT.parent))         # Agent_Test/
sys.path.insert(0, str(ROOT))                # tcn/

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from train_utils import (set_seed, DEVICE, make_loaders, train_model,
                         evaluate_model, multistep_forecast,
                         save_metrics_csv, save_training_curves,
                         save_pred_vs_actual, save_forecasts, get_data)
from models.tcn import TCN

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "tcn"
SEQ_LENGTH  = 100
STEP        = 10
HORIZON     = 1
BATCH_SIZE  = 256
LR          = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS  = 200
PATIENCE    = 15
HORIZONS    = [1, 10, 50, 100, 500]

set_seed(42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", type=str, default="Alpha1_4/2_1_3",
                        help='"Alpha1_4/2_1_3" (Round 1) | "round2" | "all"')
    args = parser.parse_args()

    print(f"\n{'='*60}\n  {MODEL_NAME.upper()}\n{'='*60}")
    print(f"Device: {DEVICE}")

    data = get_data(args.scope, SEQ_LENGTH, STEP, HORIZON)
    train_loader, val_loader, test_loader = make_loaders(data, BATCH_SIZE)

    model = TCN(
        n_features=4, num_channels=(64, 128, 256),
        kernel_size=3, dropout=0.2, fc_hidden=128, horizon=HORIZON,
    ).to(DEVICE)
    print(f"Parameters: {model.count_parameters():,}")

    ckpt = ROOT / "checkpoints" / f"{MODEL_NAME}_best.pt"
    tl, vl, tt, bv = train_model(
        model, train_loader, val_loader, ckpt,
        max_epochs=MAX_EPOCHS, lr=LR,
        weight_decay=WEIGHT_DECAY, patience=PATIENCE)

    mu_w, sigma_w = float(data["mu"][0]), float(data["sigma"][0])
    yt, yp, tm = evaluate_model(model, test_loader, mu_w, sigma_w)
    print("\n-- Test Metrics --")
    for k, v in tm.items():
        print(f"  {k}: {v:.4f}")

    fc = multistep_forecast(model, data["test_seed"], HORIZONS,
                            mu_w, sigma_w, SEQ_LENGTH)

    res = ROOT / "results"
    save_metrics_csv(res / "model_comparison.csv", MODEL_NAME, tm,
                     model.count_parameters(), tt, len(tl), bv)
    save_training_curves(res / "plots", MODEL_NAME, tl, vl)
    save_pred_vs_actual(res / "plots", MODEL_NAME, yt, yp, tm)
    save_forecasts(res / "forecasts", MODEL_NAME, fc,
                   data["y_future"], mu_w, sigma_w)

    print(f"\nResults saved to {res}")


if __name__ == "__main__":
    main()

"""
Dense naive-persistence baseline over the FULL windowed test set (exact P2).

Matches the evaluation regime of the direct multi-step models (LSTM-direct /
PatchTST): for every step-10 window in the test portion of every series,
predict the windward Cp 1..H steps ahead as the last observed windward value,
and score per-step@h over all ~147 k windows.

Streaming (per-series accumulation) so the 1.2 GB train window array is never
built. Writes results/naive_dense_metrics_round3.csv.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from data_loader import (POSTPROCESS_DIR, available_angles, load_faces,  # noqa: E402
                         zscore_fit, zscore_transform, build_sequences)

SEQ = 100
H = 500
STEP = 10
EVAL_HS = [1, 10, 50, 100, 500]
TRAIN_R, VAL_R = 0.70, 0.15


def main():
    alphas = ["Alpha1_4", "Alpha1_6"]

    # ── Pass 0: collect train portions for mu/sigma (identical to training) ──
    print("Fitting normalisation on train portions ...")
    train_parts = []
    test_segs = []
    for alpha in alphas:
        ad = POSTPROCESS_DIR / alpha
        if not ad.exists():
            continue
        for rd in sorted(ad.iterdir()):
            if not (rd / "Data").exists():
                continue
            for ang in available_angles(alpha, rd.name):
                try:
                    ts = load_faces(alpha, rd.name, ang)
                except FileNotFoundError:
                    continue
                T = len(ts)
                t1 = int(T * TRAIN_R)
                t2 = int(T * (TRAIN_R + VAL_R))
                train_parts.append(ts[:t1])
                test_segs.append(ts[t2:])
    mu, sigma = zscore_fit(np.concatenate(train_parts, axis=0))
    del train_parts
    mu_w, sigma_w = float(mu[0]), float(sigma[0])

    # ── Streaming accumulators per horizon (denormalised windward) ──
    n = np.zeros(len(EVAL_HS), dtype=np.float64)
    sum_err2 = np.zeros(len(EVAL_HS), dtype=np.float64)
    sum_y = np.zeros(len(EVAL_HS), dtype=np.float64)
    sum_y2 = np.zeros(len(EVAL_HS), dtype=np.float64)

    print(f"Scoring dense naive over {len(test_segs)} test segments "
          f"(step={STEP}, H={H}) ...")
    n_windows = 0
    for seg in test_segs:
        if len(seg) < SEQ + H:
            continue
        seg_n = zscore_transform(seg, mu, sigma)
        X, y = build_sequences(seg_n, SEQ, STEP, H)        # X:(N,100,4) y:(N,500)
        if len(X) == 0:
            continue
        n_windows += len(X)
        last_w = X[:, -1, 0]                                # naive pred (norm)
        y_d = y * sigma_w + mu_w                            # (N,500) denorm true
        pred_d = last_w * sigma_w + mu_w                    # (N,) denorm pred
        for i, h in enumerate(EVAL_HS):
            yt = y_d[:, h - 1]
            err = yt - pred_d
            n[i] += len(yt)
            sum_err2[i] += float(np.sum(err ** 2))
            sum_y[i] += float(np.sum(yt))
            sum_y2[i] += float(np.sum(yt ** 2))
        del X, y, y_d

    rows = []
    for i, h in enumerate(EVAL_HS):
        rmse = float(np.sqrt(sum_err2[i] / n[i]))
        ss_res = sum_err2[i]
        ss_tot = sum_y2[i] - (sum_y[i] ** 2) / n[i]
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rows.append({"model": "naive_dense", "horizon": h,
                     "n_windows": int(n[i]), "rmse": rmse, "r2": r2})

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "naive_dense_metrics_round3.csv"
    df.to_csv(out, index=False)
    print(f"\nTotal windows: {n_windows}")
    print(df.round(4).to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

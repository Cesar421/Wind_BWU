"""
Re-fit XGBoost for the multi-trajectory long-horizon table using the SAME
hyperparameters as the rest of the campaign.

Why
---
``evaluate_long_horizon.py`` re-fits XGBoost inline with a *different*
configuration than the one used for the Round 1-3 single-step benchmark:

    xgboost/models/xgboost_ts.py (campaign)  lr=0.05  subsample=0.9  colsample=0.9
    evaluate_long_horizon.py     (refit)     lr=0.10  subsample=0.8  colsample=0.8  max_bin=128

The thesis states (Methodology, "Unified Experimental Protocol") that the
optimiser/hyperparameters are immutable across models, and footnotes only the
window-stride change -- so the long-horizon XGBoost row was not strictly
protocol-compliant. This script closes that gap by re-fitting XGBoost with the
campaign's own ``build()`` (single source of truth) and rewriting only the
XGBoost rows of the two results CSVs. Every other model's row is left untouched.

Integrity check
---------------
``build_test_trajectories`` is fully deterministic (evenly spaced starts, no
RNG), so re-running it must reproduce the stored naive baseline bit-for-bit.
The script recomputes naive and aborts if it does not match the stored CSV --
proving the surgical row update is comparing like with like.

Usage (from Agent_Test/):
    python refit_xgboost_matched.py --classical-step 40
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data_loader import build_test_trajectories                      # noqa: E402
from evaluate_long_horizon import (                                  # noqa: E402
    EVAL_HS, H, SEQ, classical_step_fn, hac_dm, naive_predict,
    per_step_metrics, rollout,
)

MULTITRAJ_CSV = ROOT / "results" / "long_horizon_multitraj_round3.csv"
SIG_CSV = ROOT / "results" / "long_horizon_significance_round3.csv"
NAIVE_TOL = 1e-6


def load_campaign_xgb():
    """Load xgboost/models/xgboost_ts.py by path (avoids the local-folder name
    clash) and return its build() estimator -- the campaign's own config."""
    spec = importlib.util.spec_from_file_location(
        "campaign_xgboost_ts", ROOT / "xgboost" / "models" / "xgboost_ts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-per-series", type=int, default=2)
    ap.add_argument("--classical-step", type=int, default=40,
                    help="window stride for the re-fit train set (must match "
                         "the stride used for the stored classical rows)")
    args = ap.parse_args()

    t_start = time.time()

    # ── 1. Rebuild the exact same test trajectories ────────────────────────
    print(f"Building test trajectories (traj_per_series={args.traj_per_series}) ...")
    tr = build_test_trajectories(seq_length=SEQ, H=H,
                                 traj_per_series=args.traj_per_series)
    seeds, futures = tr["seeds"], tr["futures"]
    mu_w, sigma_w = float(tr["mu"][0]), float(tr["sigma"][0])
    N = seeds.shape[0]
    true_d = futures * sigma_w + mu_w
    print(f"  {N} trajectories from {tr['n_series']} series")

    # ── 2. Integrity check: naive must reproduce the stored row exactly ────
    naive_d = naive_predict(seeds, H) * sigma_w + mu_w
    old = pd.read_csv(MULTITRAJ_CSV)
    print("\nIntegrity check -- recomputed naive vs stored naive:")
    ok = True
    for h in EVAL_HS:
        got = per_step_metrics(true_d[:, h - 1], naive_d[:, h - 1])["rmse"]
        ref = float(old[(old.model == "naive") & (old.horizon == h)].rmse.iloc[0])
        delta = abs(got - ref)
        flag = "OK" if delta < NAIVE_TOL else "MISMATCH"
        if delta >= NAIVE_TOL:
            ok = False
        print(f"  h={h:<4d} recomputed={got:.6f}  stored={ref:.6f}  d={delta:.2e}  {flag}")
    if not ok:
        raise SystemExit("Trajectories differ from the stored run -- aborting: a "
                         "surgical row update would not be comparable.")
    print("  -> trajectories identical; surgical update is valid.\n")

    # ── 3. Load training windows at the same stride as the stored refit ────
    from train_utils import get_data
    from classical_utils import flatten
    print(f"Loading R3 train windows (step={args.classical_step}) ...")
    data = get_data("all", SEQ, args.classical_step, 1)
    Xtr = flatten(data["X_train"])
    ytr = data["y_train"].squeeze().astype(np.float32)
    del data
    gc.collect()
    print(f"  Xtr {Xtr.shape}  ({Xtr.nbytes / 1e9:.2f} GB)")

    # ── 4. Fit with the CAMPAIGN hyperparameters ───────────────────────────
    est = load_campaign_xgb()
    p = est.get_params()
    print("\nCampaign XGBoost config (from xgboost/models/xgboost_ts.py):")
    for k in ("n_estimators", "max_depth", "learning_rate", "subsample",
              "colsample_bytree", "tree_method", "device", "random_state"):
        print(f"    {k} = {p.get(k)}")
    t0 = time.time()
    est.fit(Xtr, ytr)
    fit_time = time.time() - t0
    print(f"  fit time: {fit_time:.1f} s")
    del Xtr, ytr
    gc.collect()

    # ── 5. Roll out ────────────────────────────────────────────────────────
    print(f"Rolling out XGBoost over {N} trajectories x {H} steps ...")
    t0 = time.time()
    xgb_d = rollout(classical_step_fn(est), seeds, H, chunk=4096) * sigma_w + mu_w
    print(f"  rollout time: {time.time() - t0:.1f} s")

    # ── 6. Per-horizon metrics ─────────────────────────────────────────────
    new_rows = []
    print("\nXGBoost per-horizon (matched hyperparameters):")
    print(f"  {'h':>5} {'RMSE new':>10} {'RMSE old':>10} {'R2 new':>9} {'R2 old':>9}")
    for h in EVAL_HS:
        m = per_step_metrics(true_d[:, h - 1], xgb_d[:, h - 1])
        o = old[(old.model == "xgboost") & (old.horizon == h)]
        o_rmse = float(o.rmse.iloc[0]) if len(o) else float("nan")
        o_r2 = float(o.r2.iloc[0]) if len(o) else float("nan")
        print(f"  {h:>5} {m['rmse']:>10.4f} {o_rmse:>10.4f} "
              f"{m['r2']:>9.4f} {o_r2:>9.4f}")
        new_rows.append({"model": "xgboost", "horizon": h, "n_traj": N, **m})

    # ── 7. Significance vs naive ───────────────────────────────────────────
    rmse_traj_x = np.sqrt(np.mean((true_d - xgb_d) ** 2, axis=1))
    rmse_traj_n = np.sqrt(np.mean((true_d - naive_d) ** 2, axis=1))
    try:
        _, w_p = wilcoxon(rmse_traj_n, rmse_traj_x)
    except ValueError:
        w_p = float("nan")
    win_rate = float(np.mean(rmse_traj_x < rmse_traj_n))
    d = np.mean((true_d - naive_d) ** 2 - (true_d - xgb_d) ** 2, axis=0)
    dm, dm_p = hac_dm(d)
    sig_new = {
        "model_vs_naive": "xgboost",
        "median_rmse_model": float(np.median(rmse_traj_x)),
        "median_rmse_naive": float(np.median(rmse_traj_n)),
        "win_rate_vs_naive": win_rate,
        "wilcoxon_p": float(w_p),
        "dm_hac": dm, "dm_hac_p": dm_p, "n_traj": N,
    }
    print("\nSignificance vs naive (matched hyperparameters):")
    for k, v in sig_new.items():
        print(f"    {k} = {v}")

    # ── 8. Surgical CSV updates (xgboost rows only) ────────────────────────
    upd = old[old.model != "xgboost"].copy()
    upd = pd.concat([upd, pd.DataFrame(new_rows)], ignore_index=True)
    upd.to_csv(MULTITRAJ_CSV, index=False)
    print(f"\nUpdated {MULTITRAJ_CSV} (xgboost rows replaced)")

    sig_old = pd.read_csv(SIG_CSV)
    sig_upd = sig_old[sig_old.model_vs_naive != "xgboost"].copy()
    sig_upd = pd.concat([sig_upd, pd.DataFrame([sig_new])], ignore_index=True)
    sig_upd.to_csv(SIG_CSV, index=False)
    print(f"Updated {SIG_CSV} (xgboost row replaced)")
    print(f"\nTotal time: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()

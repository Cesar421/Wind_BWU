"""
Multi-trajectory long-horizon evaluation  (fix for P1 + P2)
===========================================================
The legacy ``analyze_horizons.py`` / ``diebold_mariano.py`` evaluated every
autoregressive model on a SINGLE 500-step trajectory (the first 500 steps of
the last test building-angle series). The headline h=500 numbers and the whole
Diebold-Mariano table therefore came from one realisation, and the naive R^2 at
h=500 swung from -5.4 (R1) to -0.07 (R3) purely because the "last series"
changed.

This script fixes that:

  * P1 — rolls every model out autoregressively from MANY seeds sampled across
    the test portion of every series, and aggregates per-horizon metrics over
    all trajectories (a generalisation estimate, not one draw). It adds a
    proper cross-trajectory significance test (paired Wilcoxon on per-trajectory
    RMSE + win-rate vs naive) on top of the HAC Diebold-Mariano.

  * P2 — every model (incl. naive) is scored AT step h (per-step convention),
    the SAME convention the direct models (LSTM-direct / PatchTST) use, so
    naive and the direct models are finally compared apples-to-apples.

Deep models (LSTM/GRU/TCN) are rolled out from their existing single-step
checkpoints — no retraining. Classical baselines (Ridge/XGBoost/RF) need a
re-fit (they were never pickled); enable with --classical (RF with --rf).

Outputs (results/):
  long_horizon_multitraj_round3.csv      per-horizon RMSE/MAE/R2 (per-step@h)
  long_horizon_significance_round3.csv    naive-vs-model paired tests @h=500
  plots_cross_round/rmse_vs_horizon_multitraj.png
  plots_cross_round/rmse_h500_boxplot_multitraj.png
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, norm

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from data_loader import build_test_trajectories  # noqa: E402

EVAL_HS = [1, 10, 50, 100, 500]
SEQ = 100
H = 500

DL_SPECS = {
    "lstm": ("lstm.py", "PureLSTM",
             dict(n_features=4, hidden_size=256, num_layers=2, dropout=0.2,
                  fc_hidden=128, fc_dropout=0.3, horizon=1),
             "checkpoints/lstm_best.pt"),
    "gru":  ("gru.py", "PureGRU",
             dict(n_features=4, hidden_size=256, num_layers=2, dropout=0.2,
                  fc_hidden=128, fc_dropout=0.3, horizon=1),
             "checkpoints/gru_best.pt"),
    "tcn":  ("tcn.py", "TCN",
             dict(n_features=4, num_channels=(64, 128, 256), kernel_size=3,
                  dropout=0.2, fc_hidden=128, horizon=1),
             "checkpoints/tcn_best.pt"),
}


# ── model loading ─────────────────────────────────────────────────────────────

def _load_class(folder: str, filename: str, classname: str):
    spec = importlib.util.spec_from_file_location(
        f"{folder}_model", ROOT / folder / "models" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, classname)


def load_dl_model(name: str):
    filename, classname, kwargs, ckpt_rel = DL_SPECS[name]
    ckpt = ROOT / name / ckpt_rel
    if not ckpt.exists():
        print(f"  [skip] {name}: checkpoint missing ({ckpt})")
        return None
    cls = _load_class(name, filename, classname)
    model = cls(**kwargs)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


# ── batched autoregressive rollout ───────────────────────────────────────────

def rollout(step_fn, seeds: np.ndarray, H: int, chunk: int = 2048) -> np.ndarray:
    """
    Batched autoregressive rollout. ``step_fn(window)`` maps a (B, seq, 4) array
    to the (B,) next normalised windward prediction. The 3 non-windward faces are
    frozen at their last observed value (same convention as the legacy rollout).
    Returns (N, H) normalised windward predictions.
    """
    N = seeds.shape[0]
    out = np.empty((N, H), dtype=np.float32)
    for i in range(0, N, chunk):
        win = seeds[i:i + chunk].copy()                 # (B, seq, 4)
        B = win.shape[0]
        for t in range(H):
            pred = step_fn(win).astype(np.float32)      # (B,)
            out[i:i + B, t] = pred
            new = win[:, -1, :].copy()                  # freeze other faces
            new[:, 0] = pred
            win = np.concatenate([win[:, 1:, :], new[:, None, :]], axis=1)
    return out


def dl_step_fn(model):
    def _f(win: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(win).float()
            y = model(x).cpu().numpy()
        return y.reshape(y.shape[0], -1)[:, 0]
    return _f


def classical_step_fn(estimator):
    def _f(win: np.ndarray) -> np.ndarray:
        flat = win.reshape(win.shape[0], -1).astype(np.float32)
        return np.asarray(estimator.predict(flat)).reshape(-1)
    return _f


def naive_predict(seeds: np.ndarray, H: int) -> np.ndarray:
    last = seeds[:, -1, 0]                                # (N,)
    return np.repeat(last[:, None], H, axis=1)


# ── metrics ───────────────────────────────────────────────────────────────────

def per_step_metrics(true_h: np.ndarray, pred_h: np.ndarray) -> dict:
    """RMSE/MAE/R2 over all trajectories AT one horizon step (1-D arrays)."""
    err = true_h - pred_h
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true_h - true_h.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def hac_dm(d: np.ndarray) -> tuple[float, float]:
    """Diebold-Mariano on a per-step loss differential series (Newey-West HAC)."""
    n = len(d)
    lag = max(1, int(np.floor(n ** (1 / 3))))
    dc = d - d.mean()
    s = float(np.mean(dc ** 2))
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)
        s += 2.0 * w * float(np.mean(dc[k:] * dc[:-k]))
    if s <= 0:
        return float("nan"), float("nan")
    dm = float(d.mean() / np.sqrt(s / n))
    p = float(2.0 * (1.0 - norm.cdf(abs(dm))))
    return dm, p


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-per-series", type=int, default=2)
    ap.add_argument("--classical", action="store_true",
                    help="re-fit Ridge + XGBoost and include them (heavy load)")
    ap.add_argument("--rf", action="store_true",
                    help="also re-fit RandomForest (~2 h on CPU)")
    args = ap.parse_args()

    print(f"Building test trajectories (traj_per_series={args.traj_per_series}) ...")
    tr = build_test_trajectories(seq_length=SEQ, H=H,
                                 traj_per_series=args.traj_per_series)
    seeds, futures = tr["seeds"], tr["futures"]
    mu_w, sigma_w = float(tr["mu"][0]), float(tr["sigma"][0])
    N = seeds.shape[0]
    print(f"  {N} trajectories from {tr['n_series']} series  "
          f"(seeds {seeds.shape}, futures {futures.shape})")
    true_d = futures * sigma_w + mu_w                     # (N, H) denormalised

    # ── collect predictions per model ──
    preds_d: dict[str, np.ndarray] = {}
    preds_d["naive"] = naive_predict(seeds, H) * sigma_w + mu_w

    for name in DL_SPECS:
        model = load_dl_model(name)
        if model is None:
            continue
        print(f"  rolling out {name} over {N} trajectories x {H} steps ...")
        preds_d[name] = rollout(dl_step_fn(model), seeds, H) * sigma_w + mu_w

    if args.classical:
        from train_utils import get_data
        from classical_utils import flatten
        print("  loading R3 windows for classical re-fit (heavy) ...")
        data = get_data("all", SEQ, 10, 1)
        Xtr = flatten(data["X_train"]); ytr = data["y_train"].squeeze().astype(np.float32)
        from ridge.models.ridge import build as build_ridge
        import importlib
        classical = {"ridge": build_ridge()}
        xgb_mod = importlib.import_module("xgboost")
        classical["xgboost"] = xgb_mod.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=-1)
        if args.rf:
            from random_forest.models.random_forest import build as build_rf
            classical["random_forest"] = build_rf()
        for cname, est in classical.items():
            print(f"  fitting {cname} ...")
            est.fit(Xtr, ytr)
            print(f"  rolling out {cname} ...")
            preds_d[cname] = rollout(classical_step_fn(est), seeds, H,
                                     chunk=4096) * sigma_w + mu_w

    # ── per-horizon table (per-step @ h, aggregated over all trajectories) ──
    rows = []
    for name, pred in preds_d.items():
        for h in EVAL_HS:
            m = per_step_metrics(true_d[:, h - 1], pred[:, h - 1])
            rows.append({"model": name, "horizon": h, "n_traj": N, **m})
    df = pd.DataFrame(rows)
    out_csv = ROOT / "results" / "long_horizon_multitraj_round3.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    print(df.pivot(index="model", columns="horizon", values="rmse").round(4))

    # ── cross-trajectory significance vs naive (per-trajectory RMSE @ full 500) ──
    # per-trajectory RMSE over the whole 500-step rollout
    rmse_traj = {k: np.sqrt(np.mean((true_d - v) ** 2, axis=1))
                 for k, v in preds_d.items()}
    sig_rows = []
    base = rmse_traj["naive"]
    for name, rt in rmse_traj.items():
        if name == "naive":
            continue
        # paired Wilcoxon across trajectories (robust, no normality assumption)
        try:
            w_stat, w_p = wilcoxon(base, rt)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        win_rate = float(np.mean(rt < base))   # fraction of traj where model beats naive
        # HAC-DM on the trajectory-averaged per-step squared-error differential
        d = np.mean((true_d - preds_d["naive"]) ** 2 - (true_d - preds_d[name]) ** 2, axis=0)
        dm, dm_p = hac_dm(d)
        sig_rows.append({
            "model_vs_naive": name,
            "median_rmse_model": float(np.median(rt)),
            "median_rmse_naive": float(np.median(base)),
            "win_rate_vs_naive": win_rate,
            "wilcoxon_p": float(w_p),
            "dm_hac": dm, "dm_hac_p": dm_p,
            "n_traj": N,
        })
    sdf = pd.DataFrame(sig_rows)
    sig_csv = ROOT / "results" / "long_horizon_significance_round3.csv"
    sdf.to_csv(sig_csv, index=False)
    print(f"\nSaved {sig_csv}")
    print(sdf.round(4).to_string(index=False))

    # ── plots ──
    out_dir = ROOT / "results" / "plots_cross_round"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    for name in preds_d:
        ys = [df[(df.model == name) & (df.horizon == h)]["rmse"].iloc[0] for h in EVAL_HS]
        ls = "--" if name == "naive" else "-"
        lw = 2.6 if name == "naive" else 1.8
        ax.plot(EVAL_HS, ys, marker="o", linestyle=ls, linewidth=lw, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Forecast horizon (steps)")
    ax.set_ylabel("RMSE at step h  (mean over trajectories)")
    ax.set_title(f"Multi-trajectory autoregressive RMSE vs horizon "
                 f"(R3, N={N} trajectories)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "rmse_vs_horizon_multitraj.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    names = list(rmse_traj.keys())
    ax.boxplot([rmse_traj[n] for n in names], labels=names, showfliers=False)
    ax.set_ylabel("Per-trajectory RMSE over 500 steps")
    ax.set_title(f"Distribution of long-horizon RMSE across {N} trajectories (R3)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "rmse_h500_boxplot_multitraj.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved plots to {out_dir}")


if __name__ == "__main__":
    main()

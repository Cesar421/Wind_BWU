"""Train WPTSE-Net on BDH face-averaged Cp time series (round2 scope).

Usage (PowerShell):
    conda activate ML_Cesar; $env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:PYTHONIOENCODING="utf-8"
    cd C:\\Users\\verwalter\\Documents\\GitHub\\Wind_BWU\\Agent_Papers\\wptse_net
    python -u train_wptse_net.py --scope round2
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import signal as sp_signal

# ── Import shared Agent_Test data loader (read-only) ────────────────────────
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Agent_Test"))
from data_loader import (                                  # noqa: E402
    available_angles,
    load_faces,
)

from models.wptse_net import WPTSENet, encode_slice         # noqa: E402

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Conservative resource limits (per task instructions) ────────────────────
N_THREADS = max(1, (os.cpu_count() or 8) - 4)
torch.set_num_threads(N_THREADS)
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.85, 0)

# ── Round 2 scope (mirrors Agent_Test/train_utils.py) ───────────────────────
ROUND2_RATIOS = ["1_1_3", "2_1_3", "3_1_3"]
ROUND2_ALPHAS = ["Alpha1_4", "Alpha1_6"]
FACE_NAMES = ["windward", "leeward", "sideleft", "sideright"]

# ── Hyperparameters (paper-faithful where possible) ─────────────────────────
SLICE_LENGTH = 32           # paper: each output slice contains 32 samples
NOISE_SCALE = 0.1           # paper: 0.1
HIDDEN = 128                # paper: 128
N_BLOCKS = 4                # paper: ×4
LATENT_DIM = 5              # 4 stats + noise
BATCH_SIZE = 128            # task spec (paper used 64 on a single 4096-sample slice)
LR = 1e-3                   # task spec
EPOCHS = 1000               # task spec
PATIENCE = 50               # task spec early stopping
HUBER_DELTA = 1.0           # paper

# ── PSD parameters (task spec) ──────────────────────────────────────────────
PSD_FS = 1000
PSD_NPERSEG = 256
PSD_NOVERLAP = 128


# ────────────────────────────────────────────────────────────────────────────
def gather_series(scope: str):
    """Return list of (key, raw_series_1d_float32) tuples."""
    base = PROJECT_ROOT / "Data" / "Data_All_The_BDH_PostProcess"
    if scope == "round2":
        alphas, ratios = ROUND2_ALPHAS, ROUND2_RATIOS
    elif scope == "all":
        # Discover every (alpha, ratio) folder that has a Data subdir.
        alphas, ratios = [], set()
        for alpha_dir in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("Alpha")):
            alphas.append(alpha_dir.name)
            for r in sorted(p for p in alpha_dir.iterdir() if p.is_dir() and (p / "Data").exists()):
                ratios.add(r.name)
        ratios = sorted(ratios)
    else:
        raise ValueError(f"Unsupported scope: {scope}")
    series = []
    for alpha in alphas:
        for ratio in ratios:
            if not (base / alpha / ratio / "Data").exists():
                continue
            for angle in available_angles(alpha, ratio, base):
                try:
                    arr = load_faces(alpha, ratio, angle, base)   # (32768, 4)
                except FileNotFoundError:
                    continue
                for f_idx, f_name in enumerate(FACE_NAMES):
                    key = f"{alpha}/{ratio}/a{angle:03d}/{f_name}"
                    series.append((key, arr[:, f_idx].astype(np.float32)))
    return series


def split_and_slice(ts: np.ndarray, slice_length: int = SLICE_LENGTH,
                    train_ratio: float = 0.70, val_ratio: float = 0.15):
    """Chronological 70/15/15 split, then cut each part into non-overlapping slices."""
    T = len(ts)
    t1 = int(T * train_ratio)
    t2 = int(T * (train_ratio + val_ratio))
    parts = {"train": ts[:t1], "val": ts[t1:t2], "test": ts[t2:]}
    out = {}
    for k, seg in parts.items():
        n = (len(seg) // slice_length) * slice_length
        if n == 0:
            out[k] = np.zeros((0, slice_length), dtype=np.float32)
            continue
        out[k] = seg[:n].reshape(-1, slice_length).astype(np.float32)
    return out


# ────────────────────────────────────────────────────────────────────────────
def build_dataset(scope: str):
    print(f"[data] Gathering series for scope='{scope}' ...")
    series_list = gather_series(scope)
    print(f"[data] Loaded {len(series_list)} series (alpha,ratio,angle,face).")

    # Normalisation: per-series mean/std fitted on TRAIN portion only,
    # applied to all splits of that series. Store μ,σ for inverse transform.
    train_slices, val_slices, test_slices = [], [], []
    test_keys, test_mu, test_sigma, test_groups = [], [], [], []
    for key, ts in series_list:
        T = len(ts)
        t1 = int(T * 0.70)
        train_seg = ts[:t1]
        mu = float(train_seg.mean())
        sigma = float(train_seg.std()) or 1.0
        norm = (ts - mu) / sigma
        sp = split_and_slice(norm)
        train_slices.append(sp["train"])
        val_slices.append(sp["val"])
        test_slices.append(sp["test"])
        # Track per-test-series metadata for reconstruction
        test_keys.append(key)
        test_mu.append(mu)
        test_sigma.append(sigma)
        test_groups.append(sp["test"])     # normalised slices for this series

    X_train = np.concatenate(train_slices, axis=0)
    X_val = np.concatenate(val_slices, axis=0)
    X_test = np.concatenate(test_slices, axis=0)
    print(f"[data] Slices  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "test_keys": test_keys,
        "test_mu": np.asarray(test_mu, dtype=np.float32),
        "test_sigma": np.asarray(test_sigma, dtype=np.float32),
        "test_groups": test_groups,
    }


# ────────────────────────────────────────────────────────────────────────────
def iterate_batches(X: np.ndarray, batch_size: int, shuffle: bool):
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, n, batch_size):
        yield X[idx[i: i + batch_size]]


def epoch_loop(model, X, loss_fn, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total, count = 0.0, 0
    for batch_np in iterate_batches(X, BATCH_SIZE, shuffle=train_mode):
        batch = torch.from_numpy(batch_np).to(DEVICE, non_blocking=True)
        if batch.size(0) < 2:           # BatchNorm needs ≥2
            continue
        z = encode_slice(batch, noise_scale=NOISE_SCALE)
        pred = model(z)
        loss = loss_fn(pred, batch)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += loss.item() * batch.size(0)
        count += batch.size(0)
    return total / max(count, 1)


# ────────────────────────────────────────────────────────────────────────────
def reconstruct_series(model, ref_slices_norm: np.ndarray,
                       mu: float, sigma: float, n_repeats: int = 1):
    """Decode each reference slice (in normalised space) into a synthesized
    slice with fresh noise, concatenate to a full denormalised series.

    n_repeats > 1 averages multiple noise draws (paper-style ensemble);
    1 is enough for our spectrum/moment comparison since we have many series.
    """
    model.eval()
    n = len(ref_slices_norm)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    with torch.no_grad():
        batch = torch.from_numpy(ref_slices_norm).to(DEVICE)
        gens = []
        for _ in range(n_repeats):
            z = encode_slice(batch, noise_scale=NOISE_SCALE)
            gens.append(model(z).cpu().numpy())
    gen = np.mean(gens, axis=0)                              # (n, L)
    flat = gen.reshape(-1).astype(np.float32)
    return flat * sigma + mu                                 # denormalise


def moments(x: np.ndarray):
    m = float(np.mean(x))
    v = float(np.var(x))
    c = x - m
    s = float(np.mean(c ** 3) / (np.sqrt(v) ** 3 + 1e-12))
    k = float(np.mean(c ** 4) / (v ** 2 + 1e-12))
    return m, v, s, k


def welch_psd(x: np.ndarray):
    f, p = sp_signal.welch(x, fs=PSD_FS, window="hann",
                           nperseg=PSD_NPERSEG, noverlap=PSD_NOVERLAP,
                           detrend="constant", scaling="density")
    return f, p


# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", type=str, default="round2")
    args = parser.parse_args()

    results_dir = HERE / "results"
    plots_dir = results_dir / "plots"
    gen_dir = results_dir / "generated"
    ckpt_dir = HERE / "checkpoints"
    for d in (results_dir, plots_dir, gen_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[env] device={DEVICE}  threads={N_THREADS}")

    # ── Data ────────────────────────────────────────────────────────────────
    data = build_dataset(args.scope)
    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]

    # ── Model ───────────────────────────────────────────────────────────────
    model = WPTSENet(slice_length=SLICE_LENGTH, latent_dim=LATENT_DIM,
                     hidden=HIDDEN, n_blocks=N_BLOCKS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] WPTSE-Net params={n_params:,}")

    loss_fn = nn.HuberLoss(delta=HUBER_DELTA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── Train loop with early stopping ─────────────────────────────────────
    history = {"train": [], "val": []}
    best_val = math.inf
    best_state = None
    bad_epochs = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr = epoch_loop(model, X_train, loss_fn, optimizer)
        va = epoch_loop(model, X_val, loss_fn, optimizer=None)
        history["train"].append(tr)
        history["val"].append(va)
        if va < best_val - 1e-6:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 25 == 0 or bad_epochs == 0:
            print(f"[train] ep={epoch:4d}  train={tr:.5f}  val={va:.5f}  bad={bad_epochs}")
        if bad_epochs >= PATIENCE:
            print(f"[train] early stop at epoch {epoch} (best val={best_val:.5f})")
            break

    train_time = time.time() - t0
    print(f"[train] total time = {train_time:.1f} s")

    # ── Restore best checkpoint ────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = ckpt_dir / "wptse_net_best.pt"
    torch.save({"state_dict": model.state_dict(),
                "slice_length": SLICE_LENGTH,
                "latent_dim": LATENT_DIM,
                "hidden": HIDDEN,
                "n_blocks": N_BLOCKS,
                "scope": args.scope}, ckpt_path)
    print(f"[ckpt] saved → {ckpt_path}")

    # ── Training-curve plot ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train"], label="train")
    ax.plot(history["val"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Huber loss")
    ax.set_title("WPTSE-Net training curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "training_curve.png", dpi=140)
    plt.close(fig)

    # ── Evaluation: time-domain metrics on raw slices ──────────────────────
    model.eval()
    with torch.no_grad():
        rmse_acc, mae_acc, ss_res, ss_tot, n_count = 0.0, 0.0, 0.0, 0.0, 0
        # iterate in batches to control memory
        for batch_np in iterate_batches(X_test, BATCH_SIZE, shuffle=False):
            batch = torch.from_numpy(batch_np).to(DEVICE)
            if batch.size(0) < 2:
                continue
            z = encode_slice(batch, noise_scale=NOISE_SCALE)
            pred = model(z).cpu().numpy()
            tgt = batch_np
            rmse_acc += float(np.sum((pred - tgt) ** 2))
            mae_acc += float(np.sum(np.abs(pred - tgt)))
            ss_res += float(np.sum((pred - tgt) ** 2))
            ss_tot += float(np.sum((tgt - tgt.mean()) ** 2))
            n_count += tgt.size
    rmse = math.sqrt(rmse_acc / max(n_count, 1))
    mae = mae_acc / max(n_count, 1)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    print(f"[eval] time-domain  RMSE={rmse:.5f}  MAE={mae:.5f}  R^2={r2:.5f}")

    # ── Series-level reconstruction + statistical & spectral fidelity ──────
    series_rows = []
    psd_acc_true, psd_acc_gen = None, None
    n_series_used = 0
    sample_for_plot = None

    for i, (key, mu, sigma, slices) in enumerate(zip(
            data["test_keys"], data["test_mu"], data["test_sigma"], data["test_groups"])):
        if len(slices) < 8:
            continue
        # ground-truth denormalised series (the test portion)
        gt = slices.reshape(-1).astype(np.float32) * sigma + mu
        gen = reconstruct_series(model, slices, mu, sigma, n_repeats=1)
        L = min(len(gt), len(gen))
        if L < PSD_NPERSEG * 2:
            continue
        gt, gen = gt[:L], gen[:L]

        gm, gv, gs, gk = moments(gen)
        tm, tv, ts_, tk = moments(gt)

        f, p_true = welch_psd(gt)
        _, p_gen = welch_psd(gen)
        psd_ratio = float(p_gen.sum() / max(p_true.sum(), 1e-12))
        # PSD L^2 distance in log-frequency space (over positive freqs)
        m = f > 0
        l2 = float(np.sqrt(np.mean((np.log10(p_gen[m] + 1e-12) -
                                    np.log10(p_true[m] + 1e-12)) ** 2)))
        peak_true = float(f[np.argmax(p_true)])
        peak_gen = float(f[np.argmax(p_gen)])

        series_rows.append({
            "key": key,
            "n_samples": int(L),
            "mean_true": tm, "mean_gen": gm,
            "var_true": tv, "var_gen": gv,
            "skew_true": ts_, "skew_gen": gs,
            "kurt_true": tk, "kurt_gen": gk,
            "psd_power_ratio": psd_ratio,
            "psd_log_l2": l2,
            "peak_freq_true_hz": peak_true,
            "peak_freq_gen_hz": peak_gen,
        })
        psd_acc_true = p_true if psd_acc_true is None else psd_acc_true + p_true
        psd_acc_gen = p_gen if psd_acc_gen is None else psd_acc_gen + p_gen
        n_series_used += 1
        if sample_for_plot is None:
            sample_for_plot = {"key": key, "gt": gt.copy(), "gen": gen.copy(),
                               "f": f, "p_true": p_true, "p_gen": p_gen}
            # save generated time series for inspection
            np.save(gen_dir / "sample_gen.npy", gen)
            np.save(gen_dir / "sample_true.npy", gt)

    # Aggregate moments and PSD across series
    if not series_rows:
        raise RuntimeError("No usable test series — check slice length / data.")

    def col(name):
        return np.array([r[name] for r in series_rows], dtype=np.float64)

    def rel_err(true_col, gen_col):
        denom = np.abs(true_col).clip(min=1e-6)
        return float(np.mean(np.abs(gen_col - true_col) / denom))

    moments_err = {
        "mean_rel_err": rel_err(col("mean_true"), col("mean_gen")),
        "var_rel_err": rel_err(col("var_true"), col("var_gen")),
        "skew_rel_err": rel_err(col("skew_true"), col("skew_gen")),
        "kurt_rel_err": rel_err(col("kurt_true"), col("kurt_gen")),
    }
    psd_avg_true = psd_acc_true / n_series_used
    psd_avg_gen = psd_acc_gen / n_series_used
    total_power_ratio = float(psd_avg_gen.sum() / max(psd_avg_true.sum(), 1e-12))
    fmask = sample_for_plot["f"] > 0
    psd_log_l2_avg = float(np.sqrt(np.mean(
        (np.log10(psd_avg_gen[fmask] + 1e-12) -
         np.log10(psd_avg_true[fmask] + 1e-12)) ** 2)))

    # ── Save per-series metrics CSV ────────────────────────────────────────
    per_series_csv = results_dir / "per_series_metrics.csv"
    with open(per_series_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(series_rows[0].keys()))
        w.writeheader()
        w.writerows(series_rows)

    # ── Save summary metrics ────────────────────────────────────────────────
    summary = {
        "model": "WPTSE-Net",
        "scope": args.scope,
        "n_params": n_params,
        "train_time_s": round(train_time, 1),
        "epochs_run": len(history["train"]),
        "best_val_huber": round(best_val, 6),
        "rmse_slice": round(rmse, 6),
        "mae_slice": round(mae, 6),
        "r2_slice": round(r2, 6),
        "n_test_series": n_series_used,
        "psd_total_power_ratio": round(total_power_ratio, 6),
        "psd_log_l2": round(psd_log_l2_avg, 6),
        **{k: round(v, 6) for k, v in moments_err.items()},
    }
    summary_csv = results_dir / "metrics.csv"
    with open(summary_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    print("[metrics]", summary)

    # ── Plots ───────────────────────────────────────────────────────────────
    s = sample_for_plot
    # Time-domain (2-second snippet)
    snip = min(2 * PSD_FS, len(s["gt"]))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    t = np.arange(snip) / PSD_FS
    ax.plot(t, s["gt"][:snip], label="true", color="#1f77b4", lw=1.1)
    ax.plot(t, s["gen"][:snip], label="generated", color="#d62728", lw=1.1, alpha=0.8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("Cp")
    ax.set_title(f"Time-series comparison — {s['key']}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "timeseries_comparison.png", dpi=140)
    plt.close(fig)

    # PSD comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    m_ = s["f"] > 0
    ax1.loglog(s["f"][m_], s["p_true"][m_], label="true", color="#1f77b4")
    ax1.loglog(s["f"][m_], s["p_gen"][m_], label="generated", color="#d62728", alpha=0.8)
    ax1.set_xlabel("frequency [Hz]")
    ax1.set_ylabel("PSD [Cp²/Hz]")
    ax1.set_title(f"Welch PSD — {s['key']}")
    ax1.legend()
    ax1.grid(which="both", alpha=0.3)

    ax2.loglog(s["f"][m_], psd_avg_true[m_], label="true (avg)", color="#1f77b4")
    ax2.loglog(s["f"][m_], psd_avg_gen[m_], label="generated (avg)", color="#d62728", alpha=0.8)
    ax2.set_xlabel("frequency [Hz]")
    ax2.set_ylabel("PSD [Cp²/Hz]")
    ax2.set_title(f"Welch PSD — mean over {n_series_used} test series")
    ax2.legend()
    ax2.grid(which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "psd_comparison.png", dpi=140)
    plt.close(fig)

    print(f"[done] artifacts under {results_dir}")


if __name__ == "__main__":
    main()

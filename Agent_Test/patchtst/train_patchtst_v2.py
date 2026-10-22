"""
PatchTST v2 — GPU plan 2026-10
(Handoffs/2026-10-21_plan_GPU_ventana_y_restriccion_estadistica.md).

What changes versus train_patchtst.py:
  - configurable look-back (--seq) and patching (--patch-len/--patch-stride);
    patches are aligned to the END of the window by default so the most
    recent samples are seen (--align start = original model, which drops the
    last (seq - patch_len) % stride samples: 4 ms at seq 100, 8 ms at 1000,
    16 ms at 2000);
  - val/test windows take their context from the previous split
    (load_all_buildings_ctx), so every config is scored on the same targets;
  - windows are gathered on the GPU (WindowLoader), nothing is materialised;
  - training schedule equivalent in OPTIMIZER STEPS to the published stride-10
    runs: max epochs / early-stopping patience / LR patience = 200 / 15 / 5
    x step/10 (600 / 45 / 15 at step 30). Protocol A of the plan kept 300/15/5
    epochs, i.e. a third of the steps between LR cuts, and under-trained;
  - optional StatLoss (--lam-psd, --lam-std, --psd-bins) on template-free
    statistics (--legacy-statloss = original version, which the model met with
    an input-independent template);
  - extra scores: RMSE averaged over h=1..H, template share (variance of the
    shape shared by all windows) and PC1 share of the forecast deviations;
  - the naive persistence baseline is scored once per scope on the same windows.

Same resource limits as train_patchtst.py (85% GPU memory, 4 CPU cores free).

Usage (from Agent_Test/)
-----
    python patchtst/train_patchtst_v2.py --tag 1b --seq 1000 --patch-len 64 --patch-stride 32
    python patchtst/train_patchtst_v2.py --tag 2b --seq 1000 --patch-len 64 --patch-stride 32 --lam-psd 0.003 --lam-std 1
    python patchtst/train_patchtst_v2.py ... --smoke --max-epochs 1     # -> results/gpu_plan_2026-10/_smoke/

Outputs (one folder per run, named from the flags)
-------
    results/gpu_plan_2026-10/metrics.csv           one row per run x split x h
    results/gpu_plan_2026-10/runs/<run>/
        config.json  summary.json  train.log  history.csv  training_curves.png
        rmse_curve.npy (test, h=1..H)  power_ratio.csv  psd.npz
        series_meta.csv  forecast_examples.png
        big/  preds.npy  trues.npy  series_id.npy  target_start.npy   (gitignored)
    patchtst/checkpoints/gpu_plan_2026-10/<run>.pt
A run whose summary.json exists is skipped, so re-launching after a crash
resumes; --force retrains it.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # Agent_Test/patchtst/
AGENT = ROOT.parent                             # Agent_Test/
sys.path.append(str(AGENT))
sys.path.append(str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import psutil
import torch


# ─── Resource limits (must run BEFORE any CUDA op) — as train_patchtst.py ──
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

from data_loader import load_all_buildings_ctx
from train_utils import (set_seed, DEVICE, train_model, WindowLoader,
                         ROUND2_ALPHAS, ROUND2_RATIOS)
from stat_loss import StatLoss
from models.patchtst import PatchTST
import matplotlib.pyplot as plt                 # Agg backend set by train_utils


PLAN_DIR = AGENT / "results" / "gpu_plan_2026-10"
CKPT_DIR = ROOT / "checkpoints" / "gpu_plan_2026-10"
SCOPE_TAG = {"round2": "r2", "all": "r3"}

FS = 1000.0
EVAL_HS = [1, 10, 50, 100, 500]
BANDS = {"total": (2, 501), "2_10": (2, 10), "10_50": (10, 50), "50_500": (50, 501)}  # Hz, [lo, hi)
POWER_KEYS = [f"power_ratio_{b}" for b in BANDS]
# per run x split scores besides the per-h ones
EV_KEYS = POWER_KEYS + ["std_pred", "std_true", "std_ratio", "rmse_mean", "rmse_max",
                        "h_rmse_max", "tpl_share", "tpl_share_true", "pc1_share",
                        "pc1_share_true"]
EVAL_BATCH = 512

# PatchTST architecture (as train_patchtst.py); optimiser per plan rule 7
D_MODEL, N_HEADS, D_FF, N_LAYERS, DROPOUT = 128, 4, 256, 3, 0.1
LR, WEIGHT_DECAY = 1e-3, 1e-4

METRIC_COLS = (
    ["run", "tag", "scope", "split", "h", "rmse", "mae", "r2"] + EV_KEYS
    + ["n_windows", "seq", "patch_len", "patch_stride", "align", "step", "test_step",
       "lam_psd", "lam_std", "psd_bins", "statloss", "exclude_ratios", "demean", "seed",
       "max_epochs", "patience", "lr_patience",
       "params", "epochs_run", "best_epoch", "best_val_loss", "best_val_mse",
       "train_time_s", "s_per_epoch", "peak_gpu_mem_mb", "peak_ram_gb", "timestamp"]
)
# empty for the naive rows -> nullable ints, so they are not written as 1000.0
INT_COLS = ["seq", "patch_len", "patch_stride", "seed", "max_epochs", "patience", "lr_patience",
            "params", "epochs_run", "best_epoch", "peak_gpu_mem_mb", "h_rmse_max"]


def schedule_defaults(step: int):
    """(max_epochs, patience, lr_patience) with the optimizer steps of the
    published stride-10 runs (200 / 15 / 5 epochs)."""
    s = step / 10
    return round(200 * s), round(15 * s), round(5 * s)


def parse_args():
    p = argparse.ArgumentParser(description="PatchTST v2 (GPU plan 2026-10)")
    p.add_argument("--scope", choices=["round2", "all"], default="round2")
    p.add_argument("--tag", required=True, help="run id from the plan, e.g. 1a, 2b")
    p.add_argument("--seq", type=int, default=100)
    p.add_argument("--patch-len", type=int, default=16)
    p.add_argument("--patch-stride", type=int, default=8)
    p.add_argument("--align", choices=["end", "start"], default="end",
                   help="patch alignment; 'start' = original model (misses the newest samples)")
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--step", type=int, default=30, help="train/val window stride")
    p.add_argument("--test-step", type=int, default=10)
    p.add_argument("--lam-psd", type=float, default=0.0)
    p.add_argument("--lam-std", type=float, default=0.0)
    p.add_argument("--psd-bins", choices=["octave", "linear"], default="octave")
    p.add_argument("--legacy-statloss", action="store_true",
                   help="original StatLoss: statistics on the raw forecasts (templates allowed)")
    p.add_argument("--demean", action="store_true",
                   help="remove from each series its own train-portion mean (in memory; "
                        "added back before scoring)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=None, help="default 200 x step/10")
    p.add_argument("--patience", type=int, default=None,
                   help="early stopping, default 15 x step/10")
    p.add_argument("--lr-patience", type=int, default=None,
                   help="ReduceLROnPlateau, default 5 x step/10")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--exclude-ratios", nargs="+", default=None,
                   help="leave-buildings-out: drop these ratios from train/val, test on them")
    p.add_argument("--smoke", action="store_true",
                   help="write to _smoke/ (gitignored) and never skip")
    p.add_argument("--force", action="store_true",
                   help="retrain even if the run already finished")
    a = p.parse_args()
    max_ep, pat, lr_pat = schedule_defaults(a.step)
    a.max_epochs = a.max_epochs if a.max_epochs is not None else max_ep
    a.patience = a.patience if a.patience is not None else pat
    a.lr_patience = a.lr_patience if a.lr_patience is not None else lr_pat
    return a


def run_name(a) -> str:
    """Unique per configuration (plan rule 6), e.g. 2b_s1000_p64-32_lpsd0.003_lstd1_r2."""
    parts = [a.tag, f"s{a.seq}", f"p{a.patch_len}-{a.patch_stride}"]
    if a.align == "start":
        parts.append("legacypatch")
    if a.lam_psd == 0 and a.lam_std == 0:
        parts.append("mse")
    else:
        parts.append(f"lpsd{a.lam_psd:g}_lstd{a.lam_std:g}")
        if a.lam_psd > 0 and a.psd_bins == "linear":
            parts.append("lin")
        if a.legacy_statloss:
            parts.append("legacystat")
    if a.demean:
        parts.append("dm")
    if a.seed != 42:
        parts.append(f"seed{a.seed}")
    if a.step != 30:
        parts.append(f"st{a.step}")
    if (a.max_epochs, a.patience, a.lr_patience) != schedule_defaults(a.step):
        parts.append(f"ep{a.max_epochs}-pat{a.patience}-lrp{a.lr_patience}")
    if a.batch != 128:
        parts.append(f"b{a.batch}")
    if a.exclude_ratios:
        parts.append("lbo")
    parts.append(SCOPE_TAG[a.scope])
    return "_".join(parts)


class Tee:
    """Mirror a console stream into the run's train.log (keeps tracebacks of unattended runs)."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def tee_to(path: Path):
    fh = open(path, "w", encoding="utf-8")
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")     # never die on a console codepage
        setattr(sys, name, Tee(stream, fh))


def git_version() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=AGENT,
                             capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=AGENT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return ""


def to_json(path: Path, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


class LastValue(torch.nn.Module):
    """Naive persistence: last observed windward value, repeated over the horizon."""

    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon

    def forward(self, x):
        return x[:, -1:, 0].expand(-1, self.horizon)


# ─── Evaluation (streaming, Cp units) ───────────────────────────────────────

def _shape_shares(sd: np.ndarray, ss: float, C: np.ndarray, n: int, H: int):
    """Shares of the within-window variance (deviations from each window's mean):
    template = energy of the mean deviation shape over all windows (a shape that
    every forecast repeats; ~0 for a stationary truth); PC1 = largest eigenvalue
    of the deviation covariance / its trace (a family of one shape)."""
    within = ss / (n * H)
    if within <= 0:
        return 0.0, float("nan")
    T = sd / n
    eig = np.linalg.eigvalsh(C / n - np.outer(T, T))
    return float((T ** 2).mean() / within), float(eig[-1] / eig.sum())


@torch.no_grad()
def evaluate(model, loader, mu_w, sigma_w, H, keep=False, offset=None) -> dict:
    """
    Scores over every window of `loader`: per-h RMSE / MAE / R², their mean
    and max over h, window-mean periodograms (hann over the H steps, mean
    removed, df = fs/H), the mean within-window std and the template / PC1
    shares of prediction and truth. keep=True also returns preds/trues.
    offset: per-window Cp level added back to prediction and truth (the
    series means removed with --demean), in loader order.
    """
    model.eval()
    off = (torch.as_tensor(offset, dtype=torch.float64, device=DEVICE)[:, None]
           if offset is not None else None)
    win = torch.hann_window(H, dtype=torch.float64, device=DEVICE)

    def zeros(n):
        return torch.zeros(n, dtype=torch.float64, device=DEVICE)

    def periodogram(d):
        return torch.fft.rfft(win * d).abs().pow(2)

    se, ae, ys, ys2 = zeros(H), zeros(H), zeros(H), zeros(H)
    pp, pt = zeros(H // 2 + 1), zeros(H // 2 + 1)
    sp, st = zeros(1), zeros(1)
    sd_p, sd_y, ss_p, ss_y = zeros(H), zeros(H), zeros(1), zeros(1)
    C_p, C_y = zeros((H, H)), zeros((H, H))
    n = 0
    preds, trues = [], []
    for xb, yb in loader:
        p = model(xb).double() * sigma_w + mu_w
        y = yb.double() * sigma_w + mu_w
        if off is not None:
            p = p + off[n: n + len(yb)]
            y = y + off[n: n + len(yb)]
        e = p - y
        se += e.pow(2).sum(0)
        ae += e.abs().sum(0)
        ys += y.sum(0)
        ys2 += y.pow(2).sum(0)
        dp = p - p.mean(-1, keepdim=True)                 # within-window deviations
        dy = y - y.mean(-1, keepdim=True)
        pp += periodogram(dp).sum(0)
        pt += periodogram(dy).sum(0)
        sp += dp.std(-1, correction=0).sum()
        st += dy.std(-1, correction=0).sum()
        sd_p += dp.sum(0)
        sd_y += dy.sum(0)
        ss_p += dp.pow(2).sum()
        ss_y += dy.pow(2).sum()
        C_p += dp.T @ dp
        C_y += dy.T @ dy
        n += len(y)
        if keep:
            preds.append(p.float().cpu())
            trues.append(y.float().cpu())

    se, ae, ys, ys2, pp, pt, sd_p, sd_y, C_p, C_y = (
        t.cpu().numpy() for t in (se, ae, ys, ys2, pp, pt, sd_p, sd_y, C_p, C_y))
    f = np.fft.rfftfreq(H, 1 / FS)
    rmse = np.sqrt(se / n)
    ev = {
        "n": n,
        "rmse": rmse,
        "mae": ae / n,
        "r2": 1.0 - se / (ys2 - ys ** 2 / n),
        "rmse_mean": float(rmse.mean()), "rmse_max": float(rmse.max()),
        "h_rmse_max": int(rmse.argmax()) + 1,
        "f": f, "psd_pred": pp / n, "psd_true": pt / n,
        "std_pred": sp.item() / n, "std_true": st.item() / n,
    }
    ev["std_ratio"] = ev["std_pred"] / ev["std_true"]
    for band, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        ev[f"power_ratio_{band}"] = float(pp[m].sum() / pt[m].sum())
    ev["tpl_share"], ev["pc1_share"] = _shape_shares(sd_p, ss_p.item(), C_p, n, H)
    ev["tpl_share_true"], ev["pc1_share_true"] = _shape_shares(sd_y, ss_y.item(), C_y, n, H)
    if keep:
        ev["preds"] = torch.cat(preds).numpy()
        ev["trues"] = torch.cat(trues).numpy()
    return ev


def compact(ev: dict) -> dict:
    d = {"n_windows": ev["n"]}
    d.update({f"rmse_h{h}": float(ev["rmse"][h - 1]) for h in EVAL_HS})
    d.update({f"r2_h{h}": float(ev["r2"][h - 1]) for h in EVAL_HS})
    d.update({k: ev[k] for k in EV_KEYS})
    return d


def write_results(run_dir: Path, info: dict, ev_val: dict, ev_test: dict, metrics_csv: Path):
    """Per-run score files + this run's rows in metrics.csv (older rows of the same run replaced)."""
    rows = []
    for split, ev in (("val", ev_val), ("test", ev_test)):
        for h in EVAL_HS:
            rows.append({**info, "split": split, "h": h,
                         "rmse": ev["rmse"][h - 1], "mae": ev["mae"][h - 1], "r2": ev["r2"][h - 1],
                         "n_windows": ev["n"], **{k: ev[k] for k in EV_KEYS}})
    new = pd.DataFrame(rows)
    if metrics_csv.exists():
        old = pd.read_csv(metrics_csv)
        new = pd.concat([old[old["run"] != info["run"]], new], ignore_index=True)
    new = new.reindex(columns=METRIC_COLS)
    new[INT_COLS] = new[INT_COLS].astype("Int64")
    new.to_csv(metrics_csv, index=False)

    np.save(run_dir / "rmse_curve.npy", ev_test["rmse"].astype(np.float32))
    np.savez(run_dir / "psd.npz", f=ev_test["f"],
             val_pred=ev_val["psd_pred"], val_true=ev_val["psd_true"],
             test_pred=ev_test["psd_pred"], test_true=ev_test["psd_true"])
    pr = []
    for split, ev in (("val", ev_val), ("test", ev_test)):
        for band, (lo, hi) in BANDS.items():
            m = (ev["f"] >= lo) & (ev["f"] < hi)
            pr.append({"split": split, "quantity": f"power_{band}",
                       "band_hz": f"{lo}-{min(hi, int(FS // 2))}",
                       "pred": ev["psd_pred"][m].sum(), "true": ev["psd_true"][m].sum(),
                       "ratio": ev[f"power_ratio_{band}"]})
        pr.append({"split": split, "quantity": "std_window", "band_hz": "",
                   "pred": ev["std_pred"], "true": ev["std_true"], "ratio": ev["std_ratio"]})
    pd.DataFrame(pr).to_csv(run_dir / "power_ratio.csv", index=False)
    # written last: its presence marks the run as finished
    to_json(run_dir / "summary.json", {**info, "val": compact(ev_val), "test": compact(ev_test)})


# ─── Plots ───────────────────────────────────────────────────────────────────

def plot_history(path: Path, title: str, history: dict, show_mse: bool):
    ep = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ep, history["train_loss"], label="train loss", color="steelblue")
    ax.plot(ep, history["val_loss"], label="val loss", color="darkorange")
    if show_mse:
        ax.plot(ep, history["val_mse"], label="val MSE", color="darkorange", ls="--")
    ax.axvline(history["best_epoch"], color="k", lw=0.6, ls=":", label="best epoch")
    ax.set(xlabel="Epoch", ylabel="Loss (normalised)", title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_examples(path: Path, title: str, series_n: np.ndarray, test_idx: np.ndarray,
                  series_meta: list, preds: np.ndarray, mu_w: float, sigma_w: float,
                  seq: int, H: int, n_show: int = 4, ctx: int = 300, level=None):
    """Fixed test windows (the same for every run of a scope): truth, forecast, naive.
    level: per-series Cp means removed with --demean (added back to the truth)."""
    picks = [int((i + 0.5) * len(test_idx) / n_show) for i in range(n_show)]
    fig, axes = plt.subplots(n_show, 1, figsize=(12, 2.6 * n_show), sharex=True)
    for ax, k in zip(np.atleast_1d(axes), picks):
        row, start = test_idx[k]
        t0 = start + seq                                     # first target sample
        truth = series_n[row, t0 - ctx: t0 + H, 0] * sigma_w + mu_w
        if level is not None:
            truth = truth + level[row]
        _, alpha, ratio, angle = series_meta[row]
        rmse = float(np.sqrt(np.mean((preds[k] - truth[ctx:]) ** 2)))
        ax.axvspan(-min(seq, ctx), 0, color="tab:green", alpha=0.08, lw=0,
                   label=f"look-back ({seq} ms{', cut' if seq > ctx else ''})")
        ax.plot(np.arange(-ctx, H), truth, color="0.25", lw=0.8, label="true")
        ax.hlines(truth[ctx - 1], 0, H - 1, color="tab:blue", ls="--", lw=1, label="naive")
        ax.plot(np.arange(H), preds[k], color="tab:orange", lw=1.3, label="forecast")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(f"{alpha}/{ratio}/a{angle}   t0={t0 / FS:.2f} s   RMSE={rmse:.3f}",
                     fontsize=9)
        ax.set_ylabel("Cp")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=8, ncol=4)
    np.atleast_1d(axes)[-1].set_xlabel("ms from forecast origin")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    H = args.horizon
    name = run_name(args)
    out_root = PLAN_DIR / "_smoke" if args.smoke else PLAN_DIR
    run_dir = out_root / "runs" / name
    metrics_csv = out_root / "metrics.csv"
    ckpt = (CKPT_DIR / "_smoke" if args.smoke else CKPT_DIR) / f"{name}.pt"

    if (run_dir / "summary.json").exists() and not (args.force or args.smoke):
        print(f"[skip] {name}: already finished ({run_dir / 'summary.json'}); --force retrains it")
        return

    (run_dir / "big").mkdir(parents=True, exist_ok=True)
    tee_to(run_dir / "train.log")
    print(f"\n{'=' * 70}\n  PatchTST v2  {name}\n{'=' * 70}")
    print(f"Device: {DEVICE}  ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'cpu'})"
          f"  | GPU memory cap 85%  | CPU threads {torch.get_num_threads()}")
    set_seed(args.seed)

    # ── Data (rule 2: val/test context taken from the previous split) ──
    scope_kw = (dict(alphas=ROUND2_ALPHAS, ratios=ROUND2_RATIOS)
                if args.scope == "round2" else {})
    ctx_kw = dict(seq_length=args.seq, step=args.step, horizon=H, test_step=args.test_step,
                  demean=args.demean)
    data = load_all_buildings_ctx(exclude_ratios=args.exclude_ratios, **ctx_kw, **scope_kw)
    test_data = data
    if args.exclude_ratios:
        print(f"Held-out test set: ratios {args.exclude_ratios} (normalised with the train mu/sigma)")
        test_data = load_all_buildings_ctx(only_ratios=args.exclude_ratios,
                                           mu_sigma=(data["mu"], data["sigma"]),
                                           **ctx_kw, **scope_kw)
    mu_w, sigma_w = float(data["mu"][0]), float(data["sigma"][0])
    # per-window Cp level to add back when scoring (only with --demean)
    val_off = data["series_mean"][data["val_idx"][:, 0], 0] if args.demean else None
    test_off = (test_data["series_mean"][test_data["test_idx"][:, 0], 0]
                if args.demean else None)

    series = torch.from_numpy(data["series"]).to(DEVICE)
    test_series = (series if test_data is data
                   else torch.from_numpy(test_data["series"]).to(DEVICE))
    train_loader = WindowLoader(series, data["train_idx"], args.seq, H, args.batch, shuffle=True)
    # val batches for early stopping mix series like the shuffled train batches (the
    # StatLoss terms are batch-level), in a fixed order so epochs stay comparable
    val_mixed = data["val_idx"][np.random.default_rng(0).permutation(len(data["val_idx"]))]
    val_loader = WindowLoader(series, val_mixed, args.seq, H, args.batch * 2)
    val_eval = WindowLoader(series, data["val_idx"], args.seq, H, EVAL_BATCH)
    test_loader = WindowLoader(test_series, test_data["test_idx"], args.seq, H, EVAL_BATCH)

    config = {
        "run": name, **vars(args),
        "n_series_train": len(data["series_meta"]), "n_series_test": len(test_data["series_meta"]),
        "n_windows": {"train": len(data["train_idx"]), "val": len(data["val_idx"]),
                      "test": len(test_data["test_idx"])},
        "mu": data["mu"].tolist(), "sigma": data["sigma"].tolist(),
        "model": dict(d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, n_layers=N_LAYERS,
                      dropout=DROPOUT),
        "optim": dict(lr=LR, weight_decay=WEIGHT_DECAY, clip_grad_norm=1.0, seed=42,
                      lr_schedule=f"ReduceLROnPlateau(factor=0.5, patience={args.lr_patience})"),
        "checkpoint": str(ckpt.relative_to(AGENT)),
        "git": git_version(), "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu",
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    to_json(run_dir / "config.json", config)
    pd.DataFrame(test_data["series_meta"], columns=["series_id", "alpha", "ratio", "angle"]) \
        .to_csv(run_dir / "series_meta.csv", index=False)

    # ── Naive persistence on the same windows (identical for every seq) ──
    naive_name = f"naive_{SCOPE_TAG[args.scope]}" + ("_lbo" if args.exclude_ratios else "")
    naive_dir = out_root / "runs" / naive_name
    naive_sum = {}
    if (naive_dir / "summary.json").exists():
        with open(naive_dir / "summary.json", encoding="utf-8") as f:
            naive_sum = json.load(f)
    if not all(k in naive_sum.get("test", {}) for k in EV_KEYS):   # missing or older version
        print(f"\nScoring naive persistence on the same val/test windows -> {naive_name}")
        naive_dir.mkdir(parents=True, exist_ok=True)
        naive = LastValue(H)
        write_results(naive_dir,
                      {"run": naive_name, "tag": "naive", "scope": args.scope,
                       "step": args.step, "test_step": args.test_step,
                       "exclude_ratios": " ".join(args.exclude_ratios or []),
                       "timestamp": datetime.now().isoformat(timespec="seconds")},
                      evaluate(naive, val_eval, mu_w, sigma_w, H, offset=val_off),
                      evaluate(naive, test_loader, mu_w, sigma_w, H, offset=test_off),
                      metrics_csv)
    with open(naive_dir / "summary.json", encoding="utf-8") as f:
        naive_sum = json.load(f)

    # ── Model + loss ──
    model = PatchTST(n_features=4, seq_len=args.seq, horizon=H,
                     patch_len=args.patch_len, stride=args.patch_stride,
                     d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
                     n_layers=N_LAYERS, dropout=DROPOUT,
                     align_end=args.align == "end").to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    gap = (args.seq - args.patch_len) % args.patch_stride
    print(f"\nParameters: {n_par:,}  | tokens: {model.embed.n_patches}  | "
          f"uncovered samples: {gap} at the "
          f"{'oldest' if args.align == 'end' else 'MOST RECENT'} end of the window")
    use_stat = args.lam_psd > 0 or args.lam_std > 0
    loss_fn = (StatLoss(H, args.lam_psd, args.lam_std, bins=args.psd_bins,
                        detemplate=not args.legacy_statloss).to(DEVICE)
               if use_stat else None)
    print(f"Loss: {'StatLoss' if use_stat else 'MSE'}"
          + (f"  lam_psd={args.lam_psd:g} ({args.psd_bins})  lam_std={args.lam_std:g}  "
             + ("LEGACY: templates allowed" if args.legacy_statloss
                else "template-free statistics")
             if use_stat else ""))
    print(f"Schedule: max {args.max_epochs} epochs, early-stopping patience {args.patience}, "
          f"LR patience {args.lr_patience}  ({len(train_loader)} steps/epoch)")

    # ── Train ──
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history: dict = {}
    tl, _, tt, bv = train_model(model, train_loader, val_loader, ckpt,
                                max_epochs=args.max_epochs, lr=LR,
                                weight_decay=WEIGHT_DECAY, patience=args.patience,
                                lr_patience=args.lr_patience,
                                loss_fn=loss_fn, history=history)
    best_epoch = history["best_epoch"]
    s_per_epoch = float(np.mean(history["epoch_time_s"]))
    peak_gpu = (torch.cuda.max_memory_allocated() / 2 ** 20
                if DEVICE.type == "cuda" else float("nan"))
    print(f"\nTraining time: {tt:.1f} s ({tt / 60:.1f} min), {len(tl)} epochs, "
          f"{s_per_epoch:.2f} s/epoch | best epoch {best_epoch}: val loss {bv:.6f}, "
          f"val MSE {history['val_mse'][best_epoch - 1]:.6f}")

    # ── Evaluate the best checkpoint on val and test ──
    t_ev = time.time()
    ev_val = evaluate(model, val_eval, mu_w, sigma_w, H, offset=val_off)
    ev_test = evaluate(model, test_loader, mu_w, sigma_w, H, keep=True, offset=test_off)
    mem = psutil.Process().memory_info()
    peak_ram = getattr(mem, "peak_wset", mem.rss) / 1e9
    print(f"Evaluation: {time.time() - t_ev:.1f} s | peak GPU memory {peak_gpu:.0f} MB "
          f"| peak RAM {peak_ram:.1f} GB")

    info = {
        "run": name, "tag": args.tag, "scope": args.scope,
        "seq": args.seq, "patch_len": args.patch_len, "patch_stride": args.patch_stride,
        "align": args.align, "step": args.step, "test_step": args.test_step,
        "lam_psd": args.lam_psd, "lam_std": args.lam_std,
        "psd_bins": args.psd_bins if args.lam_psd > 0 else "",
        "statloss": ("" if not use_stat
                     else "legacy" if args.legacy_statloss else "template-free"),
        "exclude_ratios": " ".join(args.exclude_ratios or []),
        "demean": args.demean, "seed": args.seed,
        "max_epochs": args.max_epochs, "patience": args.patience,
        "lr_patience": args.lr_patience,
        "params": n_par, "epochs_run": len(tl), "best_epoch": best_epoch,
        "best_val_loss": bv, "best_val_mse": history["val_mse"][best_epoch - 1],
        "train_time_s": round(tt, 1), "s_per_epoch": round(s_per_epoch, 3),
        "peak_gpu_mem_mb": round(peak_gpu), "peak_ram_gb": round(peak_ram, 2),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    # ── Outputs ──
    pd.DataFrame({"epoch": np.arange(1, len(tl) + 1),
                  **{k: history[k] for k in ("train_loss", "val_loss", "val_mse", "lr",
                                             "epoch_time_s")}}) \
        .to_csv(run_dir / "history.csv", index=False)
    plot_history(run_dir / "training_curves.png", name, history, show_mse=use_stat)
    plot_examples(run_dir / "forecast_examples.png", name, test_data["series"],
                  test_data["test_idx"], test_data["series_meta"], ev_test["preds"],
                  mu_w, sigma_w, args.seq, H,
                  level=test_data["series_mean"][:, 0] if args.demean else None)
    np.save(run_dir / "big" / "preds.npy", ev_test["preds"])
    np.save(run_dir / "big" / "trues.npy", ev_test["trues"])
    np.save(run_dir / "big" / "series_id.npy", test_data["test_meta"]["series_id"].astype(np.int16))
    np.save(run_dir / "big" / "target_start.npy",
            test_data["test_meta"]["target_start"].astype(np.int32))
    write_results(run_dir, info, ev_val, ev_test, metrics_csv)

    print(f"\n-- {name} vs naive (same windows) --")
    print(f"  {'':>6}  {'val RMSE':>9} {'naive':>7}   {'test RMSE':>9} {'naive':>7}  {'test R2':>7}")
    for h in EVAL_HS:
        print(f"  h={h:>3d}  {ev_val['rmse'][h - 1]:9.4f} {naive_sum['val'][f'rmse_h{h}']:7.4f}"
              f"   {ev_test['rmse'][h - 1]:9.4f} {naive_sum['test'][f'rmse_h{h}']:7.4f}"
              f"  {ev_test['r2'][h - 1]:7.4f}")
    print(f"  test power ratio  total {ev_test['power_ratio_total']:.3f} | "
          f"2-10 Hz {ev_test['power_ratio_2_10']:.3f} | 10-50 Hz {ev_test['power_ratio_10_50']:.3f} | "
          f"50-500 Hz {ev_test['power_ratio_50_500']:.3f}")
    print(f"  test window std   pred {ev_test['std_pred']:.4f}  true {ev_test['std_true']:.4f}  "
          f"(ratio {ev_test['std_ratio']:.3f})")
    print(f"  test RMSE over h=1..{H}: mean {ev_test['rmse_mean']:.4f} "
          f"(naive {naive_sum['test']['rmse_mean']:.4f}), max {ev_test['rmse_max']:.4f} "
          f"at h={ev_test['h_rmse_max']}")
    print(f"  shared-shape check: template share {ev_test['tpl_share']:.3f} "
          f"(truth {ev_test['tpl_share_true']:.3f}) | PC1 share {ev_test['pc1_share']:.3f} "
          f"(truth {ev_test['pc1_share_true']:.3f})")
    print(f"\nRun folder: {run_dir}")


if __name__ == "__main__":
    main()

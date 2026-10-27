"""
Experiment 7: are the model's uncertainty bands worth it? Compares a
multi-quantile run (--quantiles) with two model-free bands on the same test
windows, all in Cp:

  climatology  each series' own quantiles, from its train portion (first 70 %),
               the same at every horizon
  naive        last observed value + the quantiles of the h-step change
               y[t+h] - y[t], pooled over the train portions of all series

Scores per horizon: share of true values inside each central interval
(should match its level), interval width (narrower is better at equal
coverage) and the mean pinball loss over the quantiles (lower is better; it
rewards calibration and sharpness together).

Usage (from Agent_Test/):  python evaluate_bands.py [--run 7a_s2000_p128-64_qr5_dm_r3]
Outputs -> runs/<run>/: bands_vs_baselines.csv, bands_vs_baselines.png
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_faces

AGENT = Path(__file__).resolve().parent
RUNS = AGENT / "results" / "gpu_plan_2026-10" / "runs"
T, TRAIN_R, H = 32768, 0.70, 500


def scores(lo_hi, y, qs, quantiles):
    """Coverage / width per interval and mean pinball per horizon. qs: (N, Q, H) or broadcastable."""
    out = {}
    for j, (lo, hi) in enumerate(lo_hi):
        lvl = round((quantiles[hi] - quantiles[lo]) * 100)
        out[f"coverage{lvl}"] = ((y >= qs[:, lo]) & (y <= qs[:, hi])).mean(0)
        out[f"width{lvl}"] = np.broadcast_to(qs[:, hi] - qs[:, lo], y.shape).mean(0)
    pin = np.zeros(y.shape[1])
    for i, q in enumerate(quantiles):
        e = y - qs[:, i]
        pin += np.maximum(q * e, (q - 1) * e).mean(0)
    out["pinball"] = pin / len(quantiles)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="7a_s2000_p128-64_qr5_dm_r3")
    run = RUNS / ap.parse_args().run
    quantiles = list(np.load(run / "quantile_curves.npz")["quantiles"])
    Q = len(quantiles)
    lo_hi = [(i, Q - 1 - i) for i in range(Q // 2)]

    y = np.load(run / "big" / "trues.npy").astype(np.float32)
    qm = np.load(run / "big" / "quantiles.npy").astype(np.float32)          # (N, Q, H)
    sid = np.load(run / "big" / "series_id.npy")
    t0 = np.load(run / "big" / "target_start.npy")
    meta = pd.read_csv(run / "series_meta.csv").set_index("series_id")

    # climatology quantiles per series and last observed values per window
    t1 = int(T * TRAIN_R)
    clim = np.zeros((len(y), Q, 1), dtype=np.float32)
    last = np.zeros((len(y), 1, 1), dtype=np.float32)
    diffs = []
    for s, r in meta.iterrows():
        x = load_faces(r.alpha, r.ratio, int(r.angle))[:, 0]
        k = sid == s
        clim[k, :, 0] = np.quantile(x[:t1], quantiles)
        last[k, 0, 0] = x[t0[k] - 1]
        starts = np.arange(0, t1 - H, 100)
        diffs.append(x[starts[:, None] + np.arange(1, H + 1)] - x[starts, None])
    dq = np.quantile(np.concatenate(diffs), quantiles, axis=0).astype(np.float32)  # (Q, H)

    results = {"model": scores(lo_hi, y, qm, quantiles),
               "climatology": scores(lo_hi, y, clim, quantiles),
               "naive + typical change": scores(lo_hi, y, last + dq[None], quantiles)}

    rows = []
    for name, sc in results.items():
        for h in (1, 10, 50, 100, 200, 500):
            rows.append({"band": name, "h": h, **{k: float(v[h - 1]) for k, v in sc.items()}})
        rows.append({"band": name, "h": "mean", **{k: float(v.mean()) for k, v in sc.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(run / "bands_vs_baselines.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))

    hh = np.arange(1, H + 1)
    colors = {"model": "tab:orange", "climatology": "tab:green", "naive + typical change": "tab:blue"}
    lvls = [round((quantiles[hi] - quantiles[lo]) * 100) for lo, hi in lo_hi]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for name, sc in results.items():
        for lvl, ls in zip(lvls, ("-", "--")):
            axes[0].plot(hh, sc[f"coverage{lvl}"], color=colors[name], ls=ls,
                         label=f"{name}, {lvl}% band")
            axes[1].plot(hh, sc[f"width{lvl}"], color=colors[name], ls=ls)
        axes[2].plot(hh, sc["pinball"], color=colors[name], label=name)
    for lvl in lvls:
        axes[0].axhline(lvl / 100, color="k", lw=0.6, ls=":")
    axes[0].set(title="Share of real values inside the band\n(dotted = what it should be)",
                xlabel="horizon h [ms]", ylim=(0, 1))
    axes[1].set(title="Band width [Cp] (narrower = more useful)", xlabel="horizon h [ms]")
    axes[2].set(title="Mean pinball loss [Cp] (lower = better)", xlabel="horizon h [ms]")
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")
    fig.suptitle(f"Uncertainty bands of {run.name} vs. model-free bands (test)")
    fig.tight_layout()
    fig.savefig(run / "bands_vs_baselines.png", dpi=130)
    print(f"\nSaved {run / 'bands_vs_baselines.png'}")


if __name__ == "__main__":
    main()

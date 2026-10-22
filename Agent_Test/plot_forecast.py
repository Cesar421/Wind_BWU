"""
Real windward Cp time series vs. the forecasts of a trained GPU-plan run.

Uses the run's saved test forecasts (runs/<run>/big/: preds, series_id,
target_start) and the raw series, so no model or GPU is needed.

  forecast_continuous.png  the whole test segment of each series (plus 2 s of
                           validation as lead-in), real signal in black and the
                           model's 500 ms forecasts chained every 500 ms
  forecast_detail.png      single forecasts with the full look-back the model saw

Usage (from Agent_Test/)
-----
    python plot_forecast.py                                  # run 4a, default series
    python plot_forecast.py --run 1c_s2000_p128-64_mse_r2 --series Alpha1_4/2_1_3/0 Alpha1_6/3_1_3/45
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
FS = 1000.0
H = 500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="4a_s2000_p128-64_mse_r3")
    ap.add_argument("--series", nargs="+",
                    default=["Alpha1_4/2_1_3/0", "Alpha1_4/1_1_3/45", "Alpha1_6/3_1_3/90"],
                    help="alpha/ratio/angle")
    ap.add_argument("--lead", type=float, default=2.0, help="seconds of real signal before the test segment")
    ap.add_argument("--compare", default=None,
                    help="second run drawn on the same windows (e.g. 4a_s2000_p128-64_mse_r3)")
    args = ap.parse_args()

    run = RUNS / args.run
    preds = np.load(run / "big" / "preds.npy", mmap_mode="r")
    sid = np.load(run / "big" / "series_id.npy")
    t0s = np.load(run / "big" / "target_start.npy")
    meta = pd.read_csv(run / "series_meta.csv")
    seq = int(args.run.split("_s")[1].split("_")[0])            # look-back in samples
    tag = args.run.split("_")[0]

    cmp_preds, cmp_at, cmp_tag, suffix = None, {}, None, ""
    if args.compare:
        c = RUNS / args.compare
        cmp_preds = np.load(c / "big" / "preds.npy", mmap_mode="r")
        cmp_at = {(s, t): i for i, (s, t) in enumerate(zip(np.load(c / "big" / "series_id.npy"),
                                                            np.load(c / "big" / "target_start.npy")))}
        cmp_tag = args.compare.split("_")[0]
        suffix = f"_vs_{cmp_tag}"

    def cmp_forecast(k):
        """The comparison run's forecast for the same window as index k, or None."""
        i = cmp_at.get((sid[k], t0s[k]))
        return None if i is None else cmp_preds[i]

    picked = []
    for s in args.series:
        alpha, ratio, angle = s.split("/")
        row = meta[(meta.alpha == alpha) & (meta.ratio == ratio) & (meta.angle == int(angle))]
        if row.empty:
            raise SystemExit(f"{s} is not in the test set of {args.run}")
        sid_s = int(row.series_id.iloc[0])
        idx = np.flatnonzero(sid == sid_s)
        y = load_faces(alpha, ratio, int(angle))[:, 0]              # raw windward Cp
        picked.append((s, idx, y))

    # ── 1) continuous view: forecasts chained every 500 ms over the test segment ──
    fig, axes = plt.subplots(len(picked), 1, figsize=(15, 3.4 * len(picked)), squeeze=False)
    for ax, (s, idx, y) in zip(axes[:, 0], picked):
        starts = t0s[idx]
        t_test = starts.min()
        chain = [k for k in idx if (t0s[k] - t_test) % H == 0]
        lo = int(t_test - args.lead * FS)
        t = np.arange(lo, len(y)) / FS
        ax.axvspan(lo / FS, t_test / FS, color="0.92", lw=0, label="before the test segment")
        ax.plot(t, y[lo:], color="0.2", lw=0.7, label="real Cp")
        err, cerr = [], []
        for j, k in enumerate(chain):
            tk = np.arange(t0s[k], t0s[k] + H) / FS
            cf = cmp_forecast(k) if args.compare else None
            if cf is not None:
                ax.plot(tk, cf, color="tab:blue", lw=1.4,
                        label=f"{cmp_tag} forecast" if j == 0 else None)
                cerr.append(cf - y[t0s[k]: t0s[k] + H])
            ax.plot(tk, preds[k], color="tab:orange", lw=1.8,
                    label=(f"{tag} forecast" if args.compare else
                           "forecast (500 ms ahead, from real data up to its start)") if j == 0 else None)
            ax.axvline(t0s[k] / FS, color="tab:orange", lw=0.5, ls=":")
            err.append(preds[k] - y[t0s[k]: t0s[k] + H])
        rmse = float(np.sqrt(np.mean(np.concatenate(err) ** 2)))
        score = f"RMSE {rmse:.3f}"
        if cerr:
            score = (f"RMSE {tag} {rmse:.3f} vs {cmp_tag} "
                     f"{float(np.sqrt(np.mean(np.concatenate(cerr) ** 2))):.3f}")
        ax.set_title(f"{s}   —   {len(chain)} forecasts, each made from the previous "
                     f"{seq / FS:g} s of real signal   ({score})", fontsize=10)
        ax.set_ylabel("Cp windward")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper left", fontsize=8, ncol=3)
    axes[-1, 0].set_xlabel("time [s]")
    fig.suptitle(f"Real time series vs. forecasts — {args.run}"
                 + (f" vs {args.compare}" if args.compare else "")
                 + " (test data, never seen in training)", fontsize=11)
    fig.tight_layout()
    fig.savefig(run / f"forecast_continuous{suffix}.png", dpi=130)
    plt.close(fig)

    # ── 2) detail: one forecast per series with the full look-back ──
    fig, axes = plt.subplots(len(picked), 1, figsize=(15, 3.4 * len(picked)), squeeze=False)
    for ax, (s, idx, y) in zip(axes[:, 0], picked):
        k = idx[len(idx) // 2]
        t0 = int(t0s[k])
        t = (np.arange(t0 - seq, t0 + H) - t0) / FS
        ax.axvspan(-seq / FS, 0, color="tab:green", alpha=0.08, lw=0,
                   label=f"look-back the model sees ({seq / FS:g} s)")
        ax.plot(t, y[t0 - seq: t0 + H], color="0.2", lw=0.7, label="real Cp")
        ax.hlines(y[t0 - 1], 0, (H - 1) / FS, color="0.5", ls="--", lw=1,
                  label="naive (last value)")
        cf = cmp_forecast(k) if args.compare else None
        if cf is not None:
            ax.plot(np.arange(H) / FS, cf, color="tab:blue", lw=1.4, label=f"{cmp_tag} forecast")
        ax.plot(np.arange(H) / FS, preds[k], color="tab:orange", lw=1.8,
                label=f"{tag} forecast" if args.compare else "forecast")
        ax.axvline(0, color="k", lw=0.6)
        rmse = float(np.sqrt(np.mean((preds[k] - y[t0: t0 + H]) ** 2)))
        score = f"RMSE {rmse:.3f}"
        if cf is not None:
            score = (f"RMSE {tag} {rmse:.3f} vs {cmp_tag} "
                     f"{float(np.sqrt(np.mean((cf - y[t0: t0 + H]) ** 2))):.3f}")
        ax.set_title(f"{s}   forecast made at t = {t0 / FS:.2f} s   ({score})", fontsize=10)
        ax.set_ylabel("Cp windward")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper left", fontsize=8, ncol=4)
    axes[-1, 0].set_xlabel("time relative to the forecast start [s]")
    fig.suptitle(f"What the model sees and what it predicts — {args.run}"
                 + (f" vs {args.compare}" if args.compare else ""), fontsize=11)
    fig.tight_layout()
    fig.savefig(run / f"forecast_detail{suffix}.png", dpi=130)
    plt.close(fig)
    print(f"Saved {run / f'forecast_continuous{suffix}.png'}\n      {run / f'forecast_detail{suffix}.png'}")


if __name__ == "__main__":
    main()

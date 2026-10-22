"""
Phase 0 of the GPU plan (Handoffs/2026-10-21): stationarity of the windward Cp
in 10 s blocks + per-series statistics on the TRAIN portion only. CPU, minutes.

For every (alpha, ratio, angle) series, raw windward Cp, fs = 1 kHz:
  - blocks 0-10, 10-20, 20-30 s: mean, std, skewness, kurtosis (Gaussian = 3),
    band power (<2, 2-10, 10-50, >50 Hz; Welch nperseg=4096, df=0.24 Hz) and
    spectral peak;
  - the same statistics on the train portion [0, 22.94 s) -> Phase 3 input.

If the signal is stationary, the blocks of a series agree (no drift between
block 1 and block 3), which justifies per-series rather than per-block
statistics and rules out "train on the last 10 s" (those hold all of val+test).

Outputs (results/gpu_plan_2026-10/fase0_estacionariedad/):
  stationarity_blocks.csv   one row per series x block
  series_stats_train.csv    one row per series (train portion only)
  stationarity_summary.csv  between-block variation per statistic
  stationarity_blocks.png   between-block variation (thesis figure)
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import kurtosis, skew, wilcoxon

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from data_loader import list_series, load_faces  # noqa: E402

OUT = ROOT / "results" / "gpu_plan_2026-10" / "fase0_estacionariedad"
FS = 1000.0
BLOCK = 10_000                     # samples = 10 s
N_BLOCKS = 3
TRAIN_R = 0.70
NPERSEG = 4096
BANDS = [("P_lt2", 0, 2), ("P_2_10", 2, 10), ("P_10_50", 10, 50), ("P_gt50", 50, FS)]
# statistics compared across blocks: (column, relative?, label)
COMPARE = [("mean", False, "mean  [Cp]"),
           ("std", True, "std  [%]"),
           ("skew", False, "skewness"),
           ("kurt", False, "kurtosis"),
           ("P_total", True, "AC power  [%]"),
           ("frac_2_10", False, "2-10 Hz share  [pp]")]


def stats(x: np.ndarray) -> dict:
    f, p = welch(x, fs=FS, nperseg=NPERSEG, noverlap=NPERSEG // 2,
                 window="hann", detrend="constant")
    df = f[1] - f[0]
    ac = f > 0
    out = {"mean": float(x.mean()), "std": float(x.std()),
           "skew": float(skew(x)), "kurt": float(kurtosis(x, fisher=False)),
           "P_total": float(p[ac].sum() * df)}
    for name, lo, hi in BANDS:
        out[name] = float(p[ac & (f >= lo) & (f < hi)].sum() * df)
    for name, _, _ in BANDS:
        out["frac" + name[1:]] = out[name] / out["P_total"]
    f_peak = float(f[ac][np.argmax(p[ac])])
    out["f_peak_hz"] = f_peak
    out["peak_period_ms"] = 1000.0 / f_peak
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    series = list_series()
    print(f"Phase 0: {len(series)} series, {N_BLOCKS} blocks of {BLOCK / FS:.0f} s "
          f"(Welch nperseg={NPERSEG}, df={FS / NPERSEG:.2f} Hz)")

    block_rows, train_rows = [], []
    for sid, (alpha, ratio, angle) in enumerate(series):
        w = load_faces(alpha, ratio, angle)[:, 0].astype(np.float64)   # windward
        key = {"series_id": sid, "alpha": alpha, "ratio": ratio, "angle": angle}
        for b in range(N_BLOCKS):
            block_rows.append({**key, "block": b + 1,
                               "t_start_s": b * BLOCK / FS, "t_end_s": (b + 1) * BLOCK / FS,
                               **stats(w[b * BLOCK:(b + 1) * BLOCK])})
        t1 = int(len(w) * TRAIN_R)
        train_rows.append({**key, "t_end_s": t1 / FS, **stats(w[:t1])})

    blocks = pd.DataFrame(block_rows)
    blocks.to_csv(OUT / "stationarity_blocks.csv", index=False)
    pd.DataFrame(train_rows).to_csv(OUT / "series_stats_train.csv", index=False)

    # ── Between-block variation: deviation of each block from its series' 3-block mean ──
    dev = {}
    summary = []
    for col, rel, label in COMPARE:
        wide = blocks.pivot(index="series_id", columns="block", values=col)
        centre = wide.mean(axis=1)
        d = wide.sub(centre, axis=0)
        if rel:
            d = d.div(centre, axis=0) * 100
        elif col.startswith("frac"):
            d = d * 100
        dev[col] = d
        spread = d.max(axis=1) - d.min(axis=1)
        monotonic = ((wide[1] < wide[2]) & (wide[2] < wide[3])) | \
                    ((wide[1] > wide[2]) & (wide[2] > wide[3]))
        summary.append({
            "statistic": col, "unit": label.split("[")[-1].rstrip("]") if "[" in label else "",
            "median_abs_dev": float(d.abs().median().median()),
            "p95_abs_dev": float(np.percentile(d.abs().values, 95)),
            "median_spread_b_max_min": float(spread.median()),
            "median_dev_block1": float(d[1].median()),
            "median_dev_block3": float(d[3].median()),
            "frac_monotonic": float(monotonic.mean()),          # 1/3 expected without trend
            "wilcoxon_p_block1_vs_3": float(wilcoxon(wide[1], wide[3]).pvalue),
        })
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "stationarity_summary.csv", index=False)
    pd.set_option("display.width", 160)
    print("\nDeviation of each 10 s block from its series' mean "
          "(relative % for std and power, absolute otherwise):")
    print(summ.round(4).to_string(index=False))
    print("frac_monotonic: share of series with block1<block2<block3 or the reverse "
          "(1/3 expected if there is no trend)")

    peak_same = blocks.pivot(index="series_id", columns="block", values="f_peak_hz") \
                      .nunique(axis=1).eq(1).mean()
    print(f"Spectral peak in the same Welch bin in all 3 blocks: {peak_same:.0%} of series")

    # ── Figure ──
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, (col, rel, label) in zip(axes.flat, COMPARE):
        d = dev[col]
        ax.boxplot([d[b].values for b in range(1, N_BLOCKS + 1)], whis=(5, 95),
                   showfliers=False, widths=0.55,
                   medianprops=dict(color="tab:orange", lw=1.6))
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(1, N_BLOCKS + 1),
                      [f"{b * 10}-{(b + 1) * 10} s" for b in range(N_BLOCKS)])
        s = summ.set_index("statistic").loc[col]
        ax.set_title(f"{label.split('[')[0].strip()}: median |dev| "
                     f"{s.median_abs_dev:.3g}{'%' if rel else ''}", fontsize=10)
        ax.set_ylabel(f"block - series mean  [{label.split('[')[-1]}" if "[" in label
                      else "block - series mean")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Windward Cp, {len(series)} series: deviation of each 10 s block from "
                 f"its series' mean (box = 25-75 %, whiskers = 5-95 %)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "stationarity_blocks.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()

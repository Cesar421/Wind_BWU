"""
Plot RMSE-vs-horizon comparison: Direct multi-step LSTM vs autoregressive models.

Pulls per-horizon RMSE from:
  - results/multi_horizon_metrics_round3.csv  (autoregressive models)
  - results/lstm_direct_metrics.csv           (direct LSTM, inference rows)
  - results/lstm_direct_h500_rmse_curve.npy   (full 500-step curve for direct)

Saves: results/plots_cross_round/direct_vs_autoreg_h500.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = RES / "plots_cross_round"
OUT.mkdir(parents=True, exist_ok=True)

autoreg = pd.read_csv(RES / "multi_horizon_metrics_round3.csv")
# lstm_direct_metrics.csv has heterogeneous column count (inference rows added
# a 'source' column later). Read robustly by enumerating manually.
direct_rows = []
with open(RES / "lstm_direct_metrics.csv", "r", encoding="utf-8") as f:
    header = f.readline().rstrip("\n").split(",")
    for line in f:
        parts = line.rstrip("\n").split(",")
        if len(parts) < 5:
            continue
        direct_rows.append({
            "model": parts[0],
            "horizon": int(parts[1]),
            "rmse": float(parts[2]),
            "mae": float(parts[3]),
            "r2": float(parts[4]),
            "scope": parts[8] if len(parts) > 8 else "",
            "source": parts[9] if len(parts) > 9 else "",
        })
direct = pd.DataFrame(direct_rows)
direct = direct[(direct["model"] == "lstm_direct_h500") & (direct["source"] == "inference_only")]
print(f"Direct rows for plot: {len(direct)}  horizons={sorted(direct['horizon'].tolist())}")
rmse_curve = np.load(RES / "lstm_direct_h500_rmse_curve.npy")  # (500,)

models_show = ["naive_persistence", "xgboost", "lstm", "tcn"]
colors = {"naive_persistence": "tab:gray", "xgboost": "tab:green",
          "lstm": "tab:blue", "tcn": "tab:red", "lstm_direct": "#d62728"}
colors["lstm_direct"] = "#9400d3"  # vivid purple
labels = {"naive_persistence": "Naive persistence",
          "xgboost": "XGBoost (autoreg)",
          "lstm": "LSTM (autoreg)",
          "tcn": "TCN (autoreg)",
          "lstm_direct": "LSTM direct multi-step (NEW)"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: linear scale, full picture (includes TCN catastrophe)
for m in models_show:
    df = autoreg[autoreg["model"] == m].sort_values("horizon")
    ax1.plot(df["horizon"], df["rmse"], "o-", color=colors[m],
             label=labels[m], linewidth=2, markersize=7)
direct_pts = direct.sort_values("horizon")
ax1.plot(np.arange(1, 501), rmse_curve, "-", color=colors["lstm_direct"],
         alpha=0.55, linewidth=2, zorder=4)
ax1.plot(direct_pts["horizon"], direct_pts["rmse"], "s", color=colors["lstm_direct"],
         label=labels["lstm_direct"], markersize=11, markeredgecolor="black",
         markeredgewidth=1.2, zorder=5)
ax1.set_xlabel("Forecast horizon h (steps)", fontsize=11)
ax1.set_ylabel("RMSE (denormalised $C_p$)", fontsize=11)
ax1.set_title("Round 3 — Long-horizon RMSE\n(linear scale)", fontsize=12)
ax1.legend(loc="upper left", fontsize=9, framealpha=0.92)
ax1.grid(alpha=0.3)
ax1.set_xlim(0, 510)

# Right: log scale, zooms on the contenders
for m in models_show:
    df = autoreg[autoreg["model"] == m].sort_values("horizon")
    ax2.semilogy(df["horizon"], df["rmse"], "o-", color=colors[m],
                 label=labels[m], linewidth=2, markersize=7)
ax2.semilogy(np.arange(1, 501), rmse_curve, "-", color=colors["lstm_direct"],
             alpha=0.55, linewidth=2, zorder=4)
ax2.semilogy(direct_pts["horizon"], direct_pts["rmse"], "s",
             color=colors["lstm_direct"], label=labels["lstm_direct"],
             markersize=11, markeredgecolor="black", markeredgewidth=1.2, zorder=5)
ax2.set_xlabel("Forecast horizon h (steps)", fontsize=11)
ax2.set_ylabel("RMSE (log scale)", fontsize=11)
ax2.set_title("Round 3 — Long-horizon RMSE\n(log scale)", fontsize=12)
ax2.legend(loc="lower right", fontsize=9, framealpha=0.92)
ax2.grid(alpha=0.3, which="both")
ax2.set_xlim(0, 510)

fig.suptitle("Direct multi-step vs autoregressive — wind $C_p$ forecasting",
             fontsize=13, fontweight="bold")
fig.tight_layout()
out_path = OUT / "direct_vs_autoreg_h500.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")

# Annotated summary
print("\n-- RMSE @ h=500 (Round 3) --")
for m in models_show:
    v = autoreg[(autoreg["model"] == m) & (autoreg["horizon"] == 500)]["rmse"].values
    if len(v):
        print(f"  {labels[m]:<28s} {v[0]:.4f}")
vd = direct[direct["horizon"] == 500]["rmse"].values
if len(vd):
    print(f"  {labels['lstm_direct']:<28s} {vd[0]:.4f}")

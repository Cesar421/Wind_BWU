"""
Master Training Script — Runs ALL Round 1 models sequentially.
==============================================================

Round 1 set (per MODELING_PLAN.md): Ridge, RandomForest, XGBoost, LSTM, GRU, TCN.
All trained on the SAME single building (default: Alpha1_4 / 2_1_3) to ensure
a fair comparison: same data, same split, same normalisation, same seed.

Usage:
    cd Agent_Test
    python train_all.py                              # default building 2_1_3
    python train_all.py --building Alpha1_4/2_1_3    # explicit
    python train_all.py --building all               # Round 2/3 (all buildings)
"""

import argparse
import subprocess
import sys
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


AGENT_TEST = Path(__file__).resolve().parent

# Round 1 models, in execution order (fast → slow).
MODELS = [
    ("ridge",         "train_ridge.py"),
    ("random_forest", "train_random_forest.py"),
    ("xgboost",       "train_xgboost.py"),
    ("lstm",          "train_lstm.py"),
    ("gru",           "train_gru.py"),
    ("tcn",           "train_tcn.py"),
]


def run_all(scope: str):
    print("=" * 70)
    print("  MASTER TRAINING")
    print(f"  Scope: {scope}")
    print("=" * 70)

    for model_name, script in MODELS:
        script_path = AGENT_TEST / model_name / script
        if not script_path.exists():
            print(f"\n  SKIP: {script_path} not found")
            continue
        print(f"\n{'─' * 70}")
        print(f"  Running {model_name} ...")
        print(f"{'─' * 70}")
        cmd = [sys.executable, str(script_path), "--scope", scope]
        result = subprocess.run(cmd, cwd=str(AGENT_TEST / model_name))
        if result.returncode != 0:
            print(f"  WARNING: {model_name} exited with code {result.returncode}")

    consolidate_results()


def consolidate_results():
    """Merge per-model CSVs into a single consolidated comparison."""
    all_rows = []
    for model_name, _ in MODELS:
        csv_path = AGENT_TEST / model_name / "results" / "model_comparison.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                all_rows.append(row)

    if not all_rows:
        print("\nNo per-model results found. Nothing to consolidate.")
        return

    out_dir = AGENT_TEST / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "model_comparison.csv"

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nConsolidated results: {out_csv}")

    plot_comparison(all_rows, out_dir / "plots")


def plot_comparison(rows, plots_dir):
    """Generate bar charts comparing all models across key metrics."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    models = [r["model"] for r in rows]
    metric_keys = ["rmse", "mae", "r2", "mape"]
    metric_vals = {k: [float(r.get(k, 0)) for r in rows] for k in metric_keys}
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    # ── 4-panel metric comparison ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, key in zip(axes.flat, metric_keys):
        vals = metric_vals[key]
        bars = ax.bar(models, vals, color=colors)
        ax.set_title(key.upper(), fontsize=14, fontweight="bold")
        ax.set_ylabel(key.upper())
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Model Comparison — All Metrics", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "model_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Comparison plot: {plots_dir / 'model_comparison.png'}")

    # ── Training time & parameter count ──
    times = [float(r.get("train_time_s", 0)) for r in rows]
    params = [int(r.get("parameters", 0)) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(models, times, color=colors)
    ax1.set_title("Training Time (s)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Seconds"); ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(ax1.patches, times):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    ax2.bar(models, params, color=colors)
    ax2.set_title("Parameter Count", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Parameters"); ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(ax2.patches, params):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:,}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Model Efficiency", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "model_efficiency.png", dpi=150)
    plt.close(fig)
    print(f"Efficiency plot:  {plots_dir / 'model_efficiency.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", type=str, default="Alpha1_4/2_1_3",
                        help='"Alpha1_4/2_1_3" (Round 1) | "round2" | "all"')
    args = parser.parse_args()
    run_all(args.scope)

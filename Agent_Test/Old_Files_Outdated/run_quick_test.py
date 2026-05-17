"""
Quick Test Runner - Single building (Alpha1_4/2_1_3)
=====================================================
Runs all 5 models on one building to verify they work
and get a rough performance comparison.
"""
import subprocess
import sys
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AGENT_TEST = Path(__file__).resolve().parent
BUILDING = "Alpha1_4/2_1_3"
LOG_FILE = AGENT_TEST / "quick_test_log.txt"

MODELS = [
    ("cnn_lstm",    "train_cnn_lstm.py"),
    ("lstm",        "train_lstm.py"),
    ("tcn",         "train_tcn.py"),
    ("transformer", "train_transformer.py"),
    ("ann",         "train_ann.py"),
]


def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)


def run_all():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"Quick test started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Building: {BUILDING}\n\n")

    log("=" * 70)
    log(f"  QUICK TEST - {BUILDING}")
    log("=" * 70)

    for model_name, script in MODELS:
        script_path = AGENT_TEST / model_name / script
        if not script_path.exists():
            log(f"\n  SKIP: {script_path} not found")
            continue
        log(f"\n{'-' * 70}")
        log(f"  Starting {model_name} at {time.strftime('%H:%M:%S')} ...")
        log(f"{'-' * 70}")

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(script_path), "--building", BUILDING],
            cwd=str(AGENT_TEST / model_name),
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - t0

        for line in result.stdout.splitlines():
            log(line)
        if result.stderr:
            for line in result.stderr.splitlines():
                log(f"  [STDERR] {line}")

        if result.returncode != 0:
            log(f"  FAILED: {model_name} exited with code {result.returncode}")
        else:
            log(f"  {model_name} completed in {elapsed:.1f}s")

    log(f"\n{'=' * 70}")
    log(f"  All models finished at {time.strftime('%H:%M:%S')}")
    log(f"{'=' * 70}")

    consolidate_results()


def consolidate_results():
    all_rows = []
    for model_name, _ in MODELS:
        csv_path = AGENT_TEST / model_name / "results" / "model_comparison.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, "r") as f:
            for row in csv.DictReader(f):
                all_rows.append(row)

    if not all_rows:
        log("\nNo per-model results found.")
        return

    out_dir = AGENT_TEST / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "model_comparison.csv"

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    log(f"\nConsolidated: {out_csv}")

    # Print summary table
    log(f"\n{'=' * 90}")
    log(f"  MODEL COMPARISON - {BUILDING}")
    log(f"{'=' * 90}")
    log(f"  {'Model':<15} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'MAPE':>8} {'Params':>10} {'Time(s)':>8}")
    log(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for r in all_rows:
        log(f"  {r['model']:<15} {float(r['rmse']):>8.4f} {float(r['mae']):>8.4f} "
            f"{float(r['r2']):>8.4f} {float(r['mape']):>8.2f} "
            f"{int(r['parameters']):>10,} {float(r['train_time_s']):>8.1f}")
    log(f"{'=' * 90}")

    plot_comparison(all_rows, out_dir / "plots")


def plot_comparison(rows, plots_dir):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    models = [r["model"] for r in rows]
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    metric_keys = ["rmse", "mae", "r2", "mape"]
    metric_vals = {k: [float(r.get(k, 0)) for r in rows] for k in metric_keys}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, key in zip(axes.flat, metric_keys):
        vals = metric_vals[key]
        bars = ax.bar(models, vals, color=colors)
        ax.set_title(key.upper(), fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"Quick Test - {BUILDING}", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "quick_test_comparison.png", dpi=150)
    plt.close(fig)
    log(f"Plot: {plots_dir / 'quick_test_comparison.png'}")


if __name__ == "__main__":
    run_all()

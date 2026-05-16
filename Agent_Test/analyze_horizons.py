"""
Round 1 — Multi-horizon forecast analysis
==========================================
Loads the saved multi-step forecast .npy files for every Round 1 model,
computes per-horizon metrics against the ground-truth windward Cp series,
adds a Naive-Persistence baseline, and writes:

    Agent_Test/results/multi_horizon_metrics.csv
    Agent_Test/results/plots/r2_vs_horizon.png
    Agent_Test/results/plots/rmse_vs_horizon.png

Assumption: every model used the SAME building (default Alpha1_4/2_1_3),
the SAME data_loader, the SAME seq_length and step → so the ground-truth
y_future is identical for all models, and we can recompute it once.
"""

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_single_building


AGENT_TEST = Path(__file__).resolve().parent
MODELS = ["ridge", "random_forest", "xgboost", "lstm", "gru", "tcn"]
HORIZONS = [1, 10, 50, 100, 500]

# Round 1 scope — must match what train_all.py was launched with
ALPHA = "Alpha1_4"
RATIO = "2_1_3"
SEQ_LENGTH = 100
STEP = 10
HORIZON_TRAIN = 1


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main():
    print("Recomputing ground truth (denormalised windward Cp)...")
    data = load_single_building(
        alpha=ALPHA, ratio=RATIO,
        seq_length=SEQ_LENGTH, step=STEP, horizon=HORIZON_TRAIN,
    )
    mu_w = float(data["mu"][0])
    sigma_w = float(data["sigma"][0])
    gt = data["y_future"] * sigma_w + mu_w  # length 500, denormalised
    test_seed_w = data["test_seed"][:, 0] * sigma_w + mu_w
    last_value = float(test_seed_w[-1])

    rows: List[Dict[str, str]] = []

    # ── Naive persistence baseline ──
    print("\nNaive Persistence baseline:")
    for h in HORIZONS:
        n = min(h, len(gt))
        pred = np.full(n, last_value, dtype=np.float32)
        m = metrics(gt[:n], pred)
        rows.append({"model": "naive_persistence", "horizon": h, **m})
        print(f"  h={h:>4d}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}")

    # ── Each trained model ──
    for model in MODELS:
        fc_dir = AGENT_TEST / model / "results" / "forecasts"
        if not fc_dir.exists():
            print(f"\n[skip] no forecasts for {model}")
            continue
        print(f"\n{model}:")
        for h in HORIZONS:
            fpath = fc_dir / f"{model}_h{h}.npy"
            if not fpath.exists():
                print(f"  h={h:>4d}  MISSING")
                continue
            pred = np.load(fpath)
            n = min(h, len(gt), len(pred))
            m = metrics(gt[:n], pred[:n])
            rows.append({"model": model, "horizon": h, **m})
            print(f"  h={h:>4d}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}")

    # ── Save consolidated CSV ──
    out_dir = AGENT_TEST / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "multi_horizon_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "horizon", "rmse", "mae", "r2"])
        w.writeheader()
        for r in rows:
            r2 = r.copy()
            for k in ("rmse", "mae", "r2"):
                r2[k] = round(float(r2[k]), 6)
            w.writerow(r2)
    print(f"\nSaved {csv_path}")

    # ── Plots ──
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _plot(metric_key: str, ylabel: str, fname: str, logy: bool = False):
        fig, ax = plt.subplots(figsize=(10, 6))
        names = sorted({r["model"] for r in rows})
        colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
        for name, c in zip(names, colors):
            xs = [r["horizon"] for r in rows if r["model"] == name]
            ys = [r[metric_key] for r in rows if r["model"] == name]
            ls = "--" if name == "naive_persistence" else "-"
            lw = 2.5 if name == "naive_persistence" else 1.8
            ax.plot(xs, ys, marker="o", linestyle=ls, linewidth=lw,
                    color=c, label=name)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Forecast horizon (steps)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs forecast horizon — Round 1 ({ALPHA}/{RATIO})")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        out = plots_dir / fname
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")

    _plot("r2", "R²", "r2_vs_horizon.png")
    _plot("rmse", "RMSE", "rmse_vs_horizon.png")


if __name__ == "__main__":
    main()

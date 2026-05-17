"""
Regenerate per-round and cross-round comparison plots from the preserved CSVs.

Reads:
    results/model_comparison_round{1,2,3}.csv
    results/multi_horizon_metrics_round{1,2,3}.csv

Writes:
    results/plots_round1/, plots_round2/, plots_round3/, plots_cross_round/

Only plots that depend on aggregated metrics can be regenerated.
Per-model pred-vs-actual scatters (which need the original predictions) cannot
be reconstructed for R1/R2 because the .pt checkpoints were overwritten.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"

MODELS = ["ridge", "random_forest", "xgboost", "lstm", "gru", "tcn"]
ROUNDS = [1, 2, 3]
ROUND_LABEL = {
    1: "R1: single building",
    2: "R2: 5 configs",
    3: "R3: all 20 configs",
}

COLORS = {
    "ridge": "#1f77b4",
    "random_forest": "#ff7f0e",
    "xgboost": "#2ca02c",
    "lstm": "#d62728",
    "gru": "#9467bd",
    "tcn": "#8c564b",
    "naive_persistence": "#7f7f7f",
}


def _ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_single_step_bars(rnd: int, df: pd.DataFrame, outdir: Path):
    """Bar chart of RMSE and R^2 at h=1 across models for a single round."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(MODELS))
    rmse = [df.loc[df.model == m, "rmse"].iloc[0] for m in MODELS]
    r2 = [df.loc[df.model == m, "r2"].iloc[0] for m in MODELS]
    bars0 = axes[0].bar(x, rmse, color=[COLORS[m] for m in MODELS])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(MODELS, rotation=25)
    axes[0].set_ylabel("RMSE (h=1, test)")
    axes[0].set_title(f"{ROUND_LABEL[rnd]} — single-step RMSE")
    for b, v in zip(bars0, rmse):
        axes[0].text(b.get_x() + b.get_width()/2, v, f"{v:.4f}",
                     ha="center", va="bottom", fontsize=8)
    bars1 = axes[1].bar(x, r2, color=[COLORS[m] for m in MODELS])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(MODELS, rotation=25)
    axes[1].set_ylabel("R² (h=1, test)")
    axes[1].set_title(f"{ROUND_LABEL[rnd]} — single-step R²")
    axes[1].set_ylim(min(0, min(r2) - 0.05), 1.01)
    for b, v in zip(bars1, r2):
        axes[1].text(b.get_x() + b.get_width()/2, v, f"{v:.4f}",
                     ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"single_step_bars_round{rnd}.png", dpi=140)
    plt.close(fig)


def plot_horizon_curves(rnd: int, mh: pd.DataFrame, outdir: Path):
    """RMSE and R² vs horizon for each model + naive baseline."""
    horizons = sorted(mh.horizon.unique())
    for metric, ylabel, fname in [
        ("rmse", "RMSE", f"rmse_vs_horizon_round{rnd}.png"),
        ("r2", "R²",   f"r2_vs_horizon_round{rnd}.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for m in MODELS + ["naive_persistence"]:
            sub = mh[mh.model == m].sort_values("horizon")
            if sub.empty:
                continue
            y = sub[metric].values.astype(float)
            ls = "--" if m == "naive_persistence" else "-"
            ax.plot(sub.horizon, y, marker="o", label=m, ls=ls,
                    color=COLORS.get(m, None))
        ax.set_xscale("log")
        ax.set_xlabel("Forecast horizon (steps)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ROUND_LABEL[rnd]} — {ylabel} vs horizon")
        # clip RMSE for visibility when Ridge diverges (R3 h=500)
        if metric == "rmse":
            finite = y[np.isfinite(y)]
            cap = max(1.0, float(np.nanmax(mh[mh.model.isin(MODELS + ['naive_persistence'])][metric].replace([np.inf, -np.inf], np.nan).dropna())))
            cap = min(cap, 2.0)  # never let one outlier dominate the figure
            ax.set_ylim(0, cap)
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=140)
        plt.close(fig)


def plot_cross_round(comps: dict, outdir: Path):
    """Grouped bar: RMSE at h=1 per model, one bar per round."""
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    x = np.arange(len(MODELS))
    for i, rnd in enumerate(ROUNDS):
        df = comps[rnd]
        vals = [df.loc[df.model == m, "rmse"].iloc[0] for m in MODELS]
        ax.bar(x + (i - 1) * width, vals, width, label=f"Round {rnd}")
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 1) * width, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=20)
    ax.set_ylabel("RMSE (h=1, test)")
    ax.set_title("Cross-round single-step RMSE")
    ax.grid(True, axis="y", ls=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "cross_round_rmse_h1.png", dpi=140)
    plt.close(fig)


def plot_cross_round_horizon(mhs: dict, outdir: Path, model: str):
    """RMSE vs horizon, one curve per round, for a chosen model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for rnd in ROUNDS:
        sub = mhs[rnd][mhs[rnd].model == model].sort_values("horizon")
        if sub.empty:
            continue
        ax.plot(sub.horizon, sub.rmse, marker="o", label=f"Round {rnd}")
    ax.set_xscale("log")
    ax.set_xlabel("Forecast horizon (steps)")
    ax.set_ylabel("RMSE")
    ax.set_title(f"{model}: RMSE vs horizon across rounds")
    ax.set_ylim(0, min(2.0, ax.get_ylim()[1]))
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"cross_round_horizon_{model}.png", dpi=140)
    plt.close(fig)


def main():
    comps = {r: pd.read_csv(RES / f"model_comparison_round{r}.csv") for r in ROUNDS}
    mhs = {r: pd.read_csv(RES / f"multi_horizon_metrics_round{r}.csv") for r in ROUNDS}

    for r in ROUNDS:
        outdir = _ensure(RES / f"plots_round{r}")
        plot_single_step_bars(r, comps[r], outdir)
        plot_horizon_curves(r, mhs[r], outdir)
        print(f"Round {r} plots -> {outdir}")

    cross = _ensure(RES / "plots_cross_round")
    plot_cross_round(comps, cross)
    for m in MODELS:
        plot_cross_round_horizon(mhs, cross, m)
    print(f"Cross-round plots -> {cross}")


if __name__ == "__main__":
    main()

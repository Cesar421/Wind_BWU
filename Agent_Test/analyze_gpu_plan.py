"""
Analysis of the GPU plan runs (Handoffs/2026-10-21, step 7): results table,
Pareto curves (accuracy vs spectral power), look-back comparison, PSD of the
predictions, skill of the variance added by StatLoss and the Phase 2 lambda rule.

Roles from the run tags: 1a/1b/1c = Phase 1 (look-back, MSE), 2a..2z = Phase 2
(StatLoss with SEQ*), other 2* tags (e.g. 2c-legacy) = Phase 2 controls,
1a-orig = patch-alignment control, 4* / 5* = Phases 4 / 5 (full scope), naive.
SEQ* = Phase 1 run with the lowest best val MSE (same rule as run_gpu_plan.ps1).

Accuracy is shown both as RMSE averaged over h=1..500 (the whole forecast; it
catches artefacts at any horizon, like the h~240 template of protocol A) and
as RMSE@500 (the plan's original criterion). Skill of the added variance:
corr(pred - pred_SEQ*, true - pred_SEQ*) over all test windows and horizons —
0 means the extra variability is unrelated to what the MSE model gets wrong.

Usage (from Agent_Test/)
-----
    python analyze_gpu_plan.py
    python analyze_gpu_plan.py --root results/gpu_plan_2026-10/protocolo_A

Outputs -> <root>/analisis/
-------
    tabla_resultados.csv  pareto_rmse_vs_potencia.png  pareto_rmse500.png
    fase1_ventana.png  psd_predicciones.png  forecast_comparison.png
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AGENT = Path(__file__).resolve().parent
EVAL_HS = [1, 10, 50, 100, 500]
ROLE_ORDER = {"naive": 0, "phase1": 1, "phase2": 2, "phase2-control": 3, "control": 4,
              "phase4": 5, "phase5": 6, "other": 7}


def role(tag: str) -> str:
    if tag == "naive":
        return "naive"
    if re.fullmatch(r"1[a-c]", tag):
        return "phase1"
    if tag == "1a-orig":
        return "control"
    if re.fullmatch(r"2[a-z]", tag):
        return "phase2"
    for prefix, r in (("2", "phase2-control"), ("4", "phase4"), ("5", "phase5")):
        if tag.startswith(prefix):
            return r
    return "other"


def load_runs(root: Path) -> dict:
    runs = {}
    for f in sorted((root / "runs").glob("*/summary.json")):
        with open(f, encoding="utf-8") as fh:
            s = json.load(fh)
        s["dir"], s["role"] = f.parent, role(s["tag"])
        # summaries written before these scores existed (protocol A)
        curve = f.parent / "rmse_curve.npy"
        if "rmse_mean" not in s["test"] and curve.exists():
            c = np.load(curve)
            s["test"].update(rmse_mean=float(c.mean()), rmse_max=float(c.max()),
                             h_rmse_max=int(c.argmax()) + 1)
        for split in ("val", "test"):
            for k in ("rmse_mean", "rmse_max", "h_rmse_max", "tpl_share", "pc1_share"):
                s[split].setdefault(k, float("nan"))
        runs[s["run"]] = s
    return runs


def label(s: dict) -> str:
    if s["role"] == "naive":
        return s["run"]
    if s.get("lam_psd") or s.get("lam_std"):
        txt = f"{s['tag']} psd={s['lam_psd']:g} std={s['lam_std']:g}"
        if s.get("psd_bins") == "linear":
            txt += " lin"
        if s.get("statloss") == "legacy":
            txt += " legacy"
        return txt
    return f"{s['tag']} s{s['seq']}"


def skill(run: dict, ref: dict):
    """corr(pred - ref_pred, true - ref_pred) over all test windows and horizons."""
    files = [run["dir"] / "big" / "preds.npy", ref["dir"] / "big" / "preds.npy",
             ref["dir"] / "big" / "trues.npy"]
    if not all(f.exists() for f in files):
        return float("nan")
    p, m, y = (np.load(f).astype(np.float64) for f in files)
    a, r = p - m, y - m
    return float((a * r).sum() / np.sqrt((a * a).sum() * (r * r).sum()))


def results_table(runs: dict, ref: dict) -> pd.DataFrame:
    rows = []
    for s in runs.values():
        v, t = s["val"], s["test"]
        rows.append({
            "run": s["run"], "role": s["role"], "tag": s["tag"], "scope": s.get("scope"),
            "seq": s.get("seq"), "patch": (f"{s['patch_len']}/{s['patch_stride']}"
                                           if s.get("patch_len") else ""),
            "lam_psd": s.get("lam_psd"), "lam_std": s.get("lam_std"),
            "statloss": s.get("statloss", ""), "epochs": s.get("epochs_run"),
            "best_epoch": s.get("best_epoch"),
            "train_min": (s["train_time_s"] / 60 if s.get("train_time_s") else None),
            "best_val_mse": s.get("best_val_mse"),
            "val_rmse_mean": v["rmse_mean"], "val_rmse500": v["rmse_h500"],
            **{f"test_rmse{h}": t[f"rmse_h{h}"] for h in EVAL_HS},
            "test_rmse_mean": t["rmse_mean"], "test_rmse_max": t["rmse_max"],
            "h_rmse_max": t["h_rmse_max"], "test_r2_500": t["r2_h500"],
            "power_total": t["power_ratio_total"], "power_2_10": t["power_ratio_2_10"],
            "std_ratio": t["std_ratio"], "tpl_share": t["tpl_share"],
            "pc1_share": t["pc1_share"],
            "skill_vs_seqstar": (skill(s, ref) if ref is not None and s["role"].startswith("phase2")
                                 else float("nan")),
        })
    df = pd.DataFrame(rows)
    df["_o"] = df["role"].map(ROLE_ORDER)
    return df.sort_values(["_o", "scope", "tag"]).drop(columns="_o").reset_index(drop=True)


def print_markdown(df: pd.DataFrame, cols: list, fmt: dict):
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            x = r[c]
            cells.append("" if x is None or (isinstance(x, float) and np.isnan(x))
                         else fmt.get(c, "{}").format(x))
        print("| " + " | ".join(cells) + " |")


def pareto(path: Path, key: str, xlabel: str, naive, seq_star, p1, p2, controls, chosen):
    """Accuracy (key) vs total power ratio, val (decision) and test (report)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    cmap = plt.get_cmap("viridis")
    lam_vals = sorted({s["lam_psd"] for s in p2})

    def xy(s, split):
        return s[split][key], s[split]["power_ratio_total"]

    for ax, split in zip(axes, ("val", "test")):
        trace = ([seq_star] if seq_star else []) + [s for s in p2 if s.get("psd_bins") != "linear"]
        if len(trace) > 1:
            ax.plot(*zip(*[xy(s, split) for s in trace]), color="0.7", lw=1, zorder=1)
        for i, s in enumerate(p1 + controls):
            star = seq_star is not None and s["run"] == seq_star["run"]
            ax.scatter(*xy(s, split), s=60, marker="o",
                       facecolor="tab:blue" if star else "none",
                       edgecolor="tab:blue" if s["role"] == "phase1" else "tab:gray", zorder=3)
            ax.annotate(label(s) + (" (SEQ*)" if star else ""), xy(s, split),
                        textcoords="offset points", xytext=(6, (-12, 6, -24, 18)[i % 4]),
                        fontsize=8)
        for s in p2:
            c = cmap(lam_vals.index(s["lam_psd"]) / max(1, len(lam_vals) - 1))
            ax.scatter(*xy(s, split), s=70, marker="s" if s.get("psd_bins") != "linear" else "D",
                       color=c, edgecolor="k", lw=0.5, zorder=3)
            ax.annotate(label(s), xy(s, split), textcoords="offset points", xytext=(6, 4),
                        fontsize=8)
        if chosen is not None:
            ax.scatter(*xy(chosen, split), s=260, marker="*", facecolor="none",
                       edgecolor="crimson", lw=1.4, zorder=4, label="plan's lambda rule")
        if naive:
            ax.scatter(naive[split][key], 0, s=70, marker="X", color="k", zorder=3)
            ax.annotate("naive", (naive[split][key], 0), textcoords="offset points",
                        xytext=(6, 4), fontsize=8)
        ax.axhline(1.0, color="tab:green", ls="--", lw=1, label="true signal power")
        ax.set(xlabel=f"{split} {xlabel}", title=split)
        ax.grid(alpha=0.3)
        ax.legend(loc="center right", fontsize=8)
    axes[0].set_ylabel("power ratio  P_pred / P_true  (2-500 Hz)")
    fig.suptitle(f"Pareto: {xlabel} vs. spectral power of the forecast "
                 "(choose on val, report on test)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(AGENT / "results" / "gpu_plan_2026-10"))
    root = Path(ap.parse_args().root)
    out = root / "analisis"
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(root)
    r2 = [s for s in runs.values() if s.get("scope") == "round2"]
    by_role = {r: [s for s in r2 if s["role"] == r] for r in ROLE_ORDER}
    naive = next((s for s in by_role["naive"] if not s.get("exclude_ratios")), None)
    p1 = sorted(by_role["phase1"], key=lambda s: s["tag"])
    p2 = sorted(by_role["phase2"], key=lambda s: (s["lam_std"], s["lam_psd"], s["tag"]))
    controls = by_role["control"] + by_role["phase2-control"]
    seq_star = min(p1, key=lambda s: s["best_val_mse"]) if p1 else None
    print(f"Runs in {root}: " + ", ".join(runs))

    table = results_table(runs, seq_star)
    table.to_csv(out / "tabla_resultados.csv", index=False)
    print("\n## Results (test unless stated; Cp units)\n")
    f4, f3 = "{:.4f}", "{:.3f}"
    print_markdown(table, ["tag", "scope", "seq", "lam_psd", "lam_std", "statloss", "epochs",
                           "train_min", "val_rmse_mean", "test_rmse1", "test_rmse10",
                           "test_rmse500", "test_rmse_mean", "test_rmse_max", "h_rmse_max",
                           "power_total", "std_ratio", "tpl_share", "skill_vs_seqstar"],
                   {"train_min": "{:.1f}", "val_rmse_mean": f4, "test_rmse1": f4,
                    "test_rmse10": f4, "test_rmse500": f4, "test_rmse_mean": f4,
                    "test_rmse_max": f4, "h_rmse_max": "{:.0f}", "power_total": f3,
                    "std_ratio": f3, "tpl_share": f3, "skill_vs_seqstar": f3,
                    "seq": "{:.0f}", "epochs": "{:.0f}", "lam_psd": "{:g}", "lam_std": "{:g}"})

    # ── SEQ* and the plan's lambda rule (choose on validation) ──
    chosen = None
    if seq_star:
        print(f"\nSEQ* = {seq_star['seq']} ({seq_star['run']}): lowest best val MSE "
              + ", ".join(f"{s['tag']} {s['best_val_mse']:.6f}" for s in p1))
        off = [s["run"] for s in p2 if s["seq"] != seq_star["seq"]]
        if off:
            print(f"[warn] Phase 2 runs not at SEQ*: {off}")
    if p2 and naive:
        thr, thr_mean = naive["val"]["rmse_h500"], naive["val"]["rmse_mean"]
        print(f"\nPlan's lambda rule: highest lam_psd with val RMSE@500 <= naive val {thr:.4f}"
              f"  (whole-horizon view: naive val mean RMSE {thr_mean:.4f}"
              + (f", SEQ* {seq_star['val']['rmse_mean']:.4f}" if seq_star else "") + ")")
        for s in p2:
            print(f"  {label(s):<30} val RMSE@500 {s['val']['rmse_h500']:.4f}  "
                  f"val mean RMSE {s['val']['rmse_mean']:.4f}  "
                  f"{'ok' if s['val']['rmse_h500'] <= thr else 'exceeds naive'}")
        ok_runs = [s for s in p2 if s["val"]["rmse_h500"] <= thr and s.get("psd_bins") != "linear"]
        if ok_runs:
            chosen = max(ok_runs, key=lambda s: (s["lam_psd"], s["lam_std"]))
            print(f"-> plan's rule picks {chosen['run']} — check it against the whole-horizon "
                  f"Pareto, the template share and the skill before using it")
        else:
            print("-> no Phase 2 run passes: keep the MSE model (SEQ*)")

    pareto(out / "pareto_rmse_vs_potencia.png", "rmse_mean", "RMSE averaged over h=1..500 [Cp]",
           naive, seq_star, p1, p2, controls, chosen)
    pareto(out / "pareto_rmse500.png", "rmse_h500", "RMSE@500 [Cp]",
           naive, seq_star, p1, p2, controls, chosen)

    # ── Phase 1: RMSE vs horizon (test) ──
    shown = p1 + by_role["control"] + ([naive] if naive else [])
    curves = {s["run"]: np.load(s["dir"] / "rmse_curve.npy") for s in shown}
    if curves:
        h = np.arange(1, len(next(iter(curves.values()))) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for s in shown:
            kw = dict(color="k", ls="--") if s["role"] == "naive" else \
                 dict(ls=":" if s["role"] == "control" else "-")
            axes[0].plot(h, curves[s["run"]], label=label(s), **kw)
            if naive and s["role"] != "naive":
                axes[1].plot(h, curves[s["run"]] / curves[naive["run"]], label=label(s), **kw)
        axes[0].set(xscale="log", xlabel="horizon h [ms]", ylabel="test RMSE [Cp]",
                    title="Phase 1: look-back window")
        axes[1].axhline(1, color="k", ls="--", lw=1)
        axes[1].set(xscale="log", xlabel="horizon h [ms]", ylabel="RMSE / naive RMSE",
                    title="relative to naive (same windows)")
        for ax in axes:
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fase1_ventana.png", dpi=150)
        plt.close(fig)

    # ── PSD of the predictions vs the truth (test) ──
    shown = ([seq_star] if seq_star else []) + p2 + by_role["phase2-control"]
    if shown:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        z = np.load(shown[0]["dir"] / "psd.npz")
        ax.loglog(z["f"][1:], z["test_true"][1:], color="k", lw=2, label="true Cp")
        for s in shown:
            z = np.load(s["dir"] / "psd.npz")
            ax.loglog(z["f"][1:], z["test_pred"][1:], lw=1.3,
                      label=f"{label(s)}  (P ratio {s['test']['power_ratio_total']:.2f})")
        ax.set(xlabel="frequency [Hz]", ylabel="mean periodogram (hann, 500 steps)",
               title="Spectrum of the 500-step forecasts vs. the truth (test)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "psd_predicciones.png", dpi=150)
        plt.close(fig)

    # ── Same test windows: MSE model vs StatLoss models ──
    mid = next((s for s in p2 if s["lam_psd"] == 0.003 and s.get("psd_bins") != "linear"), None)
    top = max((s for s in p2 if s.get("psd_bins") != "linear"), key=lambda s: s["lam_psd"],
              default=None)
    pair = [s for s in [seq_star, mid, top] + by_role["phase2-control"] if s is not None]
    pair = list({s["run"]: s for s in pair}.values())
    if pair and all((s["dir"] / "big" / "preds.npy").exists() for s in pair):
        preds = {s["run"]: np.load(s["dir"] / "big" / "preds.npy", mmap_mode="r") for s in pair}
        trues = np.load(pair[0]["dir"] / "big" / "trues.npy", mmap_mode="r")
        picks = [int((i + 0.5) * len(trues) / 4) for i in range(4)]
        colors = ["tab:blue", "tab:orange", "tab:red", "tab:purple", "tab:green"]
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        for ax, k in zip(axes, picks):
            ax.plot(trues[k], color="0.25", lw=0.8, label="true")
            for s, c in zip(pair, colors):
                ax.plot(preds[s["run"]][k], color=c, lw=1.2, label=label(s))
            ax.set_ylabel("Cp")
            ax.set_title(f"test window {k}", fontsize=9)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8, ncol=3)
        axes[-1].set_xlabel("ms from forecast origin")
        fig.suptitle("Same test windows: MSE model (SEQ*) vs. statistical-constraint models")
        fig.tight_layout()
        fig.savefig(out / "forecast_comparison.png", dpi=130)
        plt.close(fig)

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

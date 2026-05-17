"""
Diebold-Mariano (1995) test for forecast accuracy comparison.

For each pair of models (A, B), we compare the squared-error series on the
500-step autoregressive forecast trajectory from the same `test_seed`. The DM
statistic uses a HAC (Newey-West) variance estimator to account for the strong
autocorrelation of the loss differential in autoregressive multi-step forecasts.

H0 : E[d_t] = 0           (models A and B are equally accurate)
H1 : E[d_t] != 0          (one is significantly more accurate)
     where d_t = e_A,t^2 - e_B,t^2

Reference: Diebold & Mariano, J. Bus. Econ. Stat. 1995.
HAC lag selection: floor(n^{1/3}) ≈ 7 for n=500.

IMPORTANT: The per-model forecast .npy files were overwritten across rounds,
so only the Round 3 (scope="all") trajectories are still on disk. We therefore
restrict the DM test to Round 3.

Output: results/dm_test_round3.csv with one row per pair.
"""
import sys
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from train_utils import get_data

MODELS = ["ridge", "random_forest", "xgboost", "lstm", "gru", "tcn"]
HORIZON = 500
SEQ_LENGTH = 100
STEP = 10
SCOPES = {3: "all"}  # forecast .npy files only exist for the latest round


def hac_variance(d: np.ndarray, lag: int) -> float:
    """Newey-West HAC variance estimator with Bartlett kernel."""
    n = len(d)
    d_centered = d - d.mean()
    gamma0 = float(np.mean(d_centered ** 2))
    s = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)
        gamma_k = float(np.mean(d_centered[k:] * d_centered[:-k]))
        s += 2.0 * w * gamma_k
    return s


def dm_test(e_a: np.ndarray, e_b: np.ndarray, lag: int = None) -> dict:
    """
    Diebold-Mariano test on squared-error loss differential.

    Returns dict with DM statistic, p-value, mean loss differential, and direction.
    Positive D-bar means model A has LARGER errors (B is better).
    """
    d = e_a ** 2 - e_b ** 2
    n = len(d)
    if lag is None:
        lag = max(1, int(np.floor(n ** (1.0 / 3.0))))
    s2 = hac_variance(d, lag)
    if s2 <= 0:
        return {"DM": float("nan"), "p_value": float("nan"),
                "d_bar": float(d.mean()), "n": n, "lag": lag}
    dm = float(d.mean() / np.sqrt(s2 / n))
    p = float(2.0 * (1.0 - norm.cdf(abs(dm))))
    return {"DM": dm, "p_value": p, "d_bar": float(d.mean()), "n": n, "lag": lag}


def run_for_scope(rnd: int, scope: str):
    print(f"\n{'='*60}\n  Round {rnd}  (scope={scope})\n{'='*60}")
    data = get_data(scope, SEQ_LENGTH, STEP, 1)
    mu_w, sigma_w = float(data["mu"][0]), float(data["sigma"][0])
    gt = data["y_future"][:HORIZON] * sigma_w + mu_w  # (HORIZON,)
    test_seed_w = data["test_seed"][:, 0] * sigma_w + mu_w
    last_value = float(test_seed_w[-1])
    naive_pred = np.full(HORIZON, last_value, dtype=np.float32)

    # ── Load model trajectories ─────────────────────────────────────────
    preds: dict[str, np.ndarray] = {"naive": naive_pred}
    for m in MODELS:
        fpath = ROOT / m / "results" / "forecasts" / f"{m}_h{HORIZON}.npy"
        if not fpath.exists():
            print(f"  [skip] missing {fpath}")
            continue
        p = np.load(fpath)[:HORIZON]
        # clip ridge's R3 divergence so we can still test (it's pathological)
        if not np.all(np.isfinite(p)):
            print(f"  [skip] {m} contains non-finite values")
            continue
        # exclude models whose autoregressive trajectory diverges (>10x the
        # ground-truth range) — DM would just confirm what we already see
        gt_range = float(np.ptp(gt))
        if float(np.ptp(p)) > 10 * gt_range:
            print(f"  [skip] {m} diverged (range {np.ptp(p):.1f} >> {gt_range:.3f}); "
                  f"DM would be trivially dominated by it")
            continue
        preds[m] = p.astype(np.float32)

    available = list(preds.keys())
    print(f"  Available models for DM: {available}")
    errors = {k: gt - v for k, v in preds.items()}

    # ── Pairwise DM tests ────────────────────────────────────────────────
    rows = []
    for a, b in combinations(available, 2):
        r = dm_test(errors[a], errors[b])
        winner = "tie"
        if r["p_value"] < 0.05 and np.isfinite(r["DM"]):
            winner = b if r["d_bar"] > 0 else a
        rmse_a = float(np.sqrt(np.mean(errors[a] ** 2)))
        rmse_b = float(np.sqrt(np.mean(errors[b] ** 2)))
        rows.append({
            "round": rnd, "model_a": a, "model_b": b,
            "rmse_a": rmse_a, "rmse_b": rmse_b,
            "DM": r["DM"], "p_value": r["p_value"],
            "d_bar": r["d_bar"], "n": r["n"], "lag": r["lag"],
            "significant_005": (r["p_value"] < 0.05) if np.isfinite(r["p_value"]) else False,
            "winner": winner,
        })
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else " "
        print(f"  {a:>14s} vs {b:<14s}  DM={r['DM']:+7.3f}  p={r['p_value']:.4f} {sig}  winner={winner}")
    return rows


def main():
    out = ROOT / "results" / "dm_test_round3.csv"
    all_rows = []
    for rnd, scope in SCOPES.items():
        all_rows.extend(run_for_scope(rnd, scope))
    df = pd.DataFrame(all_rows)
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} pairwise tests to {out}")

    # Compact summary table per round
    print("\n=== SUMMARY: # significant wins (p<0.05) per model ===")
    for rnd in SCOPES:
        sub = df[df["round"] == rnd]
        wins = sub[sub["significant_005"]].groupby("winner").size()
        print(f"\nRound {rnd}:")
        if wins.empty:
            print("  (none significant)")
        else:
            for k, v in wins.sort_values(ascending=False).items():
                print(f"  {k:>14s}: {v} wins")


if __name__ == "__main__":
    main()

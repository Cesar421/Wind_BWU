# Changelog — Multi-trajectory long-horizon correction

Session correcting the single-trajectory evaluation bug (P1) and its
downstream documentation. All work below is committed on `main` in both the
`Wind_BWU` and `Wind_ML_TimeSeries` (thesis) repos.

## The bug that was fixed
`analyze_horizons.py` / `diebold_mariano.py` scored every autoregressive model
on **one** 500-step trajectory (first 500 steps of the last test series). That
near-flat trajectory inflated the naive baseline (RMSE 0.118 / R² 0.97) and the
deep-model "collapse" (R² = −6.18), and produced the headline "no model beats
naive at h=500".

## New code
| File | Purpose |
|------|---------|
| `Agent_Test/evaluate_long_horizon.py` | Multi-trajectory autoregressive rollout (naive + LSTM/GRU/TCN from checkpoints + Ridge/XGBoost re-fit) over 680 trajectories + cross-trajectory significance (paired Wilcoxon, win-rate, HAC-DM) |
| `Agent_Test/naive_dense_baseline.py` | Exact naive over the full 146,880-window test set (streaming, OOM-safe) |
| `data_loader.build_test_trajectories()` | Samples many (seed → future-H) trajectories across all test series |
| `Agent_Test/REFIT_HANDOFF.md` | Handoff note for the pending GPU/RF work |

## Retrained / re-evaluated
- **Re-fit (retrained):** Ridge + XGBoost (CPU, window stride 30).
- **Re-evaluated only (no training):** LSTM / GRU / TCN — existing checkpoints, inference rollout.
- **Untouched:** deep nets, LSTM-direct / PatchTST, Random Forest, WPTSE.

## New results (CSV)
- `results/long_horizon_multitraj_round3.csv` — full h=1…500 table, 8 models, 680 trajectories
- `results/long_horizon_significance_round3.csv` — significance vs naive (win-rate, Wilcoxon, DM)
- `results/naive_dense_metrics_round3.csv` — exact naive, 146,880 windows

### Corrected h=500 (per-step, multi-trajectory)
| Model | RMSE | R² | vs naive |
|-------|-----:|---:|----------|
| PatchTST (direct) | 0.175 | 0.85 | beats |
| LSTM-direct | 0.179 | 0.84 | beats |
| XGBoost (autoreg) | 0.209 | 0.79 | **beats** (DM +12.7) |
| Naive persistence | 0.231 | 0.74 | baseline |
| Ridge (autoreg) | 0.256 | 0.69 | stable, slightly worse |
| LSTM (autoreg) | 0.270 | 0.65 | competitive |
| GRU (autoreg) | 0.273 | 0.64 | competitive |
| TCN (autoreg) | 0.411 | 0.19 | unstable |

## New figures
- `results/plots_cross_round/rmse_vs_horizon_multitraj.png`
- `results/plots_cross_round/rmse_h500_boxplot_multitraj.png`
- Copied into the thesis: `figures/rmse_vs_horizon_multitraj.png`, `figures/rmse_h500_boxplot_multitraj.png`

## Documentation updated (Wind_BWU)
- `README.md` — corrected headline table + findings
- `docs/index.html` — results table, findings, timeline (GitHub Pages)
- `Agent_Test/RESULTS_SUMMARY.md` — §0.5 correction box, §4 / §6.b.2 notes, take-aways
- `AI_Agent/streamlit_app.py` — new "Long-horizon (corrected)" tab + legacy notes
- `analyze_horizons.py` / `diebold_mariano.py` — annotated as legacy single-trajectory

## Thesis (Wind_ML_TimeSeries)
- `front/abstract.tex`, `chapters/07_results.tex` (new multi-traj table + 2 figures + revised findings + DM finding iv), `chapters/08_discussion.tex`, `chapters/09_conclusion.tex`
- `main.pdf` rebuilt with the corrected content

## Key new findings
1. The "catastrophic R² = −6.18 collapse" was a single-trajectory artifact (real: LSTM/GRU R² ≈ 0.65).
2. **XGBoost beats naive** at h=500 (was reported as a tie) — wins 53 % of trajectories.
3. **Ridge does not diverge** (was reported ~10⁴) — stable at 0.256.
4. **Three models beat naive** at h=500: PatchTST, LSTM-direct, XGBoost (was: none).
5. The spectral-collapse finding (predicted PSD 60–75× below true) is unchanged.

## Still pending
- Random Forest multi-trajectory re-fit (~2 h CPU; `--rf`).
- GPU retrain of LSTM-direct + PatchTST (checkpoints absent → reproducibility, spectral analysis, Streamlit live demo).

## Commits
- **Wind_BWU:** `f7b8187`, `0c5286c`, `a20a7b2`, `a57e4c8`
- **Thesis:** `88174e6`, `418d6d9`, `534bc71`

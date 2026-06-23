# Handoff — classical multi-trajectory re-fit (GPU machine)

Context for continuing on another machine (the Claude Code chat does **not**
sync across devices; this note + the commit history carry the context).

## What is already done (committed to `main`)
- `evaluate_long_horizon.py` — multi-trajectory autoregressive rollout (naive +
  LSTM/GRU/TCN from checkpoints) + cross-trajectory significance.
- `naive_dense_baseline.py` — exact naive over the full 146,880-window test set.
- Corrected h=500 results in `results/long_horizon_multitraj_round3.csv`,
  `results/naive_dense_metrics_round3.csv`,
  `results/long_horizon_significance_round3.csv`.
- Docs (README, docs/index.html, RESULTS_SUMMARY) + thesis updated.

Corrected headline (h=500, per-step): PatchTST 0.175 / R² 0.85 and LSTM-direct
0.179 / 0.84 **beat** naive 0.231 / 0.74; LSTM/GRU autoreg ≈0.27 (R²≈0.65,
competitive); TCN 0.41 (unstable).

## What is PENDING — the classical re-fit
Ridge / Random Forest / XGBoost were never pickled, so they are missing from the
multi-trajectory table (rows marked † "pending re-fit"). Re-fit + roll them out:

```bash
# from Agent_Test/, env with torch + sklearn + xgboost + scipy + pandas
python evaluate_long_horizon.py --classical --rf
```
- `--classical` re-fits Ridge (~1 min) + XGBoost (~minutes) on the 776k R3
  windows and adds them to `long_horizon_multitraj_round3.csv`.
- `--rf` also re-fits Random Forest (~2 h on CPU; this is the slow one).
- GPU is **not** required (these are CPU/tree models). For GPU XGBoost, edit the
  `XGBRegressor(... tree_method="hist")` line in `evaluate_long_horizon.py` to
  `device="cuda"`.

Data is versioned in git (`Data/Data_All_The_BDH_PostProcess`), so a `git pull`
is enough — no manual data copy.

## After the re-fit
1. The CSV/table now include Ridge/RF/XGBoost multi-trajectory rows.
2. Update the † rows in `docs/index.html` and the README/RESULTS_SUMMARY tables.
3. (Optional) add them to the thesis `tab:round3_rmse_multitraj`.

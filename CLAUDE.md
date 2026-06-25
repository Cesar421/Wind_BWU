# Wind_BWU — project notes for Claude Code

Deep-learning + classical forecasting of wind-pressure coefficients (Cp) on the
TPU BDH benchmark. Pipeline lives in `Agent_Test/`; thesis is a **separate repo**
at `../Wind_ML_TimeSeries`.

## 👉 Continuing on a new machine? Read `HANDOFF.md` first.
There is pending GPU work (retrain LSTM-direct + PatchTST, re-fit Random Forest,
re-run spectral analysis). `HANDOFF.md` has the full context + exact commands, and
`Agent_Test/run_all_gpu.ps1` / `run_all_gpu.sh` run it all in one shot.

## Key facts
- **Checkpoints (`*.pt`) are gitignored** — `git pull` does not bring them; nets
  must be retrained per machine. Data (`Data/Data_All_The_BDH_PostProcess`) is in git.
- Always set `KMP_DUPLICATE_LIB_OK=TRUE`. Env needs torch (+CUDA on GPU), sklearn,
  xgboost, scipy, pandas, matplotlib. (The README's `ML_Cesar` env does not exist.)
- Run training from `Agent_Test/`: `python train_all.py --scope all` (R3), or the
  per-model `python <model>/train_<model>.py --scope all`.
- Corrected long-horizon evaluation: `evaluate_long_horizon.py` (multi-trajectory)
  supersedes the legacy single-trajectory `analyze_horizons.py` / `diebold_mariano.py`.

## Current state
Multi-trajectory correction applied, committed, pushed (see `CHANGELOG.md`). At
h=500: PatchTST 0.175, LSTM-direct 0.179, XGBoost 0.209 all beat naive 0.231.
Pending only: Random Forest re-fit + GPU retrain of the two direct models.

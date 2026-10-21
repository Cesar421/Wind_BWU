# Wind_BWU — project notes for Claude Code

Deep-learning + classical forecasting of wind-pressure coefficients (Cp) on the
TPU BDH benchmark. Pipeline lives in `Agent_Test/`; thesis is a **separate repo**
at `../Wind_ML_TimeSeries`.

## 👉 Continuing on a new machine? Read the latest file in `Handoffs/` first.
Current: [`Handoffs/2026-10-21_plan_GPU_ventana_y_restriccion_estadistica.md`](Handoffs/2026-10-21_plan_GPU_ventana_y_restriccion_estadistica.md)
— self-contained GPU experiment plan for PatchTST (longer look-back window +
statistical/spectral loss), with measured timings. The root `HANDOFF.md` (June 2026)
is **superseded**: do not re-run its pending work (RF re-fit is done; the direct-model
retrain is not needed for the new plan).

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
h=500: PatchTST 0.175 and LSTM-direct 0.179 beat naive (0.231 on the same windows);
Random Forest / XGBoost tie naive (0.212 / 0.217 vs 0.218 over 680 trajectories).
RF re-fit done. Next: the GPU plan in `Handoffs/` (2026-10-21).

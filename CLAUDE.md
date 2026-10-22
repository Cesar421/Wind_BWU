# Wind_BWU — project notes for Claude Code

Deep-learning + classical forecasting of wind-pressure coefficients (Cp) on the
TPU BDH benchmark. Pipeline lives in `Agent_Test/`; thesis is a **separate repo**
at `../Wind_ML_TimeSeries`.

## 👉 Continuing on a new machine? Read the latest file in `Handoffs/` first.
Current: [`Handoffs/2026-10-22_resultados_GPU.md`](Handoffs/2026-10-22_resultados_GPU.md)
— results of the GPU plan (PatchTST look-back window + statistical loss): what was
run, the numbers, deviations from the plan and next steps. The plan itself
(`Handoffs/2026-10-21_...`) is **completed**; the root `HANDOFF.md` (June 2026) is
**superseded**.

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
GPU plan done (2026-10-22, results in `Agent_Test/results/gpu_plan_2026-10/`):
PatchTST with a 2 s look-back (`patchtst/train_patchtst_v2.py`, end-aligned patches)
reaches RMSE@500 0.1665 on R3 (published seq-100 model: 0.1752; naive 0.231) and
generalises to unseen geometries (leave-buildings-out ≈ same error). The statistical
/ spectral loss is a negative result (λ = 0). Removing each series' own train-portion
mean (`--demean`, in memory) gives the best model so far: RMSE@500 0.1632 ± 0.0000
over 3 seeds (needs the series' mean from a prior measurement). Earlier: PatchTST 0.175 and LSTM-direct
0.179 beat naive; Random Forest / XGBoost tie naive over 680 trajectories.
Next: thesis text (see the results handoff, section 5).

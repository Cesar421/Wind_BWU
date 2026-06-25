# HANDOFF — read this first (continuing on the GPU machine)

> **For a fresh Claude Code session:** you are continuing the Wind_BWU /
> `Wind_ML_TimeSeries` (thesis) project. This file is the bridge between machines
> (the previous chat does not sync; git does). Read this, then do the **Pending
> work** below. Full background is in `CHANGELOG.md`, `Agent_Test/RESULTS_SUMMARY.md`
> and `Agent_Test/REFIT_HANDOFF.md`.

## What was just done (context)
The long-horizon (h=500) evaluation was scored on a **single 500-step trajectory**,
which was a bug: it inflated the naive baseline (RMSE 0.118 / R² 0.97) and made the
deep models look collapsed (R² = −6.18). It was re-done over **680 trajectories**
(`evaluate_long_horizon.py`) + exact naive over 146,880 windows
(`naive_dense_baseline.py`). Corrected result at h=500: PatchTST 0.175, LSTM-direct
0.179, **XGBoost 0.209 — all beat naive 0.231**; LSTM/GRU ~0.27 (competitive); TCN
0.411 (unstable). Docs + thesis already updated and pushed.

## Critical gotchas
1. **Checkpoints (`*.pt`) are gitignored** → `git pull` does NOT bring them. The
   neural nets (LSTM/GRU/TCN) and the direct models (LSTM-direct, PatchTST) must be
   **retrained on this machine**. The data (`Data/Data_All_The_BDH_PostProcess`) IS
   in git, so it arrives with the pull.
2. The README mentions a conda env `ML_Cesar` that does not exist; use whatever env
   has **torch + CUDA, sklearn, xgboost, scipy, pandas, matplotlib**.
3. Always `export KMP_DUPLICATE_LIB_OK=TRUE` (Windows: `$env:KMP_DUPLICATE_LIB_OK="TRUE"`).

## Pending work (the reason for the GPU machine)
1. **Retrain the two DIRECT models** (checkpoints absent — the project's best results,
   and required to regenerate the spectral analysis):
   - `python lstm/train_lstm_direct.py --scope all --horizon 500`  (~45 min A4000)
   - `python patchtst/train_patchtst.py --scope all --horizon 500`  (~71 min A4000)
2. **Random Forest** multi-trajectory re-fit (CPU, ~2 h): include via `--rf` below.
3. Re-run evaluation and spectral analysis:
   - `python evaluate_long_horizon.py --classical --rf`
   - `python naive_dense_baseline.py`
   - `python spectral_analysis.py`   (needs the direct trues/preds from step 1)

**Or just run everything with one command** (see below).

## One-command runner
From `Agent_Test/`, after activating the env:
- Windows:  `./run_all_gpu.ps1`
- Linux/Mac: `bash run_all_gpu.sh`

## After it finishes
- New numbers land in `results/long_horizon_multitraj_round3.csv` (now with Random
  Forest), `results/spectral_metrics.csv`, and the regenerated checkpoints.
- Update the † "pending" Random-Forest rows in `docs/index.html`, `README.md`,
  `Agent_Test/RESULTS_SUMMARY.md`, and the thesis `tab:round3_rmse_multitraj`.
- Commit + push both repos. Rebuild the thesis PDF (`latexmk -pdf main.tex`).

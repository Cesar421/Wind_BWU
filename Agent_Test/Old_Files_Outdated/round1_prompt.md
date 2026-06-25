# Round 1 — Smoke Test Training Prompt

**Goal:** Validate the full training pipeline on a single building configuration
(`Alpha1_4 / 2_1_3`) before scaling to the full BDH dataset.

**Dataset for this round:** Only `Alpha1_4/2_1_3`
(B=0.2, D=0.1, H=0.3 → 360 taps, 21 wind incidence angles 0°–100°).

**Models to train in this round (6 total):**

| # | Model | Category | Where |
|---|---|---|---|
| 1 | Ridge Regression | linear baseline | CPU |
| 2 | Random Forest | tree ensemble baseline | CPU |
| 3 | XGBoost | gradient boosting baseline | CPU/GPU |
| 4 | LSTM | recurrent DL | GPU |
| 5 | GRU | recurrent DL | GPU |
| 6 | TCN (TemporalConvNet) | convolutional DL | GPU |

**Do NOT train in this round:** TemporalFusionTransformer, PatchTST.
Those are scheduled for Round 3 only if Rounds 1–2 succeed.

---

## Prompt to paste into the agent on the other machine

```text
@wind-cp-forecaster

ENVIRONMENT:
- Conda env: ML_Cesar
- All terminal commands:
    conda activate ML_Cesar; $env:KMP_DUPLICATE_LIB_OK="TRUE"; python <script>
- GPU: NVIDIA RTX A4000 (16 GB VRAM), CUDA 11.8
- torch.cuda.is_available() must be True. DO NOT fall back to CPU for DL models.
- device = torch.device("cuda") for all DL models and tensors.
- DataLoaders: pin_memory=True, num_workers=0.

DATA SCOPE (ROUND 1 ONLY):
- Use ONLY Alpha1_4 / building ratio 2_1_3 (360 taps, 21 angles).
- Faces: ["windward", "leeward", "sideleft", "sideright"] (multivariate, 4 channels).
- Sliding window: seq_length=100, step=10.
- Splits: chronological 70/15/15 (train/val/test).
- Normalization: z-score, fit on train split only, save scaler params.

STEP 1 — Read the 7 papers in `Wind pressure coefficients/` and write a structured
summary into `Agent_Test/paper_summaries.md` (model arch, inputs, loss, hyperparams,
metrics for each paper). Skip this step if the file already exists.

STEP 2 — Create `Agent_Test/data_loader.py` as a SHARED module:
- Loads .npy facade data for Alpha1_4/2_1_3 only (read `default_alpha` and
  `default_building_ratio` from `config/settings.yaml`).
- Applies z-score normalization (fit on train).
- Returns sliding-window PyTorch Datasets (train/val/test).
- Provides `get_loaders(batch_size, seq_length, step)` helper.

STEP 3 — Implement these 6 models, each in its own folder
`Agent_Test/<model_name>/`:
  - RidgeRegression (sklearn, no GPU)
  - RandomForest (sklearn, no GPU)
  - XGBoostTimeSeries (xgboost, GPU if available via tree_method="gpu_hist")
  - LSTM (PyTorch, CUDA)
  - GRU (PyTorch, CUDA)
  - TemporalConvNet (PyTorch, CUDA)

Each folder must contain:
  - `models/<model_name>.py` — architecture class
  - `train_<model_name>.py` — training + evaluation + forecasting
  - `checkpoints/` — saved weights (.pt or .pkl)
  - `results/model_comparison.csv` — metrics (one row per model)
  - `results/plots/` — training curves + prediction vs actual
  - `results/forecasts/` — multi-step forecast arrays (.npy)

For each model:
  - Import shared `data_loader.py` (NO duplicate data loading code).
  - For DL models: device = torch.device("cuda"), pin_memory=True.
  - Early stopping (patience=15), ReduceLROnPlateau (patience=5, factor=0.5).
  - Compute metrics on the test set: RMSE, MAE, R², MAPE, directional accuracy.
  - Log training time (seconds) and parameter count.
  - Generate multi-step predictions at horizons: 1, 10, 50, 100, 500.
  - Save training curves + prediction vs actual plots.

STEP 4 — Create `Agent_Test/train_all.py` that:
  - Runs all 6 model training scripts sequentially via subprocess using:
      conda activate ML_Cesar; $env:KMP_DUPLICATE_LIB_OK="TRUE"; python <script>
  - Collects metrics from each `<model_name>/results/`.
  - Generates consolidated `Agent_Test/results/model_comparison.csv`.
  - Creates cross-model comparison plots in `Agent_Test/results/plots/`:
      * Bar chart of RMSE per horizon per model.
      * R² vs forecast horizon (line plot, one line per model).
      * Training time vs accuracy tradeoff scatter.

GLOBAL:
- Use float32.
- Random seed: 42.
- Total expected runtime for Round 1: 2–4 hours on RTX A4000.
- If any single model exceeds 60 minutes of training, stop it and log the issue.

REPORT WHEN DONE:
- Total runtime per model.
- Best model by test RMSE at horizon=1.
- Best model by test RMSE at horizon=100.
- Any models that failed and why.
```

---

## After Round 1 finishes

1. Pull `Agent_Test/results/model_comparison.csv` back to the laptop.
2. Review:
   - Do DL models beat baselines at horizon ≥ 50? (key thesis question)
   - Any pipeline bugs (NaN losses, exploding gradients, etc.)?
   - Training times realistic for scaling up?
3. If results look healthy → proceed to **Round 2** (multi-building subset).
4. If results look broken → debug before scaling up.

## Round 2 (planning ahead, do NOT run yet)

- Dataset: ratios `1_1_3`, `2_1_3`, `3_1_3` × both Alpha profiles = 6 configs.
- Same 6 models.
- Estimated runtime: 12–24 hours.
- Change in `config/settings.yaml`:
  ```yaml
  use_all_buildings: true
  building_ratios:
    - "1_1_3"
    - "2_1_3"
    - "3_1_3"
  ```
  (temporarily prune the building_ratios list)

## Round 3 (only if Round 2 is clean)

- Dataset: all 12 ratios × both Alpha profiles.
- Add Transformers (TFT, PatchTST) to seed_models.
- Estimated runtime: 3–5 days.

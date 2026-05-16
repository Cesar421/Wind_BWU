# Modeling Plan — Wind Pressure Cp Time Series

**Purpose:** Single source of truth for the modeling strategy of the thesis.
Follow this plan on any machine. Each round must be completed and validated
before moving to the next.

**Hardware available:**
- NVIDIA RTX A4000 (16 GB VRAM) — primary GPU for training
- NVIDIA RTX 4500 (24 GB VRAM) — Germany (optional, larger models)
- NVIDIA RTX 3070 (8 GB VRAM) — Colombia (small jobs only)

**Conda environment (all machines):** `ML_Cesar`

---

## Thesis objective (recap)

Build a model that can **forecast and/or generate wind pressure coefficient
($C_p$) time series** on building surfaces from the TPU / BDH aerodynamic
database, capable of capturing at minimum the mean and ideally the full
temporal dynamics. The model must generalize across building geometries,
wind incidence angles, and terrain exposure categories.

---

## Dataset

**Source:** BDH (Building Database Hub) postprocessed data
**Path:** `Wind_BWU/Data/Data_All_The_BDH_PostProcess`
**Format:** `.npy` files, multivariate (4 façades: windward, leeward, sideleft, sideright)
**Sampling:** 1000 Hz, 32768 timesteps per record
**Configurations:**
- 2 terrain profiles: `Alpha1_4`, `Alpha1_6`
- 12 building ratios (B/D/H combinations)
- 11–21 wind incidence angles per config (0°–50° or 0°–100°)
- Total: ~380 time series across all combinations

---

## Models (final list for the thesis)

| # | Model | Category | Reason for inclusion |
|---|---|---|---|
| 1 | Ridge Regression | linear baseline | Standard reference, very fast |
| 2 | Random Forest | tree ensemble baseline | Nonlinear baseline, no temporal modeling |
| 3 | XGBoost | gradient boosting baseline | Strong tabular baseline (Hu et al. 2020) |
| 4 | LSTM | recurrent DL | Standard for time series, cited in Nav et al. 2025 |
| 5 | GRU | recurrent DL | Simplified LSTM, fewer parameters |
| 6 | TCN (TemporalConvNet) | convolutional DL | Bai et al. 2018 — competes with LSTM |
| 7 | Temporal Fusion Transformer | transformer | Attention-based, handles covariates |
| 8 | PatchTST | transformer | Recent SOTA for long-horizon forecasting |

**Explicitly excluded (with reasoning):**
- ❌ NBEATS — no native covariate support (needs geometry + angle as inputs)
- ❌ DeepAR — probabilistic forecasting, out of thesis scope
- ❌ WaveNet — redundant with TCN
- ❌ Informer — redundant with TFT and PatchTST
- ❌ SVR — kept as literature reference only, not implemented
- ❌ GAN / WPTSE-Net — cited as related work, too complex to implement from scratch

---

## Evaluation protocol (applies to ALL models and rounds)

**Splits:** chronological 70 / 15 / 15 (train / val / test)
**Normalization:** z-score, fit on train split only
**Sliding window:** seq_length=100, step=10
**Random seed:** 42
**Metrics on test set:**
- RMSE
- MAE
- R²
- MAPE
- Directional accuracy
**Forecast horizons:** 1, 10, 50, 100, 500 steps ahead
**Training controls:**
- Early stopping (patience=15)
- ReduceLROnPlateau (patience=5, factor=0.5)
- Mixed precision (DL models)
**Logging:** training time (s) and parameter count per model
**Tracking:** MLflow (`./mlruns`)

---

## Round 1 — Smoke test (single building)

**Goal:** Validate the pipeline end-to-end. Find bugs early.

| Setting | Value |
|---|---|
| Dataset | `Alpha1_4 / 2_1_3` (360 taps, 21 angles 0°–100°) |
| Models | Ridge, Random Forest, XGBoost, LSTM, GRU, TCN |
| Transformers | ❌ Skip (postponed to Round 3) |
| Estimated runtime | 2–4 hours on RTX A4000 |
| Settings flag | `use_all_buildings: false`, `default_building_ratio: "2_1_3"` |

**Detailed prompt to run:** see [`round1_prompt.md`](./round1_prompt.md)

**Success criteria for proceeding to Round 2:**
- All 6 models train without crashes (no NaN losses)
- At least one DL model beats the Naive Persistence baseline at horizon ≥ 50
- Total runtime ≤ 6 hours (so multi-building scaling stays feasible)

**Deliverables:**
- `Agent_Test/results/model_comparison.csv` (consolidated metrics)
- `Agent_Test/results/plots/` (cross-model plots)
- `Agent_Test/<model_name>/checkpoints/` (saved weights)
- `Agent_Test/<model_name>/results/forecasts/` (multi-step forecast arrays)

---

## Round 2 — Multi-building subset

**Goal:** Test generalization across geometries and terrain categories.
This is the **core experimental result** for the thesis.

| Setting | Value |
|---|---|
| Dataset | Ratios `1_1_3`, `2_1_3`, `3_1_3` × both Alpha profiles = 6 configs |
| Models | Same 6 models as Round 1 (re-trained on larger data) |
| Estimated runtime | 12–24 hours |

**Settings.yaml changes for Round 2:**
```yaml
data:
  use_all_buildings: true
  building_ratios:
    - "1_1_3"
    - "2_1_3"
    - "3_1_3"
  # (temporarily prune the full list)
```

**Success criteria for proceeding to Round 3:**
- DL models outperform baselines on at least one geometry/angle combination
- No memory issues (must fit in 16 GB VRAM with batch_size=64)
- Forecast quality at horizon=100 still informative (R² > 0 for at least 1 model)

---

## Round 3 — Full dataset + Transformers (optional)

**Goal:** Final results for the thesis. Only run if Rounds 1–2 are clean.

| Setting | Value |
|---|---|
| Dataset | All 12 building ratios × both Alpha profiles (~24 configs) |
| Models | All 8 models (Round 2 models + TFT + PatchTST) |
| Estimated runtime | 3–5 days |

**Settings.yaml changes for Round 3:** revert to full `building_ratios` list and
re-enable transformers in the active training script.

**Risk:** Long runtime → run as overnight jobs in segments, checkpoint often,
never rely on a single uninterrupted run.

---

## Thesis chapter mapping

| Chapter | Source data |
|---|---|
| §4.1 Baseline results | Round 1: classical baselines (one-step + multi-step) |
| §4.2 Deep learning results | Round 1–2: LSTM, GRU, TCN |
| §4.3 Multi-building generalization | Round 2 only |
| §4.4 (optional) Transformer comparison | Round 3 only |
| §4.5 Discussion: separation zones, peak events | Cross-cutting from all rounds |

---

## Workflow checklist (per round)

Before running each round:
- [ ] Pull latest `Wind_BWU` repo on the training machine (`git pull`)
- [ ] Confirm `conda activate ML_Cesar` works
- [ ] Confirm `torch.cuda.is_available() == True`
- [ ] Confirm the BDH data exists at `../Data/Data_All_The_BDH_PostProcess`
- [ ] Update `config/settings.yaml` for the current round (see tables above)
- [ ] Run the round-specific prompt (`round1_prompt.md`, etc.)

After each round:
- [ ] Commit results: `Agent_Test/results/` and checkpoints
- [ ] Push to repo so the laptop can pull and analyze
- [ ] Update this plan if scope changes (e.g., dropping a model)
- [ ] Decide: proceed, repeat, or pivot

---

## Things NOT to do

- ❌ Do not change normalization between rounds (breaks comparability)
- ❌ Do not change random seed between rounds
- ❌ Do not skip Round 1 to "save time" — it is your safety net
- ❌ Do not implement extra models on a whim (NBEATS, DeepAR, etc.)
  unless this plan is updated first
- ❌ Do not fall back to CPU for DL models (would invalidate runtime metrics)
- ❌ Do not run Round 3 without completing Round 2 successfully

---

## Open questions (decide before Round 3)

1. Hyperparameter optimization (Optuna): enable for which models? Cost is
   `n_trials × per-trial-time` — easily 5× the base training time.
2. Cross-validation folds: keep 5 or reduce to 3 to save time?
3. Should the thesis include a **per-angle** breakdown of results, or only
   aggregate metrics? (Per-angle is more informative but more space.)
4. Final visualization: include $C_p$ time-history reconstruction plots
   (true vs predicted) for selected taps in separation zones.

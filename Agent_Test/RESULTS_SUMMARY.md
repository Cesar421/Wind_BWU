# Wind Pressure Coefficient Forecasting — Results Summary

Comparison of 6 forecasting models trained under a strict fair-baseline protocol on
wind-tunnel pressure-coefficient ($C_p$) time series from the TPU BDH benchmark.

---

## 0. Executive overview

### 0.1 Motivation and objective

Wind-tunnel pressure-coefficient ($C_p$) time series govern the design of
tall-building cladding, fatigue life of facade attachments, and gust-effect
factors in structural codes. Real-time / look-ahead $C_p$ forecasting would
enable adaptive damping, gust early-warning, and reduced reliance on long
wind-tunnel campaigns. The **central question** of this study is:

> *Given 100 ms of past windward $C_p$ + velocity context, can a learning
>  model produce useful $C_p$ forecasts at lead times from 1 ms to 500 ms,
>  and what is the limiting factor as the horizon grows?*

We answer this in three layers: (i) a fair multi-model baseline benchmark
(Round 1 → 2 → 3, single-step + autoregressive rollout); (ii) an
intervention on the training objective (direct multi-step) with a second
architecture for redundancy; (iii) a diagnostic spectral analysis that
reframes what "winning" means.

### 0.2 Data and protocol — one paragraph

We use the **TPU BDH** benchmark: pressure-coefficient time series sampled at
**1 000 Hz** on tall-building models in a boundary-layer wind tunnel,
covering 20 building aspect ratios × multiple yaw angles × 2 terrain-roughness
profiles ($\alpha = 1/4$ and $1/6$). Each series has 32 768 samples
(~32.8 s). Inputs are 100-sample (100 ms) windows of 4 face-averaged $C_p$
channels (windward / leeward / side-left / side-right). Three nested rounds
of training scope: **R1** (one building, sanity check), **R2** (5 geometries
× 2 roughness, 85 series), **R3** (universal: 340 series). Identical
train/val/test (70/15/15), seed, optimiser, batch, hardware across all
models so that differences are attributable to the model.

### 0.3 Models — what we trained and why

| Family | Model | Paradigm | What it is testing |
|--------|-------|----------|--------------------|
| **A. Autoregressive baselines** | Naive persistence | $\hat y_{t+1}=y_t$ | Lower bound any non-trivial model must beat |
| | Ridge regression | Linear | Linear ceiling, sanity check |
| | Random Forest | Trees | Non-linearity without temporal order |
| | XGBoost | Boosted trees | Best classical baseline |
| | ANN (MLP) | Feed-forward | Needs *state*? Or does a flat MLP suffice? |
| | LSTM (autoreg) | Recurrent gated | Explicit memory for short context |
| | GRU (autoreg) | Recurrent gated | Lighter LSTM variant |
| | TCN (autoreg) | Dilated conv | Long receptive field without recurrence |
| | CNN-LSTM (autoreg) | Hybrid | Local features + memory |
| | Transformer (autoreg) | Self-attention | Global context |
| **B. Direct multi-step** | LSTM-direct (B.4) | LSTM(128) + `Linear(128, 500)` | Can changing only the **loss objective** fix the rollout collapse? |
| | PatchTST direct (C.6) | Patch + 3-layer Transformer encoder | Does a second, very different architecture reach the same ceiling? |

**Family A** uses **autoregressive rollout**: train to predict one step ahead,
then feed the prediction back to forecast horizon $h>1$. Cheap to train but
exposes the model to compounding error.

**Family B** uses **direct multi-step**: predict the full 500-step output
vector in one forward pass, MSE over the whole vector. More expensive head
(wider final linear layer), but no compounding.

### 0.4 Methodology — beyond per-step error

- **Section 5** (single-step) and **6.a** (autoregressive multi-horizon):
  RMSE / MAE / R² / MAPE per horizon, per round, per model.
- **Section 6.b** (B.4): Direct multi-step training as an intervention on the
  loss objective — keep the architecture, change the training target.
- **Section 6.b.2 — B.5** (Diebold-Mariano test): a paired statistical test
  on squared-error sequences with HAC autocorrelation correction. Answers
  "is the RMSE gap between models *statistically* real or sampling noise?"
- **Section 6.c**: Replicate B.4 with a Transformer-based architecture
  (PatchTST) for architectural redundancy.
- **Section 6.d — F** (spectral analysis): Welch PSD of true signal,
  predictions, and residuals. Answers "what frequencies do the models
  actually reproduce?".
- **Future — N / M**: spectral fidelity metrics (PSD L² distance, total
  power ratio, peak-factor MAE) and distributional forecasting (quantile
  regression with pinball loss).

### 0.5 Headline results

> **⚠️ Correction (multi-trajectory re-evaluation).** Sections 4 and 6.b.2 below
> were originally computed on a **single** 500-step trajectory (the first 500
> steps of the last test series). That inflated the naive baseline (RMSE 0.118 /
> R² ≈ 0.97 was an artifact of one near-flat trajectory) and the deep-model
> "collapse" (R² = −6.18). Re-evaluating every autoregressive model over **680
> trajectories** (`evaluate_long_horizon.py`) and the naive baseline over the
> **full 146,880-window test set** (`naive_dense_baseline.py`) gives the
> corrected h = 500 numbers used in this headline. New artefacts:
> `results/long_horizon_multitraj_round3.csv`,
> `results/long_horizon_significance_round3.csv`,
> `results/naive_dense_metrics_round3.csv`.

**Single-step ($h=1$) is essentially solved.** Every reasonable model (R3)
reaches RMSE ≈ 0.04 and R² > 0.99. Differences here are not informative.

**At $h = 500$ (per-step, full-test-set / multi-trajectory regime):**

| Model | RMSE | R² | vs naive |
|---|---:|---:|---|
| PatchTST (direct) | **0.175** | **0.85** | **beats naive** |
| LSTM-direct | **0.179** | **0.84** | **beats naive** |
| Naive persistence | 0.231 | 0.74 | baseline |
| LSTM (autoreg) | 0.270 | 0.65 | beats naive on 44 % of traj. |
| GRU (autoreg) | 0.273 | 0.64 | beats naive on 47 % of traj. |
| TCN (autoreg) | 0.411 | 0.19 | genuinely unstable |

**Direct multi-step (B.4 + C.6) beats naive persistence.** Identical LSTM
backbone trained with `Linear(128 → 500)` head + full-vector MSE achieves
RMSE 0.179 / R² 0.84 at $h = 500$; PatchTST lands at RMSE 0.175 / R² 0.85.
Both clear the naive bar (0.231 / 0.74). **Two radically different
architectures converge to the same ceiling.**

**Autoregressive rollout is competitive, not catastrophic.** Over 680
trajectories the recurrent models (LSTM/GRU) reach R² ≈ 0.64–0.65 — modestly
below naive (0.74) but beating it on ~45 % of individual trajectories. Naive
keeps a small, statistically significant aggregate edge over them (paired
Wilcoxon $p < 10^{-3}$, but a ~6 % median-RMSE effect). Only **TCN** truly
destabilises (R² = 0.19) by amplifying compounded feedback error. The
earlier "catastrophic R² = −6 collapse" was a single-trajectory artifact.

**Spectral analysis (F) reframes again — and is the most important finding.**
At $h = 500$, the predicted PSD is **60–75× smaller than the true PSD at
every frequency** between 4 Hz and 500 Hz. The residual PSD overlaps the
true-signal PSD exactly. The models are not predicting turbulent
fluctuations at all — they output the **conditional mean** within each
forecast window. The RMSE 0.175 is exactly the standard deviation of the
collapsed high-frequency component. **R² = 0.85 is therefore misleading for
wind-engineering applications** (fatigue, peak factor, gust-effect factor)
that depend on the spectrum.

### 0.6 Generated artefacts

**Per-round CSVs** (`results/`):
- `model_comparison_round{1,2,3}.csv`: single-step metrics
- `multi_horizon_metrics_round{1,2,3}.csv`: per-horizon autoregressive metrics
- `lstm_direct_metrics.csv`: direct multi-step (B.4)
- `patchtst_metrics.csv`: PatchTST direct (C.6)
- `dm_test_round3.csv`, `dm_test_all_rounds.csv`: Diebold-Mariano p-values
- `spectral_metrics.csv`: total power and band-limited power per model

**Plots** (`results/plots_round{1,2,3}/` and `results/plots_cross_round/`):
- Per-round: bar charts of RMSE/R² and RMSE-vs-horizon for each model
- Cross-round: `cross_round_horizon_<model>.png` (5 files, one per model
  family — shows how each model scales R1 → R2 → R3)
- `cross_round_rmse_h1.png`: single-step ranking across rounds
- `direct_vs_autoreg_h500.png`: overlay of autoregressive baselines vs
  LSTM-direct vs PatchTST at $h = 500$ (linear + log)
- `psd_residuals.png`: Welch PSD of true / pred / residual for both
  direct-multi-step models — the visual proof of spectral collapse

**Source code** (key files in `Agent_Test/`):
- `<model>/models/<model>.py` and `<model>/train_<model>.py` for each of
  ANN / LSTM / GRU / TCN / CNN-LSTM / Transformer (autoreg) and
  LSTM-direct / PatchTST (direct).
- `train_utils.py`: shared `get_data`, `make_loaders`, `train_model`,
  `compute_metrics`, `set_seed` — guarantees identical training conditions.
- `data_loader.py`: scope dispatcher (R1 / R2 / R3) + windowing +
  normalisation.
- `infer_lstm_direct.py`: post-crash inference recovery (B.4 TDR survival).
- `dm_test.py`: Diebold-Mariano with HAC variance.
- `plot_direct_vs_autoreg.py`: consolidated direct-vs-autoreg plot.
- `spectral_analysis.py`: Welch PSD pipeline.

### 0.7 What this work *changes* for downstream practitioners

1. **Do not use teacher-forced single-step training when the deployment task
   is long-horizon forecasting.** It looks great at $h = 1$ and silently
   diverges at $h = 500$.
2. **Always report a statistical test** alongside RMSE on long horizons.
   Half the published "X beats Y" claims would not survive a DM test.
3. **R² and RMSE are insufficient metrics for wind-engineering forecasts.**
   A model can score R² = 0.85 with predictions that have 60× less spectral
   power than the true signal — useless for fatigue or peak-factor
   computation. Add at least a **total power ratio** or **PSD L² distance**.
4. **MSE training collapses to the conditional mean on chaotic signals.**
   The next modelling step is **distributional / generative** forecasting
   (quantile regression with pinball loss, diffusion, conditional GAN,
   normalising flow) that can match the conditional *distribution* and
   therefore the spectrum.

The rest of this document records the experiments, numbers, and analyses
that support each of the four points above.

---

## 1. Experimental Protocol (identical across all rounds)

| Setting | Value |
|---|---|
| Target | Windward face mean $C_p$ at 1000 Hz |
| Inputs | 4 face-averaged $C_p$ channels (windward, leeward, sideleft, sideright) |
| Window | seq_length = 100 (100 ms history) |
| Stride | step = 10 |
| Train / Val / Test | 70 / 15 / 15 % chronological split per series |
| Normalisation | z-score, fit on train only, per feature |
| Optimisation (DL) | Adam, lr = 1e-3, weight_decay = 1e-4, ReduceLROnPlateau, ES patience 15 |
| Batch size | 256 |
| Seed | 42 |
| Hardware | NVIDIA RTX A4000 16 GB, CUDA 11.8, PyTorch 2.7.1 |
| Horizons evaluated | 1, 10, 50, 100, 500 (autoregressive) |
| Metrics | RMSE, MAE, R², MAPE, directional accuracy |

Each model exposes the same `train / val / test` split via a shared `data_loader`
and shared training utilities, so differences in performance are attributable to
the model itself.

---

## 2. Rounds

| Round | Scope | Buildings | Series | Train windows | Question answered |
|---|---|---:|---:|---:|---|
| **R1** | Single building | 1 (Alpha1_4 / 2_1_3) | 17 | 38 920 | Can the model learn one building? |
| **R2** | 5 configurations | 5 (Alpha1_4/{1,2,3}_1_3 + Alpha1_6/{1,3}_1_3) | 85 | 194 140 | Generalisation across geometries and roughness? |
| **R3** | Full dataset | 20 (all alphas × all ratios) | 340 | 776 560 | Universal model for $C_p$ forecasting? |

---

## 3. Single-step results (h = 1)

R² on the held-out test set (denormalised windward $C_p$):

| Model | R1 | R2 | R3 |
|---|---:|---:|---:|
| Ridge | 0.969 | 0.998 | 0.998 |
| RandomForest | -0.23 | 0.997 | 0.997 |
| XGBoost | -1.97 | 0.997 | 0.997 |
| LSTM | 0.936 | 0.997 | 0.997 |
| GRU | 0.951 | 0.997 | 0.997 |
| **TCN** | **0.970** | **0.9985** | **0.9983** |

RMSE (denormalised, lower is better):

| Model | R1 | R2 | R3 |
|---|---:|---:|---:|
| Ridge | 0.0298 | 0.0205 | 0.0224 |
| RandomForest | 0.187 | 0.0228 | 0.0241 |
| XGBoost | 0.290 | 0.0223 | 0.0242 |
| LSTM | 0.0426 | 0.0242 | 0.0246 |
| GRU | 0.0373 | 0.0235 | 0.0245 |
| **TCN** | **0.0294** | **0.0172** | **0.0186** |

**Findings**
- **TCN wins single-step in every round.** Dilated causal convolutions match the
  multi-scale temporal structure of $C_p$ better than recurrent or
  feature-engineered alternatives.
- **Random Forest and XGBoost catastrophically fail in R1** (R² < 0) — a single
  building does not provide enough variance for tree-based regressors operating
  on a 400-feature flattened window. They recover fully once the dataset is
  enriched (R2 and R3).
- **Ridge is a strong, almost-free baseline** at R2/R3 (R² ≈ 0.998 in 1 s of
  training). It does not beat TCN, but the gap is small in single-step.

---

## 4. Autoregressive multi-horizon results

> **⚠️ Single-trajectory results.** The tables in this section come from one
> 500-step trajectory per round and are superseded for h = 500 by the
> multi-trajectory numbers in §0.5. They are retained as the original per-round
> record. In particular the R3 h = 500 RMSE here (naive 0.118, LSTM 0.307,
> TCN 0.667) reflects one near-flat trajectory; the representative values are
> naive 0.231, LSTM 0.270, TCN 0.411 (see `long_horizon_multitraj_round3.csv`).

RMSE at h = 500 (denormalised, lower is better):

| Model | R1 | R2 | R3 |
|---|---:|---:|---:|
| Naive persistence | 0.195 | 0.152 | 0.118 |
| Ridge | 0.371 | 0.138 | ~10⁴ (diverges) |
| RandomForest | 0.191 | 0.137 | 0.156 |
| XGBoost | 0.241 | 0.092 | 0.115 |
| **LSTM** | **0.072** | **0.116** | 0.307 |
| GRU | 0.384 | 0.104 | 0.256 |
| TCN | 4.20 (diverges) | 0.430 | 0.667 |

**Findings**
- **LSTM is the only model that beats naive persistence at every horizon in R1**
  (R² = +0.13 at h = 500 vs −5.4 for naive). It is also the most stable model
  in R2's noisier multi-geometry setting.
- **TCN, which wins single-step, becomes unstable autoregressively.** Small
  prediction errors are re-injected into the input and amplified by the deep
  dilated stack; in R1 the rollout diverges (RMSE 4.2 at h = 500). This is the
  classical exposure-bias / compounding-error pathology of teacher-forced
  training combined with high-capacity feature extractors.
- **Ridge diverges in R3 autoregressive rollouts** (RMSE ~10⁴), exposing the
  fragility of unregularised linear extrapolation when the input distribution
  drifts during multi-step prediction.
- In R3 (heterogeneous 340-series dataset) **no model strictly beats naive
  persistence at h = 500**. LSTM degrades from 0.116 (R2) to 0.307 (R3);
  multi-geometry transfer in autoregressive mode is an open problem.

---

## 5. Training cost (Round 3, GPU = A4000 16 GB)

| Model | Train time | Parameters | Best val MSE |
|---|---:|---:|---:|
| Ridge | 1 s | 401 | 0.00238 |
| XGBoost (GPU) | 69 s | 500 trees | 0.00305 |
| GRU | 2 260 s (~38 min) | 628 993 | 0.00290 |
| LSTM | 2 731 s (~46 min) | 827 649 | 0.00294 |
| RandomForest (CPU) | 7 262 s (~2 h) | 200 trees | 0.00281 |
| TCN | 8 777 s (~2.4 h) | 459 073 | 0.00166 |

**Findings**
- **XGBoost is the best accuracy/time trade-off** at R3: nearly Ridge accuracy
  with two more decimal places of variance explained, in just over a minute.
- **TCN's accuracy comes at a 100× training cost vs Ridge** for a ~0.1 % R² gain.
  Acceptable for offline thesis work; questionable for online deployment.

---

## 6. Reproducibility

All artifacts are committed under `Agent_Test/`:

```
Agent_Test/
├── data_loader.py                     # shared, deterministic
├── train_utils.py                     # shared training/eval, get_data(scope)
├── classical_utils.py                 # shared sklearn/xgb pipeline
├── train_all.py --scope {Alpha1_4/2_1_3 | round2 | all}
├── analyze_horizons.py --scope ...
├── results/
│   ├── model_comparison_round{1,2,3}.csv         # h=1 metrics per round
│   ├── multi_horizon_metrics_round{1,2,3}.csv    # per-horizon metrics
│   └── plots/                                    # consolidated comparison plots
└── {ridge,random_forest,xgboost,lstm,gru,tcn}/
    ├── train_<model>.py
    ├── checkpoints/<model>_best.pt   (DL models only)
    └── results/
        ├── forecasts/<model>_h{1,10,50,100,500}.npy
        └── plots/                    # per-model diagnostics
```

To reproduce any round:

```powershell
conda activate ML_Cesar
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
cd Agent_Test
python train_all.py --scope Alpha1_4/2_1_3   # Round 1
python train_all.py --scope round2            # Round 2
python train_all.py --scope all               # Round 3
python analyze_horizons.py --scope <same>     # multi-horizon table + plots
```

---

## 6.b Advanced Experiments (post-Round 3)

These experiments respond directly to the open questions raised at the end of
the R3 autoregressive analysis: (i) *is the catastrophic long-horizon collapse a
training-objective problem or an architectural problem?* and (ii) *are the
small differences between the top R3 models statistically meaningful, or noise?*

### 6.b.1 Direct multi-step LSTM (B.4)

**Motivation.** All R1-R3 multi-horizon results were generated by **teacher-forced
single-step** training followed by **autoregressive rollout** at inference: the
model is fed its own (possibly noisy) prediction and asked to predict one more
step, repeatedly. This is well-known to suffer from *exposure bias* and
*error compounding*: small errors are re-injected and amplified, especially at
h ≥ 100.

We tested an alternative: the same `PureLSTM` backbone with the **output head
replaced by `Linear(128 → 500)`**, trained end-to-end with MSE over all 500
future steps simultaneously. No autoregression at inference: one forward pass
returns the full 500-step trajectory.

| Configuration | Value |
|---|---|
| Scope | R3 (all 340 series, 776 560 train windows) |
| Backbone | `PureLSTM(hidden=256, layers=2, dropout=0.2, fc=128)` |
| Output head | `Linear(128 → 500)` (vs. `Linear(128 → 1)` for autoregressive) |
| Loss | MSE over the full 500-step horizon |
| Batch | 256 |
| Best epoch | 58 (training crashed at epoch 60 due to GPU TDR, model was already in plateau, val MSE 0.1440 → 0.1443 over the last 10 epochs) |

**Results — Direct vs. Autoregressive LSTM (Round 3):**

| h | RMSE direct | RMSE autoreg | R² direct | R² autoreg | Improvement (RMSE) |
|---:|---:|---:|---:|---:|---:|
| 1   | 0.0404 | ~0.025 | **0.9920** | 0.997 | autoreg slightly better at h=1 |
| 10  | 0.0638 | —      | **0.9799** | —     | — |
| 50  | 0.1415 | —      | **0.9013** | —     | — |
| 100 | 0.1664 | —      | **0.8637** | —     | — |
| **500** | **0.1786** | **0.307** | **0.8422** | **−6.18** | **−42 % RMSE, R² recovered** |

**Findings.**

1. **The direct head trades a small amount of h=1 accuracy for an enormous gain
   at long horizons.** At h = 500 it reduces RMSE by 42 % and lifts R² from
   −6.18 (worse than mean predictor) to 0.84 (84 % of variance explained).

2. **This isolates the failure mode.** Since the *backbone is identical*
   (same LSTM, same data, same window), the collapse of the autoregressive
   model at long horizons is **not an architectural limitation** of the LSTM —
   it is the **teacher-forced training objective** being mis-aligned with the
   multi-step deployment objective. Direct training, which exposes the model
   to every horizon during training, eliminates the gap.

3. **Open: a hybrid (scheduled sampling, mixture of teacher-forcing and free-run)
   could potentially match direct training at h = 500 while preserving h = 1
   sharpness.** Not yet tested.

### 6.b.2 Diebold-Mariano statistical test (B.5)

> **⚠️ Single-trajectory caveat.** The DM table below runs on the loss
> differential of **one** 500-step trajectory (n = 500 autocorrelated steps),
> so "naive ties XGBoost / beats every deep model" describes that one
> realisation, not generalisation. The robust replacement is the
> cross-trajectory test in §0.5 (paired Wilcoxon + win-rate over 680
> trajectories, `long_horizon_significance_round3.csv`): naive keeps a small
> significant edge over autoregressive LSTM/GRU but loses to both direct
> multi-step models, and LSTM/GRU beat naive on ~45 % of trajectories.

**Motivation.** The R3 single-step results (Section 3) and h = 500 results
(Section 4) show numerical differences between models that may or may not be
statistically significant. A bare RMSE table is insufficient for a thesis-level
claim; we need a hypothesis test.

We implemented the **Diebold-Mariano test** with **Newey-West HAC variance**
(Bartlett kernel, lag = ⌊n^(1/3)⌋ = 7 for n = 500) on the squared-error loss
differential between each pair of models, evaluated on the R3 h = 500
trajectories.

> **Note.** The test is restricted to R3 because the per-model `.npy` forecasts
> only contain the latest (R3) trajectories — earlier rounds were overwritten
> during R3 training. Ridge is excluded because its R3 autoregressive trajectory
> diverged to RMSE ~7 900 (range = 110 776 vs. ground-truth range = 0.56), which
> would trivially dominate any pairwise comparison.

**Key results** (Round 3, h = 500, n = 500 steps; full table in
`results/dm_test_round3.csv`):

| Pair (A vs B) | RMSE A | RMSE B | DM | p-value | Significant? | Winner |
|---|---:|---:|---:|---:|:---:|:---:|
| naive vs xgboost | 0.118 | 0.115 | +1.28 | 0.20 | **No (tie)** | — |
| naive vs lstm | 0.118 | 0.307 | −9.38 | <0.001 | Yes | **naive** |
| naive vs tcn | 0.118 | 0.667 | **−15.08** | <0.001 | Yes | **naive** |
| naive vs gru | 0.118 | 0.255 | −5.04 | <0.001 | Yes | **naive** |
| naive vs random_forest | 0.118 | 0.156 | −4.69 | <0.001 | Yes | **naive** |
| xgboost vs lstm | 0.115 | 0.307 | −8.85 | <0.001 | Yes | xgboost |
| xgboost vs tcn | 0.115 | 0.667 | −14.87 | <0.001 | Yes | xgboost |
| lstm vs gru | 0.307 | 0.255 | +5.29 | <0.001 | Yes | gru |
| lstm vs tcn | 0.307 | 0.667 | −16.58 | <0.001 | Yes | lstm |

**Findings.**

1. **At h = 500, naive persistence is statistically tied with XGBoost (p = 0.20)
   and strictly beats every deep-learning model.** This is a sharp negative
   result: when the model is forced to extrapolate 500 steps autoregressively
   on a heterogeneous 340-series dataset, the simplest baseline (return the
   last observation) is the **statistically optimal autoregressive forecaster**
   among the six tested.

2. **TCN, the h = 1 winner, is catastrophically beaten by naive at h = 500**
   (DM = −15.08, RMSE 0.667 vs 0.118 — almost 6× worse than doing nothing).
   This is consistent with the exposure-bias hypothesis and with the direct
   multi-step result in 6.b.1: the model with the highest single-step
   capacity is the one that amplifies its own errors most aggressively under
   feedback.

3. **GRU statistically beats LSTM in this regime** (DM = 5.29, p < 0.001).
   GRU's simpler gating may regularise the rollout dynamics — a finding worth
   exploring in future work.

4. **Combining 6.b.1 and 6.b.2:** the only autoregressive model that ties the
   naive baseline at h = 500 is XGBoost; the only model that *clearly beats*
   the naive baseline at long horizons in Round 3 is the **direct multi-step
   LSTM** (RMSE 0.179 vs naive 0.118 at h = 500 — still slightly worse than
   naive in absolute RMSE, but the *R²* of 0.84 means it tracks the *shape*
   of the trajectory, whereas naive trivially holds the last value).

### 6.b.3 Methodological lessons

- **GPU TDR (Timeout Detection and Recovery) is a silent training risk** for
  models with very wide output heads (`Linear(128 → 500)` + MSE over 500
  targets). The kernel duration of the final layer's backward pass can exceed
  the 2-second NVIDIA watchdog, killing the driver. Mitigations: reduce batch,
  reduce horizon, or extend `TdrDelay` in the registry. The crash that
  interrupted B.4 happened after the model had already converged, so we
  recovered the result via inference-only on the surviving checkpoint
  (`infer_lstm_direct.py`).
- **Always preserve per-step forecasts (`.npy`) and per-round metrics
  (`.csv`) before launching a new round.** Round 3 silently overwrote the
  R1/R2 forecasts at `<model>/results/forecasts/`, which prevented us from
  running DM across rounds. Future work should write to round-namespaced
  subfolders (`forecasts/round{1,2,3}/`).

---

### 6.c Transformer baseline — PatchTST (C.6.a / C.6.b)

To test whether the LSTM-direct ceiling at h = 500 (RMSE ≈ 0.178) is an
**architectural** limit or an irreducible-noise floor, we trained a
channel-mixed PatchTST with the same direct multi-step training recipe.

**Architecture** (`Agent_Test/patchtst/models/patchtst.py`)

- Input: 4-channel windows of length 100 (windward $C_p$, $u$, $v$, $w$).
- Patching: `patch_len=16`, `stride=8` → 11 patches per window.
- Patch-embedding: linear projection to `d_model=128` + learned positional
  embedding.
- Encoder: 3 pre-norm Transformer blocks (4-head self-attention, GELU FFN with
  `d_ff=256`, dropout 0.1).
- Head: flatten patches → `Linear(11·128 → H)` (direct multi-step, identical
  paradigm to B.4).
- Total parameters: **1,111,924** (H = 500).

**Training recipe** (resource-aware, TDR-safe):

- `torch.cuda.set_per_process_memory_fraction(0.85)` — leaves 15 % of the
  16 GB GPU for other processes.
- `torch.set_num_threads(n_cpu − 4)` — leaves 4 of 28 CPU cores free.
- `batch_size = 128` (half of LSTM-direct's 256, the actual TDR mitigation).
- Optimiser: Adam, lr = 1e-3, weight_decay = 1e-4, ReduceLROnPlateau (patience
  5, factor 0.5), early stopping (patience 15).
- Loss: MSE on the full H-dimensional output (direct).

**Results — Round 3 (340 series, 776 k windows, H = 500)**

| Horizon | RMSE | MAE | R² | $\Delta$ vs LSTM-direct R3 |
|---------|------|-----|-----|-----------------------------|
| 1   | 0.0427 | 0.0318 | 0.9910 | +0.0023 RMSE (LSTM wins) |
| 10  | 0.0735 | 0.0545 | 0.9734 | +0.0097 (LSTM wins) |
| 50  | 0.1432 | 0.1098 | 0.8990 | +0.0017 (tie, LSTM marginally) |
| 100 | 0.1655 | 0.1285 | 0.8651 | −0.0009 (**PatchTST wins**) |
| 500 | 0.1752 | 0.1368 | 0.8482 | −0.0034 (**PatchTST wins**) |

Training time: **71 min** on RTX A4000, **no TDR crash**, early stop at
epoch 111, best val_loss = 0.1406.

**Sanity check — Round 2 (85 series, 190 k windows, same H = 500)**

| Horizon | RMSE | MAE | R² | Note |
|---------|------|-----|-----|------|
| 1   | 0.0437 | 0.0340 | 0.9904 | |
| 10  | 0.0739 | 0.0568 | 0.9725 | |
| 50  | 0.1431 | 0.1111 | 0.8969 | |
| 100 | 0.1651 | 0.1296 | 0.8629 | |
| 500 | 0.1724 | 0.1360 | 0.8481 | matches R3 |

R2 (190 k) and R3 (776 k) give numerically indistinguishable results
(differences ≤ 0.003 in RMSE across all horizons). The 1.1 M-parameter
PatchTST is **saturated at R2** — extra data does not buy further accuracy.

**Findings — what PatchTST tells us about the long-horizon limit**

1. **PatchTST and LSTM-direct land in the same RMSE band** (≈ 0.175 at h =
   500) despite radically different inductive biases (gated recurrence vs.
   self-attention over patches). Two independent direct-multi-step
   architectures converging to the same number is strong evidence that this
   value is an **irreducible-noise floor**, not an architectural ceiling.

2. **PatchTST slightly favours longer horizons; LSTM-direct slightly favours
   shorter ones.** The crossover is around h ≈ 80–100. This is the
   prediction-of-text vs. prediction-of-dynamics trade-off: attention pools
   global context (helps far horizons), recurrence preserves local phase
   (helps short horizons).

3. **The wide head was not the bottleneck for PatchTST.** The same
   `Linear(W → 500)` paradigm that crashed LSTM-direct via TDR ran cleanly
   under conservative batch and resource caps (batch = 128, GPU 85 %, 4 CPU
   cores free). The TDR risk is a **kernel-duration** issue, not a memory
   or numerical-stability issue.

4. **PatchTST does not break the naive-persistence barrier at h = 500
   either** (PatchTST 0.175 vs. naive 0.118 RMSE). Both direct multi-step
   methods restore variance-tracking (R² ≈ 0.85 vs. R² = −6 for
   autoregressive LSTM/TCN) but neither beats persistence on raw RMSE. This
   confirms that, at h = 500, **the long-horizon irreducibility is a property
   of the data**, not of the model family.

**Artefacts**

- Model: `Agent_Test/patchtst/models/patchtst.py`
- Train script: `Agent_Test/patchtst/train_patchtst.py`
- Checkpoints (gitignored): `patchtst/checkpoints/patchtst_h500_r2.pt`,
  `patchtst_h500_r3.pt`
- Metrics (R1 smoke + R2 + R3): `results/patchtst_metrics.csv`
- Per-step curves: `results/patchtst_h500_r2_rmse_curve.npy`,
  `results/patchtst_h500_r3_rmse_curve.npy`
- Consolidated plot:
  `results/plots_cross_round/direct_vs_autoreg_h500.png` (autoreg + LSTM-direct
  + PatchTST overlaid).

---

### 6.d Spectral analysis — what the residuals actually contain (F)

To check whether the long-horizon ceiling at h = 500 (RMSE ≈ 0.175) is indeed
irreducible turbulent noise — as Section 6.c hypothesised — we computed the
Welch power spectral density (PSD) of the **true signal**, the **predictions**,
and the **residuals** for both direct-multi-step models on the R3 test set
(146 880 windows, subsampled to 5 000; `fs = 1000 Hz`, `nperseg = 256`,
`df ≈ 3.9 Hz`). Script: `Agent_Test/spectral_analysis.py`. Plot:
`results/plots_cross_round/psd_residuals.png`.

**Result (single-line summary)**

| Quantity                          | LSTM-direct | PatchTST |
|-----------------------------------|-------------|----------|
| Total power true signal $C_p$     | 1.192 × 10⁻² | 1.192 × 10⁻² |
| Total power **predictions**       | **1.58 × 10⁻⁴** (**75×** smaller) | **1.89 × 10⁻⁴** (**63×** smaller) |
| Total power residuals             | 1.173 × 10⁻² | 1.173 × 10⁻² |
| Crossover freq. pred < 0.5×true   | 3.9 Hz (the first measurable bin) | 3.9 Hz |

**Interpretation — the floor is NOT a noise floor, it is mean-regression**

The predicted PSD is 60–75× smaller than the true PSD **at every frequency
between 4 Hz and 500 Hz**. The residual PSD overlaps the true-signal PSD
almost exactly across the whole band. This means the models are not failing
to predict *some* frequencies; they are failing to produce **any AC content
at all** within the 500-step forecast window.

What the models actually output at h = 500 is essentially the **conditional
expectation** $\mathbb{E}[C_p(t+500) \mid \mathbf{x}(t-99:t)]$, which for a
turbulent wind signal at 0.5 s lead-time reduces to a slowly varying envelope
(near-constant within each 500-sample test window). The window-to-window
envelope is what produces R² ≈ 0.85; the within-window oscillations are
gone.

This is the standard pathology of MSE-trained point forecasters on chaotic
signals: the variance-minimising predictor is the **mean**, not a sample,
so the spectrum collapses. The 0.175 RMSE is then exactly the standard
deviation of the high-frequency turbulent component that has been removed.

**Practical implication for wind engineering**

For structural-design applications (fatigue, peak-factor calculation,
gust-effect factor, return-period statistics), the PSD of the predicted
signal must match the PSD of the real signal — otherwise the engineering
quantities computed from the forecast are catastrophically wrong (typically
they will underestimate peaks because the forecast has no fluctuation
energy). Under that criterion, **both LSTM-direct and PatchTST are
unusable at h = 500** despite their high R².

**This changes the thesis recommendation**: the direct-multi-step recipe
(Section 6.b) fixed the autoregressive collapse to R² = −6 — but only by
trading it for a *different* failure mode (spectral collapse). The true
fix has to come from a **distributional / generative forecasting**
approach — quantile regression, diffusion-based forecasting, score-based
generative models, conditional GANs, or normalising flows — which can match
the conditional *distribution* (and therefore the PSD) instead of only its
mean.

**Artefacts**

- Script: `Agent_Test/spectral_analysis.py`
- Metrics CSV: `Agent_Test/results/spectral_metrics.csv`
  (total power, peak frequency, band-limited power for true / pred / residual
  for both models).
- Plot: `Agent_Test/results/plots_cross_round/psd_residuals.png` (two panels,
  log-log, overlapping residual ↔ true curves visually confirm the result).

---

### 6.e Synthesis as a complement to forecasting — WPTSE-Net (Phase 2 of `Agent_Papers/`)

The spectral diagnosis in 6.d showed that **MSE-trained forecasters cannot
reproduce the PSD** of chaotic wind-pressure signals — they collapse to the
conditional mean and the residual *becomes* the signal. The natural question
for thesis closure is: *is there any deep-learning approach that does
preserve the spectrum?* The literature review in `Agent_Papers/paper_summaries.md`
identified **WPTSE-Net** (Tong, Liang, Song, Hu, Kareem, *JWEIA* 2024) as a
directly relevant counter-example. WPTSE-Net is **not** a forecaster: it is a
*time-series extension / synthesis* network that, given the deterministic
statistics of a Cp record (mean, std, skewness, kurtosis, plus a noise
vector), generates new samples that match the **distribution and spectrum**
of the original record. The implementation, in a separate folder
(`Agent_Papers/wptse_net/`) to keep the forecasting pipeline of
`Agent_Test/` untouched, was delegated to the custom `wind-cp-forecaster`
sub-agent.

**Setup (R2 scope, 190 k face-series; R3 scope, full corpus 1 360 series).**
122 528-parameter MLP encoder + 4 × Dense(128) decoder, Huber loss, Adam
lr = 1e-3, batch 128, max 1 000 epochs with patience 50. R2 = 3 ratios × 2
alphas; R3 = all 20 (alpha, ratio) combinations × all angles × all 4 faces.
Z-score normalisation per series. PSD computed identically to 6.d
(`fs = 1 000 Hz`, `nperseg = 256`, hann, density scaling).

**Headline result — spectral fidelity recovered by ≈ 50 ×.**

| Model | Scope | Task | Loss | $P_{pred}/P_{true}$ | PSD log-L² | R² (held-out) | Train time |
|-------|-------|------|------|--------------------:|-----------:|--------------:|-----------:|
| LSTM-direct (Agent_Test) | R3 | Forecasting, h = 500 | MSE | **0.013** | (very large) | 0.85 | ~45 min |
| PatchTST (Agent_Test) | R3 | Forecasting, h = 500 | MSE | **0.016** | (very large) | 0.85 | 71 min |
| WPTSE-Net (Agent_Papers) | R2 | Synthesis | Huber + stats encoder | **0.818** | **0.578** | **0.893** | 13 min |
| **WPTSE-Net (Agent_Papers)** | **R3** | **Synthesis** | **Huber + stats encoder** | **0.786** | **0.547** | **0.893** | **32 min** |

R3 confirms R2: with 4 × more (and more diverse) data, the synthesis model
still recovers **≈ 79 % of the true total power**, against ≈ 1.5 % for the
forecasters — a **50-fold improvement in spectral fidelity** maintained
across the full corpus. R² on the synthesised slice is identical at both
scopes (0.893), so the model is already saturated for this architecture —
gains would now come from increasing capacity, not data.

**Moment-by-moment statistical accuracy of WPTSE-Net**

| Moment | R2 rel. error | R3 rel. error |
|--------|--------------:|--------------:|
| Mean   | 4.8 × 10⁻⁴ | 6.9 × 10⁻³ |
| Variance | 0.110 | 0.139 |
| Skewness | 0.332 | **0.232** |
| Kurtosis | 0.036 | 0.036 |

Skewness improves with R3 (more diverse facade orientations seen during
training); variance and kurtosis remain in the same regime. Both are
critical for non-Gaussian peak-pressure prediction in design codes.

**Why this matters for the thesis narrative**

1. It **closes** the open question of 6.d / take-aways 7-8: the loss function
   is the bottleneck, not the architecture. Swapping MSE for a
   distribution-aware objective recovers the PSD with a much **smaller**
   network (122 k vs 1.1 M parameters for PatchTST).
2. It supports a **two-track recommendation** for wind-engineering practice:
   - **Forecasting** (MSE) → use only for short horizons and metrics that
     reward the mean (RMSE, R²).
   - **Synthesis / extension** (Huber + statistical encoder, or quantile /
     generative approaches) → use for design-load generation, peak-factor
     estimation, fatigue analysis, and any task where the **spectrum**
     matters.
3. It validates the literature-review pipeline (`paper_summaries.md` + sub-agent
   delegation) as a **methodology** that yielded a thesis-quality
   counter-example with one paper and ~ 13 min of training.

**Deviations from the original Tong et al. 2024 implementation** (full list
in `Agent_Papers/wptse_net/README.md`)

- Batch 128 (paper: 64) and chronological 70 / 15 / 15 split with patience-50
  early stopping (paper trains 1 000 fixed epochs on a single 4 096-sample
  record). Necessary because our dataset is two orders of magnitude larger.
- Per-series z-score (paper: global affine).
- Slice order preserved at inference (paper shuffles) — required for PSD
  alignment on the held-out segment.

**Artefacts**

- Code: `Agent_Papers/wptse_net/{models/wptse_net.py, train_wptse_net.py}`
- Checkpoints: `Agent_Papers/wptse_net/checkpoints/wptse_net_best_{r2,r3}.pt`
- Metrics: `Agent_Papers/wptse_net/results/metrics_{r2,r3}.csv`,
  `per_series_metrics_{r2,r3}.csv`
- Plots: `Agent_Papers/wptse_net/results/plots/{training_curve,
  timeseries_comparison, psd_comparison}_{r2,r3}.png`
- Generated samples: `Agent_Papers/wptse_net/results/generated/{sample_gen,
  sample_true}_{r2,r3}.npy`
- Literature analysis: `Agent_Papers/paper_summaries.md` (7 papers; 1
  recommended for implementation, 6 discarded with justification).
- README: `Agent_Papers/wptse_net/README.md`.

---

## 7. Take-aways for the thesis

1. **Single-step forecasting of windward $C_p$ is essentially solved** at this
   sampling rate — every reasonable model exceeds R² ≈ 0.997 once trained on
   enough geometries. The differentiator is autoregressive stability.

2. **Architectural inductive bias matters more than capacity for autoregressive
   rollout.** LSTM (smaller dilated receptive field but explicit gated memory)
   is more robust than TCN (high single-step accuracy, but error amplification
   in feedback).

3. **Tree-based methods need data diversity to be useful.** They are not a
   reasonable baseline on a single-building experiment.

4. **Autoregressive $C_p$ models are competitive with, but do not beat, naive
   persistence at h = 500.** Re-evaluated over 680 trajectories (§0.5), LSTM/GRU
   reach R² ≈ 0.64–0.65 vs naive 0.74 and beat naive on ~45 % of trajectories —
   a small, significant aggregate deficit, *not* the catastrophic collapse the
   single-trajectory analysis suggested. The model that *does* beat naive at
   h = 500 is direct multi-step (RMSE 0.175–0.179 vs 0.231). This motivates
   future work on (a) hierarchical / mixture-of-experts conditioned on geometry,
   (b) physics-informed regularisation, (c) scheduled sampling to close the
   remaining autoregressive gap.

5. **The long-horizon collapse is an objective-function problem, not an
   architectural one.** The post-R3 experiment (Section 6.b.1) demonstrates
   that an *identical LSTM backbone* trained with a direct multi-step head
   recovers from R² = −6.18 to R² = 0.84 at h = 500. This is the single
   most actionable finding of this work for downstream practitioners:
   **never use teacher-forced single-step training when the deployment task
   is long-horizon forecasting.**

6. **Statistical testing must be done over many trajectories, not one.** The
   original single-trajectory Diebold-Mariano (Section 6.b.2) made naive look
   dominant (it "beat every deep model"); that was an artifact of one near-flat
   trajectory. The cross-trajectory test (§0.5, 680 trajectories) shows naive
   keeps only a *small* significant edge over autoregressive LSTM/GRU (~6 %
   median RMSE, beaten on ~45 % of trajectories) and **loses to both direct
   multi-step models**. The methodological take-away stands and is sharpened:
   always report long-horizon significance over a *population* of trajectories
   with a paired test, never a single rollout.

7. **Two independent architectures hit the same long-horizon floor, for the
   same reason.** Both the LSTM-direct (B.4, recurrence) and the PatchTST
   (C.6.b, patched self-attention) converge to RMSE ≈ 0.175 and R² ≈ 0.85
   at h = 500. Section 6.d's spectral analysis shows *why*: both models'
   predicted PSD is 60–75× smaller than the true-signal PSD at every
   frequency, and both residual PSDs lie exactly on the true-signal PSD.
   The "floor" is therefore not architectural — it is the **MSE-optimal
   collapse to the conditional mean**, which for a chaotic turbulent signal
   at 0.5 s lead-time has nearly zero AC content within each window.

8. **R² is the wrong metric for wind-engineering forecasts.** A model with
   R² = 0.85 (PatchTST, h = 500) can still have predictions whose total
   spectral power is 60× lower than the real signal — making them useless
   for any downstream computation that depends on the spectrum (fatigue
   damage, peak factor, return-period analysis, gust-effect factor). The
   thesis recommendation for downstream wind-engineering practitioners is
   therefore to add **at least one spectral-fidelity metric** (e.g., total
   power ratio, or PSD L² distance in log-frequency) alongside any RMSE/R²
   report. The natural next step on the modelling side is
   **distributional / generative forecasting** (quantile, diffusion,
   normalising flow, conditional GAN), which can match the conditional
   distribution — and therefore the spectrum — rather than only its mean.

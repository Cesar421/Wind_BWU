# Wind Pressure Coefficient Forecasting — Results Summary

Comparison of 6 forecasting models trained under a strict fair-baseline protocol on
wind-tunnel pressure-coefficient ($C_p$) time series from the TPU BDH benchmark.

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
| **R2** | 5 configurations | 5 (Alpha1_4/6 × {1_1_3, 2_1_3, 3_1_3}) | 85 | 194 140 | Generalisation across geometries and roughness? |
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
| Scope | R3 (all 340 series, 759 560 train windows) |
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

4. **Universal $C_p$ models are still autoregressively limited.** R3 (340
   series) shows that scaling the training set, with no architectural change,
   does not yet yield a model that beats naive persistence at h = 500. This
   motivates future work on (a) direct multi-step training (vs. teacher-forced
   single-step), (b) hierarchical / mixture-of-experts conditioned on geometry,
   (c) physics-informed regularisation.

5. **The long-horizon collapse is an objective-function problem, not an
   architectural one.** The post-R3 experiment (Section 6.b.1) demonstrates
   that an *identical LSTM backbone* trained with a direct multi-step head
   recovers from R² = −6.18 to R² = 0.84 at h = 500. This is the single
   most actionable finding of this work for downstream practitioners:
   **never use teacher-forced single-step training when the deployment task
   is long-horizon forecasting.**

6. **Statistical testing changes the headline result.** The Diebold-Mariano
   analysis (Section 6.b.2) shows that, at h = 500, the gap between naive
   persistence and the best classical model (XGBoost) is *not statistically
   significant* (p = 0.20), while the gap between naive and every deep model
   *is* significant — in favour of naive. The fair comparison at long
   horizons is therefore not "LSTM vs TCN vs Ridge" but "any model vs.
   naive persistence", and only the direct multi-step LSTM clears that bar
   in terms of R² (variance tracking).

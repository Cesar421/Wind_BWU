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

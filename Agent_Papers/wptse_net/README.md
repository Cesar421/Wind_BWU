# WPTSE-Net — Wind Pressure Time-Series Encoder/Generator

Implementation of the deterministic latent-stats generator described in:

> **Tong, Z., Liang, R., Song, R., Hu, G., Kareem, A.** (2024).
> *Machine learning for wind engineering: A review.*
> **J. Wind Eng. Ind. Aerodyn., 254, 105909.**
> (See `Agent_Papers/_extracted/Kareem_ML_JWEIA_2024.txt`, section on the
> Wind-Pressure Time-Series Encoder Network, lines ≈ 380–430.)

The paper proposes a small fully-connected decoder that maps a 5-D latent
vector `[mean, var, skew, kurt, noise·0.1]` of a 32-sample slice back to the
slice's 32 raw samples. A normalized 10-min wind-pressure trace (4 096
samples) is partitioned into `l = 128` non-overlapping slices of length
`4096 / l = 32`. The decoder is trained slice-wise; at inference the slice
sequence is concatenated to reconstruct/synthesize a full series.

---

## Architecture (matches paper)

```
slice s ∈ R^32
   │
   ▼
encode_slice(·)        →  z = [μ, σ², m3/σ³, m4/σ⁴, η·0.1] ∈ R^5,  η ~ N(0,1)
   │
   ▼
DecoderBlock × 4       Dense(128) → ReLU → Dense(128) → ReLU
                       → BatchNorm1d(128)
                       → x * γ + β            (γ multiplicative, β additive)
   │
   ▼
Linear(128 → 32)       →  ŝ ∈ R^32
```

* Loss : **Huber (δ = 1)**
* Opt. : **Adam, lr = 1e-3**
* Slice : **32 samples**, non-overlapping
* Params : **122 528** (verified)

File: [models/wptse_net.py](models/wptse_net.py)

---

## Training data

* Source : `Data/Data_All_The_BDH_PostProcess/`
  loaded through the **shared** `Agent_Test/data_loader.py`
  (this folder reads it via `sys.path.insert`; no Agent_Test file is touched).
* Scope : `round2` ⇒ ratios `{1_1_3, 2_1_3, 3_1_3}` × alphas
  `{Alpha1_4, Alpha1_6}` × all wind angles on disk × 4 façades.
  Missing combos (e.g. `Alpha1_6/2_1_3`) are skipped via existence checks.
* Per-series **z-score** normalisation, μ/σ fit on the train portion only
  and stored for inverse-transform.
* **Chronological 70 / 15 / 15** split on the time axis (no shuffle).
* Sliced into **non-overlapping** 32-sample windows.

Final test set : **340 series**.

---

## Run

```powershell
conda activate ML_Cesar
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING     = "utf-8"
cd Agent_Papers\wptse_net
python -u train_wptse_net.py --scope round2 | Tee-Object train.log
```

GPU : NVIDIA RTX A4000, CUDA 11.8, PyTorch 2.7.1+cu118.

---

## Results — round 2 (340 test series)

| Quantity                       |     Value |
| ------------------------------ | --------: |
| Parameters                     |   122 528 |
| Training time                  |     779 s |
| Epochs run (early-stopped)     |       144 |
| Best val Huber                 |   0.05053 |
| RMSE   (slice, normalized)     |   0.31766 |
| MAE    (slice, normalized)     |   0.23761 |
| R²     (slice)                 |    0.8932 |
| PSD total-power ratio P_g/P_t  |    0.8178 |
| PSD log-L² distance            |    0.5775 |
| Mean rel. error                | 4.76 e-04 |
| Variance rel. error            |    0.1099 |
| Skewness rel. error            |    0.3316 |
| (raw) Kurtosis rel. error      |    0.0355 |

Artefacts under `results/` :

```
results/
├── metrics.csv                 # summary above
├── per_series_metrics.csv      # one row per test series
├── plots/
│   ├── training_curve.png
│   ├── timeseries_comparison.png   # 2-second snippet, gen vs true
│   └── psd_comparison.png          # Welch PSD, single + averaged
└── generated/
    ├── sample_gen.npy
    └── sample_true.npy
checkpoints/wptse_net_best.pt
```

---

## Deviations from the paper (with justification)

1. **Batch size 128** (paper uses 64). Required by the task spec for this
   run; doubling the batch is well within memory and changes loss only
   marginally for this tiny model.
2. **Training corpus**: paper trains on a *single* 10-minute 4 096-sample
   record; we train on the full BDH wind-tunnel corpus
   (340 test series + train/val) ≈ 32 768 samples per face. Two orders of
   magnitude more data → improves generalisation but makes per-series
   reconstruction harder (heterogeneous statistics).
3. **70 / 15 / 15 chronological split + early stopping** (patience 50) instead
   of the paper's fixed 1 000 epochs on a single record. Chosen to prevent
   over-fitting given the much larger / more diverse data and to match the
   protocol already used by all `Agent_Test/` thesis models.
4. **Slice order preserved** at inference (no shuffling) so PSD / time-series
   comparison aligns with the ground-truth time axis. The paper shuffles
   slices because it only cares about marginal stats; we want both
   statistics *and* spectra to be comparable on the held-out test segment.
5. **Per-series z-score normalisation** with statistics fit on the train
   portion only (paper uses one global affine map on its single record).
6. **Kurtosis** reported is the raw `m4 / σ⁴` (non-excess) used in the
   latent vector — same definition as the paper.
7. `torch.cuda.set_per_process_memory_fraction(0.85, 0)` — explicit device
   index `0` required by the PyTorch API on this machine.

No part of the BDH dataset, no file in `Agent_Test/`, and no file outside
`Agent_Papers/wptse_net/` was modified.

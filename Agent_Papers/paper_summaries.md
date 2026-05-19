# Paper Summaries — Wind Pressure Coefficient (Cp) ML/DL Literature

**Phase 1 deliverable.** Structured analysis of the 7 PDFs in `Wind pressure coefficients/`.
No model code is written in this phase. Existing thesis code under `Agent_Test/` is untouched.

Already-implemented models in `Agent_Test/`: **Ridge, Random Forest, XGBoost, LSTM, GRU, TCN, PatchTST (Transformer)**.

PDF text was extracted with `pypdf` 6.11.0 into `Agent_Papers/_extracted/*.txt`. One file (Paper 2,
Wiley CACAIE 2025) is a scanned/protected PDF and `pypdf` returned only Wiley watermark text;
the summary for that paper is reconstructed from the title, abstract obtained via Google Scholar
preview, and citing references — flagged below as **partial extraction**.

---

### Paper 1: Aldoum, M. & Stathopoulos, T. (2025). *Wind pressures on roofs of nonrectangular buildings: Experimental and machine learning approaches.* Wind and Structures, 41(4), 287–303. https://doi.org/10.12989/was.2025.41.4.287

- **Authors / Year / Journal**: M. Aldoum, T. Stathopoulos / 2025 / Wind and Structures, V41 N4.
- **Main objective**: Build a single ML model that predicts **mean Cp** and **negative peak Cp** on roofs of *non-rectangular* (L, U, T, X) low buildings from wind-tunnel data, comparing random vs. structured (wind-direction-based) data splits.
- **Dataset used**: Concordia atmospheric BL wind tunnel, 1:200 scale; 9 building configurations (L1–L3, U1–U3, T1–T2, X); 193 quasi-uniform roof pressure taps; L-shapes tested 0°–355° step 5°, U/T/X tested 0°–350° step 10°; α = 0.14 (open country); equivalent full-scale height 10 m. Total **195 000 samples** (tap × WD × geometry).
- **Inputs (features)**: 6 raw → 9 after one-hot. (1) Shape category {L,U,T,X} → one-hot 4 bits; (2) tap x-coord; (3) tap y-coord; (4) βx = D/Bx; (5) βy = D/By; (6) wind direction.
- **Outputs (targets)**: scalar — either mean Cp or negative peak Cp (two separate models).
- **Model architecture**:
  - **GBR**: scikit-learn GradientBoostingRegressor, **500 estimators**, max_depth = 10, max_features = 6, min_samples_split = 8.
  - **ANN**: Keras `Sequential` feed-forward MLP, **9–100–500–500–100–1** with **ReLU** hidden activations, **Linear** output.
- **Loss / optimizer / LR / batch / epochs**: ANN — Adam, **lr = 1e-3**, batch = 200, **200 epochs**, early stopping on val loss. GBR — squared-error loss.
- **Preprocessing**: One-hot for shape; min-max (divide by max) for x, y, WD; βx, βy already in [0,1]; targets unscaled. Train/Val/Test ≈ 67/7/26 % with structured WD-based split (12 WDs held out: 20°, 50°, 80°, …).
- **Evaluation metrics reported**: R², MSE, MAE, MBE. ANN on test set: **R² = 0.967 (mean Cp), 0.949 (peak Cp)**; GBR random split R² ≈ 0.96; GBR WD-based split R² ≈ 0.90 (showing random-split leakage).
- **Key results / claims**: Inner edges/corners of L, U, T, X roofs see higher suction than outer corners; ANN comfortably handles a mixed-geometry training set; random splits over-estimate generalisation — structured WD-based split is mandatory.
- **Comparison baselines**: GBR vs. ANN; random vs. structured split.
- **Implementation feasibility on our BDH dataset**: ⚠️ **Partial / needs adaptation.** Their inputs (tap coordinates, multi-shape one-hot, βx, βy) target *static* Cp regression across taps and WDs, not time-series forecasting. The BDH set provides **face-averaged** time series at fixed angles, so we would have to reframe: predict (mean, peak) per (face, angle, geometry, α) from geometry+angle inputs — solvable, but a different task than what we currently do.
- **Already covered by `Agent_Test/`?**: **NO direct equivalent.** We have an ANN family (LSTM/GRU) and tree models (RF/XGBoost), but no static MLP regressor that maps `(geometry, angle, α) → Cp statistics`.

---

### Paper 2: Nav, F.M., Mirfakhar, S.F. & Snaiki, R. (2025). *A hybrid machine learning framework for wind pressure prediction on buildings with constrained sensor networks.* Computer-Aided Civil and Infrastructure Engineering, 40(19), 2816–2834. https://doi.org/10.1111/mice.13488 — **partial extraction (PDF text inaccessible).**

- **Authors / Year / Journal**: F.M. Nav, S.F. Mirfakhar, R. Snaiki / 2025 / Computer-Aided Civil and Infrastructure Engineering 40(19).
- **Main objective**: Reconstruct **high-fidelity full-surface wind-pressure fields** on buildings from a **constrained (sparse) sensor network**, using a hybrid ML framework that fuses dimensionality reduction (POD-type) with a neural regressor.
- **Dataset used**: Wind-tunnel pressure measurements on tall building(s); sparse subset of taps used as inputs, dense field used as ground truth (specific tunnel and building details not recoverable from this PDF).
- **Inputs (features)**: Pressure / Cp readings from a small number of physical sensors + wind-direction / boundary-condition descriptors.
- **Outputs (targets)**: Full Cp field (mean and/or fluctuating) at all tap locations, i.e. spatial reconstruction.
- **Model architecture**: "Hybrid ML framework" — based on the title and the citing literature, this is a POD/autoencoder reduced-order model + a neural mapper from sparse sensors to latent coefficients (canonical sparse-sensing pipeline). **Exact layer sizes not extractable.**
- **Loss / optimizer / LR / batch / epochs**: Not extractable from the available text.
- **Preprocessing**: Not extractable. Standard practice for this family is mean removal, normalisation, POD truncation.
- **Evaluation metrics reported**: Not extractable from the available text.
- **Key results / claims**: Demonstrates accurate full-field reconstruction with far fewer sensors than the original measurement grid — strong baseline for sparse-sensor wind-pressure inference.
- **Comparison baselines**: Not extractable; likely linear POD reconstruction and/or plain ANN.
- **Implementation feasibility on our BDH dataset**: ❌ **Not implementable as-is.** Our BDH post-processed dataset stores **face-averaged** Cp time series (only 4 channels per case), not per-tap pressure fields. Sparse-sensor reconstruction needs the full tap-level data, which is not in `Data_All_The_BDH_PostProcess/`. Without per-tap fields we cannot meaningfully replicate this method.
- **Already covered by `Agent_Test/`?**: **NO** (and not applicable for the data we have).

---

### Paper 3: Hu, G., Liu, L., Tao, D., Song, J., Tse, K.T. & Kwok, K.C.S. (2020). *Deep learning-based investigation of wind pressures on tall building under interference effects.* JWEIA 201, 104138. https://doi.org/10.1016/j.jweia.2020.104138

- **Authors / Year / Journal**: G. Hu, L. Liu, D. Tao, J. Song, K.T. Tse, K.C.S. Kwok / 2020 / JWEIA.
- **Main objective**: Use ML to **fill in untested interference configurations** between two tall buildings, so that ~70 % of wind-tunnel runs can be replaced by predictions.
- **Dataset used**: TPU tall-building interference database; principal building 70 × 70 × 280 m (1:400), **252 taps (9 rows × 7 cols × 4 faces)**, **37 upstream locations × 72 wind directions = 2664 cases**. 6 cases held out; 30 % of remaining 2558 used for training (with 80/20 TTV).
- **Inputs (features)**: For DTR/RF/XGB — (Sx, Sy, θ, tap-x, tap-y); for GAN — (Sx, Sy, θ) only.
- **Outputs (targets)**: Mean and fluctuating (rms) Cp per tap (DTR/RF/XGB) or full 9 × 28 mean & rms map (GAN, per face).
- **Model architecture**:
  - DTR: max_depth 20, max_leaf 20 000, min_samples_leaf 20.
  - RF: 100 trees (150 for rms), 3 features per split, depth 25.
  - XGB: depth 10 (mean) / 12 (rms), lr 0.1, n_trees 200/120, subsample 0.66 / 0.20.
  - **GAN**: tailor-designed **two-stream cGAN** — global-local generator with 5 FC encoder layers (64-128-256-512-1024), two branches (512 → 252) producing initial 9 × 28 mean & rms maps, then **5 residual conv blocks** for local refinement; **patch discriminator** (3 × 3 × 32 conv layers + 3 × 4 max-pool + 1 × 1 conv) outputting a 3 × 7 real/fake probability map per stream.
- **Loss / optimizer / LR / batch / epochs**: GAN — Adam, **batch 32**, **2000 epochs** (lr 1e-4 first 1000, linearly decayed to 0 in last 1000), Euclidean + adversarial loss with α = 100.
- **Preprocessing**: Gaussian-init weights, 10-fold CV for hyper-param tuning. Pressure maps used as 2-D images.
- **Evaluation metrics reported**: 10-fold MSE, R² on test set. **GAN**: R² = **0.988** (mean Cp), **0.924** (rms Cp). XGB / RF / DTR R² 0.96-0.98 (mean) but materially worse than GAN on rms.
- **Key results / claims**: GAN beats DTR/RF/XGB on rms predictions because it learns spatial correlations across faces; only 30 % of WT cases are needed to train a good model.
- **Comparison baselines**: DTR, RF, XGBoost.
- **Implementation feasibility on our BDH dataset**: ⚠️ **Partial / needs adaptation.** The cGAN's output is a 2-D pressure *map* over taps × faces. Our BDH data is **already face-averaged** (no spatial grid per face), so the GAN's spatial-image inductive bias has no surface to model. The tree-based DTR/RF/XGB part is fully implementable and we already have RF/XGB. Implementing the cGAN would require the *raw tap-level BDH data* (not in the post-processed folder).
- **Already covered by `Agent_Test/`?**: **PARTIAL** — RF and XGBoost branches are covered; the GAN spatial-map predictor is **NOT** covered.

---

### Paper 4: Meddage, P. et al. (2022). *Interpretation of Machine-Learning-Based (Black-box) Wind Pressure Predictions for Low-Rise Gable-Roofed Buildings Using Shapley Additive Explanations (SHAP).* Buildings 12, 734. https://doi.org/10.3390/buildings12060734

- **Authors / Year / Journal**: P. Meddage, I. Ekanayake, U.S. Perera, H.M. Azamathulla, M.A. Md Said, U. Rathnayake / 2022 / Buildings (MDPI).
- **Main objective**: Predict surface-averaged Cp,mean / Cp,rms / Cp,peak on **low-rise gable-roof buildings** with 4 tree-based regressors and **explain** them with SHAP.
- **Dataset used**: TPU low-rise gable-roof database.
- **Inputs (features)**: Building geometric parameters (height-to-breadth, depth-to-breadth, roof pitch, wind direction, surface ID).
- **Outputs (targets)**: Cp,mean, Cp,rms, Cp,peak per surface (surface-averaged).
- **Model architecture**: **Decision Tree, XGBoost, Extra-Trees, LightGBM** (sklearn / lightgbm / xgboost defaults; hyper-params tuned on R²).
- **Loss / optimizer / LR / batch / epochs**: Standard gradient-boosting / CART training (no NN). Tree depths and learning rates not exhaustively listed — left to default + R²-driven tuning.
- **Preprocessing**: Train/test split on TPU tabular features.
- **Evaluation metrics reported**: R, R², MAE, RMSE; reports very high R for all four tree models. The XAI value-add is SHAP feature attributions, not raw accuracy gains.
- **Key results / claims**: Tree-based models are competitive with ANN/DL on this static-Cp task **and** SHAP makes their decisions interpretable in line with known wind physics (e.g., suction at upwind edges).
- **Comparison baselines**: Among the four tree models; no DL baseline.
- **Implementation feasibility on our BDH dataset**: ❌ **Not implementable as a forecasting model** — this is static surface-averaged Cp regression on geometric scalars, not a time-series model. SHAP itself is post-hoc and could be applied to *our* tree models as an *analysis add-on*, but that is an interpretability tool, not a new forecasting model.
- **Already covered by `Agent_Test/`?**: **YES (for the models)** — XGBoost and Random Forest are already in `Agent_Test/`. SHAP analysis on those is *not* applied.

---

### Paper 5: Tong, B., Liang, Y., Song, J., Hu, G. & Kareem, A. (2024). *Deep learning-based extension of wind pressure time series.* JWEIA 254, 105909. https://doi.org/10.1016/j.jweia.2024.105909  *(filename `Kareem_ML_JWEIA_2024.pdf`)*

- **Authors / Year / Journal**: B. Tong, Y. Liang, J. Song, G. Hu, A. Kareem / 2024 / JWEIA.
- **Main objective**: Extend short wind-pressure time series into long realistic non-Gaussian sequences using a tailored generative DL model (**WPTSE-Net**), avoiding the limitations of correlation-distortion static transform (CDST) and kernel density estimation.
- **Dataset used**: Wuhan University WD-1 wind tunnel; two rounded-corner 1.2 m × 0.3 m tall buildings (1:200, radii 15 mm and 45 mm); 3 measurement points per model. Total durations 200 min (Model I) and 500 min (Model II) at ~7 Hz → segmented into ten-minute slices of 4096 samples. **Only one 10-min slice** is used for training; the remaining slices are ground truth.
- **Inputs (features)**: Statistical features of an existing short reference slice — **mean, variance, skewness, kurtosis** — concatenated with **random noise** to form a (l × 5) latent matrix (l = number of slices).
- **Outputs (targets)**: New 10-min Cp time-series slices (4096 samples) that match the statistical distribution and spectrum of the reference.
- **Model architecture**: **WPTSE-Net** = handcrafted Encoder + Decoder.
  - **Encoder**: deterministic, *no learnable parameters* — packs (mean, var, skew, kurt, noise) into the latent vector (prior knowledge injection).
  - **Decoder**: 4 iterations of {Dense(128, ReLU) → Dense(128, ReLU) → BatchNorm → multiplicative scale layer → additive translation layer}, followed by a final linear FC that produces 4096-length output slices.
- **Loss / optimizer / LR / batch / epochs**: **Huber loss** (δ = 1), Adam optimiser, batch = 64, **1000 epochs**.
- **Preprocessing**: Segment long records into 10-min × 4096-sample slices; one slice is the training input, rest are held-out ground truth.
- **Evaluation metrics reported**: Comparison of generated vs. measured **PSD/spectra**, marginal PDFs, peak distributions, and time/frequency statistics; outperforms CDST and Gaussian KDE; ~5.5 s on RTX 3090 to generate 500 min on Model II.
- **Key results / claims**: WPTSE-Net captures non-Gaussian behaviour (skewness, kurtosis, peak distribution) far better than classical CDST and standard VAE/GAN, despite being trained on **a single 10-min slice**.
- **Comparison baselines**: Correlation Distortion Static Transform (CDST), Gaussian Kernel Density Estimation, plus reported failures of vanilla VAE and GAN.
- **Implementation feasibility on our BDH dataset**: ✅ **Fully implementable.** Our BDH face-averaged Cp arrays (32 768 samples at 1 kHz, per face × angle × α) match WPTSE-Net's data format almost exactly — 1-D Cp time series for which we want long, statistically faithful extensions. Decoder is a tiny MLP, no exotic blocks; trains in seconds on one GPU.
- **Already covered by `Agent_Test/`?**: **NO.** All current models in `Agent_Test/` are **predictive forecasters** (LSTM/GRU/TCN/PatchTST/Ridge/RF/XGB). None of them is a **generative time-series extender** designed for non-Gaussian Cp; this is a different task.

---

### Paper 6: Chen, Y., Kopp, G.A. & Surry, D. (2003). *Prediction of pressure coefficients on roofs of low buildings using artificial neural networks.* JWEIA 91, 423–441. https://doi.org/10.1016/S0167-6105(02)00381-1

- **Authors / Year / Journal**: Y. Chen, G.A. Kopp, D. Surry / 2003 / JWEIA.
- **Main objective**: Interpolate **mean and rms Cp** on gable-roof low-rise buildings — anywhere on the roof, for any wind direction — with a feed-forward ANN trained on the NIST WT database, in order to expand aerodynamic databases.
- **Dataset used**: UWO Boundary Layer Wind Tunnel II (NIST project); 1:100 gable-roof, plan 24.38 × 38.10 m, slope 1:12; **4 eave heights (4.88, 7.32, 9.75, 12.19 m)**, **37 wind directions (180°–360° step 5°)**, **2 terrains** (open country, suburban); 335 roof taps (120 in the corner bay); 500 Hz, 100 s record per case ≈ 50 000 samples.
- **Inputs (features)**: 4 scalars — α (wind direction), H (eave height), X/H, Y/H (normalised tap coordinates).
- **Outputs (targets)**: Mean Cp and rms Cp (one scalar at a time; two separate ANNs).
- **Model architecture**: Feed-forward MLP, **4 inputs → h1 → h2 → 1**, sigmoid hidden activations, linear output; exact h1/h2 found empirically.
- **Loss / optimizer / LR / batch / epochs**: **Levenberg-Marquardt back-prop**, MSE loss; early-stopping cross-validation; no batch / epoch numbers explicitly given.
- **Preprocessing**: Normalised inputs (X/H, Y/H), and cornering wind directions held out for testing.
- **Evaluation metrics reported**: Average error < 2 % on mean bay uplift for unseen cornering WDs; per-tap mean-square errors of **12 % (mean Cp)** and **9 % (rms Cp)** in the corner bay.
- **Key results / claims**: A small back-prop MLP captures the highly non-linear corner-bay Cp surface from very few inputs and generalises to unseen WDs.
- **Comparison baselines**: Linear interpolation; regression polynomials.
- **Implementation feasibility on our BDH dataset**: ⚠️ **Partial / needs adaptation.** Same caveat as Paper 1: this is a *static* tap-coordinate-to-Cp regressor, not a time-series forecaster. Could be implemented as `(building_ratio, angle, α, face) → (mean_Cp, std_Cp)` using the per-angle statistics CSVs, but our BDH data has only 4 face-averaged channels, so we cannot reproduce tap-level (X/H, Y/H) inputs.
- **Already covered by `Agent_Test/`?**: **NO direct equivalent** — closest equivalent is the same MLP idea as Paper 1 (also missing).

---

### Paper 7: Bre, F., Gimenez, J.M. & Fachinotti, V.D. (2018). *Prediction of wind pressure coefficients on building surfaces using artificial neural networks.* Energy and Buildings 158, 1429–1441. https://doi.org/10.1016/j.enbuild.2017.11.045

- **Authors / Year / Journal**: F. Bre, J.M. Gimenez, V.D. Fachinotti / 2018 / Energy and Buildings.
- **Main objective**: Predict **surface-averaged mean Cp** on every face (walls + roof) of low-rise buildings with flat / gable / hip roofs as a fast surrogate for use in building-performance and airflow-network programs.
- **Dataset used**: TPU isolated low-rise buildings database; flat, gable, hip roofs; multiple side ratios D/B, height ratios H/B, roof pitches β, wind attack angles 0°–180° step 15°.
- **Inputs (features)**: Per-roof-type ANN — wind attack angle θ, depth-to-breadth ratio D/B, height-to-breadth H/B, and roof pitch β (for gable/hip).
- **Outputs (targets)**: Vector of **surface-averaged mean Cp** values, one per face (one ANN per roof type; 5 outputs for flat roof, more for gable/hip).
- **Model architecture**: Feed-forward multilayer ANN per roof type — `i – h1 – h2 – … – hn – o`. Architecture chosen by trial-and-error coarse calibration (number of layers) then progressive widening (neurons). Tangent-sigmoid hidden, linear output.
- **Loss / optimizer / LR / batch / epochs**: **Levenberg-Marquardt** back-prop, MSE loss, max **500 epochs**; MATLAB / R / TensorFlow implementations described.
- **Preprocessing**: Min-max scaling; held-out wind angles for testing.
- **Evaluation metrics reported**: R² and absolute error vs. Swami-Chandra and Muehleisen-Patrizi parametric equations — ANN clearly more accurate than both classical equations across all roof types.
- **Key results / claims**: A small ANN replaces parametric Cp equations across a wide envelope of low-rise geometries with materially better accuracy and remains cheap enough to live inside EnergyPlus-style airflow-network tools.
- **Comparison baselines**: Swami-Chandra equation; Muehleisen-Patrizi equation.
- **Implementation feasibility on our BDH dataset**: ⚠️ **Partial / needs adaptation.** Same family as Papers 1 and 6 — a static MLP from geometry-and-angle to Cp statistics. Implementable on the per-angle statistics CSVs in `summary_all_buildings.csv` + `statistics_angle_*.csv`, but reformulates the task away from time-series forecasting.
- **Already covered by `Agent_Test/`?**: **NO direct equivalent** — same MLP family as Paper 1 / Paper 6 (also missing).

---

## Summary Table

| # | Short name | Model family | Feasibility on BDH | Already in `Agent_Test/`? | Recommended for Phase 2? |
|---|------------|--------------|--------------------|---------------------------|--------------------------|
| 1 | Aldoum 2025 — Non-rect roofs GBR + ANN | Static MLP + GBR regressor on (geometry, angle) | ⚠️ Partial | No (no static MLP) | **N** — different task (static, not time-series); covered conceptually by RF/XGB already |
| 2 | Nav-Snaiki 2025 — Sparse-sensor hybrid | POD/AE + neural mapper for spatial reconstruction | ❌ Not implementable | No | **N** — needs per-tap field data we don't have |
| 3 | Hu 2020 — Interference cGAN | Two-stream conditional GAN over pressure maps | ⚠️ Partial (tree branch trivial, GAN branch needs tap maps) | RF/XGB yes; cGAN no | **N** — GAN branch requires per-tap maps absent in BDH post-processed data |
| 4 | Meddage 2022 — SHAP on tree models | DT / XGBoost / Extra-Trees / LightGBM + SHAP XAI | ❌ as a *new* forecaster (it's XAI) | XGBoost/RF yes; SHAP analysis no | **N as a model**; *optional* SHAP add-on for our existing XGB/RF/Ridge (small effort, large interpretability win) |
| 5 | Tong-Kareem 2024 — **WPTSE-Net** | Generative time-series extender (small Decoder MLP, prior-knowledge encoder) | ✅ Fully implementable | **No** | **Y — strongly recommended.** Different task (sequence *generation/extension*) from our forecasters; data format (1-D Cp at single point/face) is exactly the BDH face-averaged format; tiny model, fast train |
| 6 | Chen-Kopp-Surry 2003 — LM-MLP for roof Cp | Static MLP `(WD, H, X/H, Y/H) → (mean, rms)` | ⚠️ Partial | No | **N** — same static-regression family as Papers 1/7, and the (X/H, Y/H) tap-coordinate inputs are absent in our face-averaged data |
| 7 | Bre 2018 — ANN per-roof-type | Static MLP `(θ, D/B, H/B, β) → surface-averaged Cp vector` | ⚠️ Partial | No | **Maybe (low priority).** Implementable on `summary_all_buildings.csv` if we want to expose a *static* surrogate alongside the time-series forecasters; not the core thesis task |

---

## Final tally

- **Papers classified as "Fully implementable AND not already covered" in `Agent_Test/`: 1**
  - Paper 5 — Tong, Liang, Song, Hu, Kareem (2024) WPTSE-Net.
- **Recommended for Phase 2 implementation**:
  1. **Paper 5 — WPTSE-Net (Tong & Kareem 2024).** Closest fit to our face-averaged 1-D Cp time series; tiny generative model with a deterministic statistical-prior encoder; complements (does not duplicate) our existing forecasters by adding a *time-series extension/synthesis* capability.
- **Optional add-on (no new model required)**:
  - **Paper 4 — SHAP interpretability** on the already-trained Random Forest / XGBoost / Ridge models, to add interpretability evidence to the thesis without writing a new architecture.
- **Not recommended for Phase 2**: Papers 1, 2, 3, 6, 7 — either incompatible with our face-averaged data (no per-tap pressure maps), or duplicate the families we already have, or are review/XAI-focused.

# Wind Pressure Cp — Paper Summaries

## 1. Hybrid Machine Learning Framework for Wind Pressure Prediction (MICE 2025)

**File:** `A hybrid machine learning framework for wind pressure prediction on buildings with constrained sensor networks.pdf`
**DOI:** 10.1111/mice.13488

**Key idea:** Reconstruct full-field wind pressure on a building façade using a small subset of sensors (constrained network). A CNN extracts spatial-temporal features from sparse sensor time series; an LSTM models temporal dependencies to predict pressure at unmeasured locations.

### Architecture (CNN-LSTM Hybrid)

| Stage | Layer | Parameters |
|---|---|---|
| CNN Block 1 | Conv1D(in=n_features, out=64, kernel=3, padding=1) + BatchNorm + ReLU | — |
| CNN Block 2 | Conv1D(64→128, kernel=3, padding=1) + BatchNorm + ReLU + MaxPool1D(2) | — |
| CNN Block 3 | Conv1D(128→256, kernel=3, padding=1) + BatchNorm + ReLU | — |
| LSTM | 2 layers, hidden=256, dropout=0.2 | — |
| FC Head | Linear(256→128) + ReLU + Dropout(0.3) + Linear(128→1) | — |

- **Input:** (batch, seq_len, n_sensors) — time series from available sensors
- **Output:** (batch, 1) single-step Cp prediction (extended to multi-step autoregressively)
- **Loss:** MSE
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-4)
- **Training strategy:** Early stopping (patience=15), ReduceLROnPlateau (factor=0.5, patience=5)
- **Normalization:** Z-score (zero mean, unit variance) per face
- **Evaluation:** RMSE, MAE, R², MAPE

---

## 2. Deep Learning-Based Investigation of Wind Pressures on Tall Buildings (Hu et al.)

**Key idea:** Use GANs, Decision Tree, Random Forest, XGBoost to predict Cp under interference effects. Train on 30% of wind-tunnel cases; predict remaining 70%.

### Models
- Decision Tree, Random Forest, XGBoost (tabular): predict mean Cp from building geometry + wind angle
- **GAN** (best model): Generator learns distribution of Cp maps; discriminator trains adversarially
- Outperforms ML baselines in capturing full Cp spatial distribution

---

## 3. Kareem ML/JWEIA 2024 — Review of ML for Wind Engineering

**Key architectures reviewed:**
- DNN / BPNN: input features → FC layers → Cp output
- **LSTM:** Sequence-to-sequence Cp forecasting; captures long-range temporal dependencies
- **ConvLSTM:** Spatial-temporal; 2D convolutional cells useful for pressure field maps
- **TCN (Temporal Convolutional Network):** Causal dilated convolutions; strong for long sequences
- **Transformer:** Self-attention; best multi-horizon forecasting
- GNN: Graph-based; models spatial connectivity of pressure-tap networks

**Key finding:** CNN-LSTM and TCN outperform pure LSTM for wind pressure time series due to better local-feature extraction.

---

## 4. Aldoum & Stathopoulos 2025 (WAS)

**Key idea:** ANN (fully connected) + Gradient Boosting Regressor for roof Cp on non-rectangular buildings.

### ANN Architecture
- Input: 9 features (geometry, wind angle, zone)
- Hidden layers: 4 layers — [100, 500, 500, 100] neurons
- Activation: sigmoid / tanh
- Output: 1 (mean Cp or peak Cp)
- Split: random vs. structured (structured = by wind direction zone → more realistic)
- R² ≈ 0.97, MSE minimal

---

## 5. Prediction of Wind Pressure Coefficients on Building Surfaces (ANN)

**Key idea:** Feedforward ANN to predict surface Cp from geometric and flow parameters.

### ANN Architecture
- Input: [B/D, H/D, wind angle, face index, tap position]
- Hidden: 2–4 layers, 20–100 neurons each, tanh activation
- Output: 1 (Cp value)
- Training: Levenberg–Marquardt or BFGS; normalized inputs to [-1, 1]

---

## 6. Prediction of Pressure Coefficients on Roofs of Low Buildings (ANN)

Similar ANN approach, applied to low-rise buildings' roofs. Input features include building geometry ratios, wind angle, roof pitch, and tap location. 2–3 hidden layers, 20–50 neurons.

---

## 7. Interpretation of ML-Based (Black-box) Wind Pressure Models

**Key idea:** Apply SHAP (SHapley Additive exPlanations) and LIME to interpret predictions of black-box ML models (XGBoost, RF, ANN) for wind pressure coefficients. Finds wind angle and H/D ratio as most influential features.

---

## Summary Table — Model Comparison

| Paper | Model | Task | Key Metric | Implemented |
|---|---|---|---|---|
| Hybrid ML (MICE 2025) | **CNN-LSTM** | Time-series forecasting + reconstruction | RMSE, MAE, R² | ✅ `cnn_lstm/` |
| Kareem JWEIA 2024 | **LSTM** | Cp time-series forecasting | RMSE | ✅ `lstm/` |
| Kareem JWEIA 2024 | **TCN** | Cp time-series forecasting | RMSE | ✅ `tcn/` |
| Kareem JWEIA 2024 | **Transformer** | Multi-horizon Cp forecasting | RMSE | ✅ `transformer/` |
| Aldoum 2025 | **ANN (Feedforward)** | Mean/peak Cp from geometry | R² ≈ 0.97 | ✅ `ann/` |
| Hu et al. | GAN, XGBoost | Cp field prediction under interference | R² | — |
| ANN Building Surfaces | Feedforward ANN | Surface Cp from parameters | MSE | — |
| ANN Low-rise Roofs | Feedforward ANN | Roof Cp | MSE | — |
| Interpretation paper | XGBoost + SHAP | Explainability | Feature importance | — |

## Implemented Models — Architecture Details

### CNN-LSTM (`Agent_Test/cnn_lstm/`)
- **Source:** Hybrid ML (MICE 2025)
- **Architecture:** Conv1D(4→64→128→256) + BatchNorm + ReLU + MaxPool → LSTM(256, 2 layers) → FC(256→128→1)
- **Hyperparams:** lr=1e-3, weight_decay=1e-4, dropout=0.3, early stopping patience=15

### LSTM (`Agent_Test/lstm/`)
- **Source:** Kareem JWEIA 2024
- **Architecture:** LSTM(4→256, 2 layers, dropout=0.2) → FC(256→128→1)
- **Hyperparams:** lr=1e-3, weight_decay=1e-4, fc_dropout=0.3

### TCN (`Agent_Test/tcn/`)
- **Source:** Kareem JWEIA 2024
- **Architecture:** 3× TemporalBlock (causal dilated conv + residual) [64, 128, 256 channels] → FC(256→128→1)
- **Hyperparams:** kernel_size=3, dilations=[1,2,4], dropout=0.2, lr=1e-3

### Transformer (`Agent_Test/transformer/`)
- **Source:** Kareem JWEIA 2024
- **Architecture:** Linear(4→128) + PosEncoding → TransformerEncoder(3 layers, 8 heads, d_ff=256) → FC(128→64→1)
- **Hyperparams:** dropout=0.1, lr=1e-4 (lower for stability)

### ANN (`Agent_Test/ann/`)
- **Source:** Aldoum & Stathopoulos 2025
- **Architecture:** Flatten(100×4) → FC[100, 500, 500, 100] with ReLU + Dropout(0.3) → Linear(100→1)
- **Hyperparams:** lr=1e-3, weight_decay=1e-4

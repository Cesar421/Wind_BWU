# Wind Pressure Cp Forecasting — BWU Research

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-live-blue?logo=github)](https://cesar421.github.io/Wind_BWU/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://bwucesarcpforecasting.streamlit.app/)

Deep-learning and classical forecasting of wind pressure coefficients ($C_p$) on tall buildings using the [TPU BDH](https://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/highrise/Homepage/homepageHDF.htm) benchmark.

→ **[Project Website](https://cesar421.github.io/Wind_BWU/)** · **[Interactive Dashboard](https://bwucesarcpforecasting.streamlit.app/)**

---

## Headline Results

| Model | Paradigm | RMSE h=500 | R² h=500 |
|-------|----------|:----------:|:--------:|
| Naive persistence | — | 0.118 | ≈ 0.97 |
| XGBoost | Autoregressive | 0.115 | — (ties naive, p=0.20) |
| LSTM autoregressive | Autoregressive | 0.307 | **−6.18** ← collapse |
| **LSTM-direct** | Direct multi-step | **0.179** | **0.84** |
| **PatchTST** | Direct multi-step | **0.175** | **0.85** |

**Key finding:** direct multi-step training (single forward pass over 500 future steps) fixes autoregressive collapse, but the predicted PSD remains 60–75× below the true signal PSD at all frequencies — MSE training collapses to the conditional mean.

## Repo Structure

```
Agent_Test/      ← all models, training scripts, results
Agent_Papers/    ← WPTSE-Net synthesis model
AI_Agent/        ← Streamlit dashboard + multi-agent system
Data/            ← TPU BDH raw data (gitignored)
docs/            ← GitHub Pages website
```

## Reproduce

```powershell
conda activate ML_Cesar
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
cd Agent_Test
python train_all.py --scope all          # Round 3 (full)
python analyze_horizons.py --scope all   # multi-horizon metrics + plots
```

## Dashboard (local)

```powershell
pip install -r requirements.txt
streamlit run AI_Agent/streamlit_app.py
```

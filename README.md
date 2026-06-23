# Wind Pressure Cp Forecasting — BWU Research

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-live-blue?logo=github)](https://cesar421.github.io/Wind_BWU/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://bwucesarcpforecasting.streamlit.app/)

Deep-learning and classical forecasting of wind pressure coefficients ($C_p$) on tall buildings using the [TPU BDH](https://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/highrise/Homepage/homepageHDF.htm) benchmark.

→ **[Project Website](https://cesar421.github.io/Wind_BWU/)** · **[Interactive Dashboard](https://bwucesarcpforecasting.streamlit.app/)**

---

## Headline Results

Long-horizon (h=500) numbers below are **per-step error aggregated over the full
146,880-window test set** — the same regime for every model. (An earlier version
scored the autoregressive models on a *single* 500-step trajectory, which
inflated the naive baseline to RMSE 0.118 / R² 0.97; that was a measurement
artifact — see [`evaluate_long_horizon.py`](Agent_Test/evaluate_long_horizon.py).)

| Model | Paradigm | RMSE h=500 | R² h=500 | vs Naive |
|-------|----------|:----------:|:--------:|:--------:|
| **PatchTST** | Direct multi-step | **0.175** | **0.85** | **beats** |
| **LSTM-direct** | Direct multi-step | **0.179** | **0.84** | **beats** |
| Naive persistence | — | 0.231 | 0.74 | baseline |
| LSTM autoregressive | Autoregressive | 0.270 | 0.65 | slightly worse |
| GRU autoregressive | Autoregressive | 0.273 | 0.64 | slightly worse |
| TCN autoregressive | Autoregressive | 0.411 | 0.19 | collapses |

**Key findings:**
1. **Direct multi-step training beats naive persistence at h=500** (RMSE 0.175–0.179 vs 0.231); autoregressive rollout does not.
2. The earlier "catastrophic collapse" of autoregressive deep models (reported R² = −6.18) was largely a **single-trajectory artifact**. Over 680 trajectories, LSTM/GRU reach R² ≈ 0.65 and beat naive on ~45 % of trajectories; only **TCN** is genuinely unstable.
3. Despite winning on RMSE/R², the predicted PSD of the direct models stays **60–75× below the true-signal PSD at all frequencies** — MSE training collapses to the conditional mean, so the forecasts remain unusable for spectrum-dependent wind-engineering quantities (fatigue, peak factor).

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

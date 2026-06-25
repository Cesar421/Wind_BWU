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
| **XGBoost** | Autoregressive | **0.209** | **0.79** | **beats** (DM +12.7) |
| Naive persistence | — | 0.231 | 0.74 | baseline |
| Ridge | Autoregressive | 0.256 | 0.69 | slightly worse (stable) |
| LSTM autoregressive | Autoregressive | 0.270 | 0.65 | slightly worse |
| GRU autoregressive | Autoregressive | 0.273 | 0.64 | slightly worse |
| TCN autoregressive | Autoregressive | 0.411 | 0.19 | collapses |

(Classical rows re-fit at stride 30; Random Forest multi-trajectory re-fit still pending. Naive is the exact 146,880-window value; the other rows use 680 trajectories.)

**Key findings:**
1. **Three models beat naive persistence at h=500** — both direct multi-step models (PatchTST 0.175, LSTM-direct 0.179) and, more surprisingly, **autoregressive XGBoost** (0.209, beats naive on 53 % of trajectories, DM +12.7).
2. The earlier "catastrophic collapse" of autoregressive deep models (reported R² = −6.18) and "Ridge diverges to ~10⁴" were both **single-trajectory artifacts**. Over 680 trajectories, LSTM/GRU reach R² ≈ 0.65 (beating naive on ~45 % of trajectories), Ridge is stable (0.256), and only **TCN** is genuinely unstable.
3. Despite winning on RMSE/R², the predicted PSD of the direct models stays **60–75× below the true-signal PSD at all frequencies** — MSE training collapses to the conditional mean, so the forecasts remain unusable for spectrum-dependent wind-engineering quantities (fatigue, peak factor).

## Repo Structure

```
Agent_Test/      ← all models, training scripts, results
Agent_Papers/    ← WPTSE-Net synthesis model
AI_Agent/        ← Streamlit dashboard + multi-agent system
Data/            ← TPU BDH raw data (gitignored)
docs/            ← GitHub Pages website
```

📚 **All docs in one place:** see [DOCS.md](DOCS.md) (documentation index).

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

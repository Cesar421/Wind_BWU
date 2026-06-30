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
| Random Forest | Autoregressive | 0.212 | 0.78 | beats (marginal) |
| XGBoost | Autoregressive | 0.217 | 0.77 | tie |
| Naive persistence | — | 0.218 | 0.77 | baseline |
| LSTM autoregressive | Autoregressive | 0.270 | 0.65 | slightly worse |
| GRU autoregressive | Autoregressive | 0.273 | 0.64 | slightly worse |
| Ridge | Autoregressive | 0.281 | 0.62 | slightly worse (stable) |
| TCN autoregressive | Autoregressive | 0.411 | 0.19 | collapses |

(Classical rows re-fit at window-stride 40; naive = 680-trajectory value. The exact 146,880-window naive is 0.231, which the two direct models also beat. The exact tree-vs-naive margin is sensitive to training-set size.)

**Key findings:**
1. **The direct multi-step models clearly beat naive persistence at h=500** (PatchTST 0.175, LSTM-direct 0.179 vs 0.218–0.231) — by changing only the training objective, not the architecture.
2. **The tree-based autoregressive models sit right at the naive level** — Random Forest (0.212) marginally beats it, XGBoost (0.217) ties. The exact margin moves with training-set size, which is itself the finding: at long horizon these models hit the **predictability ceiling** around persistence, they do not clear it.
3. The earlier "catastrophic collapse" of autoregressive deep models (reported R² = −6.18) and "Ridge diverges to ~10⁴" were both **single-trajectory artifacts**. Over 680 trajectories LSTM/GRU reach R² ≈ 0.65, Ridge is stable (0.281), and only **TCN** is genuinely unstable.
4. Despite winning on RMSE/R², the predicted PSD of the direct models stays **60–75× below the true-signal PSD at all frequencies** — MSE training collapses to the conditional mean, so the forecasts remain unusable for spectrum-dependent wind-engineering quantities (fatigue, peak factor).

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

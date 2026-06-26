"""
Wind Pressure Cp - Forecasting Dashboard
=========================================
Visualizes all models trained in Agent_Test without retraining.
Also includes the AI Agent tab (AI_Agent).

Run:
    streamlit run AI_Agent/streamlit_app.py
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent          # AI_Agent/
PROJ        = ROOT.parent                    # Wind_BWU/
AT          = PROJ / "Agent_Test"            # Agent_Test/
AT_RESULTS  = AT / "results"
LOGS_DIR    = ROOT / "logs"
LIT_DIR     = ROOT / "literature_results"

# All Agent_Test models: (folder, display_name)
MODELS = [
    ("naive",         "Naive Persistence"),
    ("ridge",         "Ridge"),
    ("random_forest", "Random Forest"),
    ("xgboost",       "XGBoost"),
    ("lstm",          "LSTM (autoreg)"),
    ("gru",           "GRU (autoreg)"),
    ("tcn",           "TCN (autoreg)"),
]
HORIZONS = [1, 10, 50, 100, 500]
ROUNDS   = [1, 2, 3]

MODEL_COLORS = {
    "naive_persistence": "#aaaaaa",
    "naive":             "#aaaaaa",
    "ridge":             "#4fc3f7",
    "random_forest":     "#81c784",
    "xgboost":           "#ffb74d",
    "lstm":              "#ba68c8",
    "gru":               "#f06292",
    "tcn":               "#4dd0e1",
    "lstm_direct_h10":   "#ff8a65",
    "lstm_direct_h500":  "#ff5722",
    "patchtst_h10":      "#d4e157",
    "patchtst_h500":     "#cddc39",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Wind Cp - Model Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
div[data-testid="stMetric"]{background:#1a1d2e;border-radius:8px;padding:12px;margin:2px;}
</style>
""", unsafe_allow_html=True)

for k, v in {"running": False, "log_text": ""}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_csv(path):
    try:
        p = Path(path)
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return None


def load_npy(path):
    try:
        p = Path(path)
        if p.exists():
            return np.load(str(p))
    except Exception:
        pass
    return None


def model_has_checkpoint(folder):
    d = AT / folder / "checkpoints"
    return d.exists() and any(d.glob("*.pt"))


def forecast_npy(folder, horizon):
    p = AT / folder / "results" / "forecasts" / f"{folder}_h{horizon}.npy"
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Wind Cp Dashboard")
    st.caption("Agent_Test - All models")
    st.divider()

    model_options         = {name: folder for folder, name in MODELS}
    selected_model_name   = st.selectbox("Model (Per Model tab)", list(model_options.keys()))
    selected_model_folder = model_options[selected_model_name]

    st.divider()
    selected_round = st.selectbox(
        "Round", ROUNDS, index=2, format_func=lambda r: f"Round {r}"
    )
    st.divider()
    st.caption(f"Project: `{PROJ.name}`")
    st.caption(f"Python: `{sys.version.split()[0]}`")

# ---------------------------------------------------------------------------
# Title + Tabs
# ---------------------------------------------------------------------------
st.title("Wind Pressure Cp - Forecasting Dashboard")
st.caption("All trained models - TPU BDH Benchmark - No retraining")

(tab_overview, tab_horizon, tab_longh, tab_live, tab_model,
 tab_direct, tab_dm, tab_cross, tab_plots, tab_agent) = st.tabs([
    "Summary",
    "Multi-Horizon",
    "Long-horizon (corrected)",
    "Live Forecast",
    "Per Model",
    "LSTM-direct / PatchTST",
    "Diebold-Mariano",
    "Cross-Round",
    "Plots",
    "AI Agent",
])

# ===========================================================================
# TAB 1 — RESUMEN
# ===========================================================================
with tab_overview:
    st.subheader(f"Model Comparison - Round {selected_round}")

    df_comp = load_csv(AT_RESULTS / f"model_comparison_round{selected_round}.csv")
    if df_comp is None:
        st.warning("Comparison CSV not found for this round.")
    else:
        # Metric cards
        cols = st.columns(len(df_comp))
        for col, (_, row) in zip(cols, df_comp.iterrows()):
            with col:
                st.metric(
                    label=row["model"],
                    value=f"R2={row['r2']:.4f}",
                    delta=f"RMSE={row['rmse']:.5f}",
                )

        st.divider()
        st.dataframe(
            df_comp.style
                   .background_gradient(subset=["r2"],   cmap="RdYlGn")
                   .background_gradient(subset=["rmse"], cmap="RdYlGn_r"),
            use_container_width=True,
        )

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                df_comp.sort_values("r2", ascending=True),
                x="r2", y="model", orientation="h",
                title=f"R2 (h=1) - Round {selected_round}",
                color="model", color_discrete_map=MODEL_COLORS,
                template="plotly_dark",
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(
                df_comp.sort_values("rmse"),
                x="model", y="rmse",
                title=f"RMSE (h=1) - Round {selected_round}",
                color="model", color_discrete_map=MODEL_COLORS,
                template="plotly_dark",
            )
            fig2.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig2, use_container_width=True)

        if {"train_time_s", "parameters"}.issubset(df_comp.columns):
            fig3 = px.scatter(
                df_comp, x="parameters", y="train_time_s",
                text="model", color="model",
                color_discrete_map=MODEL_COLORS,
                title="Parameters vs Training Time",
                log_x=True, log_y=True,
                template="plotly_dark",
            )
            fig3.update_traces(textposition="top center")
            fig3.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig3, use_container_width=True)


# ===========================================================================
# TAB 2 — MULTI-HORIZONTE
# ===========================================================================
with tab_horizon:
    st.subheader(f"Multi-Horizon Metrics - Round {selected_round}")
    st.caption(
        "⚠️ Legacy single-trajectory evaluation (one 500-step rollout from the "
        "last test series). The h=500 numbers here are not representative — see "
        "the **Long-horizon (corrected)** tab for the multi-trajectory results."
    )

    df_mh = load_csv(AT_RESULTS / f"multi_horizon_metrics_round{selected_round}.csv")
    if df_mh is None:
        st.warning("Multi-horizon CSV not found for this round.")
    else:
        # Append direct models
        extras = []
        for f in [AT_RESULTS / "lstm_direct_metrics.csv",
                  AT_RESULTS / "patchtst_metrics.csv"]:
            d = load_csv(f)
            if d is not None:
                extras.append(d[["model", "horizon", "rmse", "mae", "r2"]])
        if extras:
            df_mh = pd.concat([df_mh] + extras, ignore_index=True)

        metric = st.radio("Metric", ["r2", "rmse", "mae"], horizontal=True, key="mh_m")
        fig = px.line(
            df_mh, x="horizon", y=metric, color="model",
            markers=True,
            title=f"{metric.upper()} vs Horizon - Round {selected_round}",
            color_discrete_map=MODEL_COLORS,
            template="plotly_dark",
            log_x=True,
        )
        if metric == "r2":
            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
        fig.update_layout(height=500, xaxis_title="Horizon (ms)")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader(f"Pivot Table - {metric.upper()}")
        try:
            pivot = df_mh.pivot_table(index="model", columns="horizon", values=metric)
            pivot.columns = [f"h={c}" for c in pivot.columns]
            cmap = "RdYlGn" if metric == "r2" else "RdYlGn_r"
            st.dataframe(
                pivot.style.background_gradient(cmap=cmap, axis=None),
                use_container_width=True,
            )
        except Exception:
            st.dataframe(df_mh, use_container_width=True)


# ===========================================================================
# TAB — LONG-HORIZON (CORRECTED, MULTI-TRAJECTORY)
# ===========================================================================
with tab_longh:
    st.subheader("Long-horizon h=500 — multi-trajectory (corrected)")
    st.caption(
        "Per-step error at h=500 aggregated over the full test set / 680 "
        "trajectories — the same regime for every model. Fixes the earlier "
        "single-trajectory artifact that inflated naive (RMSE 0.118 / R² 0.97) "
        "and the deep-model collapse (R² −6.18)."
    )

    df_lh = load_csv(AT_RESULTS / "long_horizon_multitraj_round3.csv")
    df_nd = load_csv(AT_RESULTS / "naive_dense_metrics_round3.csv")
    df_sig = load_csv(AT_RESULTS / "long_horizon_significance_round3.csv")

    if df_lh is None:
        st.warning(
            "long_horizon_multitraj_round3.csv not found. Run "
            "`python evaluate_long_horizon.py` and `python naive_dense_baseline.py`."
        )
    else:
        # Headline corrected comparison at h=500
        rows = []
        h500 = df_lh[df_lh.horizon == 500]
        for _, r in h500.iterrows():
            rows.append({"model": r["model"], "RMSE h=500": round(r["rmse"], 3),
                         "R² h=500": round(r["r2"], 3)})
        # dense naive (exact, full window set) + direct models
        if df_nd is not None:
            nd = df_nd[df_nd.horizon == 500].iloc[0]
            rows.append({"model": "naive_dense (146k win)",
                         "RMSE h=500": round(nd["rmse"], 3),
                         "R² h=500": round(nd["r2"], 3)})
        for f, nm in [("lstm_direct_metrics.csv", "LSTM-direct"),
                      ("patchtst_metrics.csv", "PatchTST")]:
            d = load_csv(AT_RESULTS / f)
            if d is not None:
                dd = d[(d.horizon == 500)]
                if len(dd):
                    rr = dd.iloc[-1]
                    rows.append({"model": nm, "RMSE h=500": round(rr["rmse"], 3),
                                 "R² h=500": round(rr["r2"], 3)})
        comp = pd.DataFrame(rows).sort_values("RMSE h=500").reset_index(drop=True)
        st.dataframe(comp.style.background_gradient(cmap="RdYlGn_r",
                     subset=["RMSE h=500"]), use_container_width=True)
        st.markdown(
            "**Direct multi-step (LSTM-direct 0.179, PatchTST 0.175) beats naive "
            "(0.231).** Autoregressive LSTM/GRU (~0.27) are competitive — they "
            "beat naive on ~45 % of individual trajectories; only TCN (0.41) is "
            "genuinely unstable."
        )

        st.divider()
        metric_lh = st.radio("Metric vs horizon", ["rmse", "r2"],
                             horizontal=True, key="lh_m")
        fig_lh = px.line(df_lh, x="horizon", y=metric_lh, color="model",
                         markers=True, log_x=True, template="plotly_dark",
                         color_discrete_map=MODEL_COLORS,
                         title=f"{metric_lh.upper()} vs horizon (multi-trajectory, R3)")
        fig_lh.update_layout(height=460, xaxis_title="Horizon (ms)")
        st.plotly_chart(fig_lh, use_container_width=True)

        if df_sig is not None:
            st.divider()
            st.subheader("Significance vs naive (680 trajectories)")
            st.caption("win_rate_vs_naive = fraction of trajectories where the "
                       "model beats naive; wilcoxon_p = paired test across "
                       "trajectories.")
            st.dataframe(df_sig, use_container_width=True)

        for p, cap in [("rmse_vs_horizon_multitraj.png",
                        "RMSE vs horizon (mean over trajectories)"),
                       ("rmse_h500_boxplot_multitraj.png",
                        "Per-trajectory RMSE distribution over 500 steps")]:
            fp = AT_RESULTS / "plots_cross_round" / p
            if fp.exists():
                st.image(str(fp), caption=cap, use_container_width=True)


# ===========================================================================
# TAB — LIVE FORECAST (direct multi-step model, loaded from checkpoint, CPU)
# ===========================================================================
with tab_live:
    st.subheader("Live forecast — direct multi-step model (h = 500)")
    st.caption(
        "Pick a TPU BDH test segment as the 100-sample seed; the model returns "
        "the next 500 steps in a single forward pass. Expected behaviour: the "
        "forecast tracks the slow envelope but is smooth — at 0.5 s lead the "
        "MSE-trained model outputs the conditional mean (see Spectral Findings)."
    )

    import importlib.util
    if str(AT) not in sys.path:
        sys.path.insert(0, str(AT))

    DIRECT_SPECS = {
        "LSTM-direct": (AT / "lstm" / "models" / "lstm.py", "PureLSTM",
                        dict(n_features=4, hidden_size=256, num_layers=2,
                             dropout=0.2, fc_hidden=128, fc_dropout=0.3, horizon=500),
                        AT / "lstm" / "checkpoints" / "lstm_direct_h500.pt"),
        "PatchTST":    (AT / "patchtst" / "models" / "patchtst.py", "PatchTST",
                        dict(n_features=4, seq_len=100, horizon=500),
                        AT / "patchtst" / "checkpoints" / "patchtst_h500_r3.pt"),
    }

    @st.cache_resource(show_spinner="Loading model checkpoint …")
    def _load_direct(choice):
        path, cls_name, kwargs, ckpt = DIRECT_SPECS[choice]
        if not ckpt.exists():
            return None, f"Checkpoint missing: {ckpt}"
        try:
            import torch
            spec = importlib.util.spec_from_file_location(f"live_{choice}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            model = getattr(mod, cls_name)(**kwargs)
            state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            model.load_state_dict(state.get("state_dict", state))
            model.eval()
            return model, None
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"

    @st.cache_data(show_spinner="Building test seeds (first time ~20 s) …")
    def _seeds():
        from data_loader import build_test_trajectories
        tr = build_test_trajectories(seq_length=100, H=500, traj_per_series=1)
        return (tr["seeds"], tr["futures"],
                float(tr["mu"][0]), float(tr["sigma"][0]), tr["meta"])

    choice = st.selectbox("Model", list(DIRECT_SPECS.keys()), key="live_model")
    model, err = _load_direct(choice)
    if model is None:
        st.error(
            f"{err}\n\nThe direct checkpoints are gitignored. Copy "
            "`lstm_direct_h500.pt` / `patchtst_h500_r3.pt` into "
            "`Agent_Test/<model>/checkpoints/` (or retrain on GPU)."
        )
    else:
        seeds, futures, mu_w, sigma_w, meta = _seeds()
        n = seeds.shape[0]
        idx = st.slider("Test segment (building-angle trajectory)", 0, n - 1, 0,
                        key="live_idx")
        a, r, ang, start = meta[idx]
        st.write(f"**Seed:** {a} / ratio {r} / yaw {ang}° — start sample {start}")

        import torch
        with torch.no_grad():
            x = torch.from_numpy(seeds[idx][None]).float()      # (1, 100, 4)
            out = model(x).cpu().numpy().reshape(-1)[:500]      # (500,)
        pred = out * sigma_w + mu_w
        true = futures[idx] * sigma_w + mu_w
        seed_w = seeds[idx][:, 0] * sigma_w + mu_w               # past windward

        rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
        ss = float(np.sum((true - pred) ** 2))
        st_tot = float(np.sum((true - true.mean()) ** 2))
        r2 = 1 - ss / st_tot if st_tot > 0 else float("nan")
        c1, c2 = st.columns(2)
        c1.metric("RMSE (h=500 trajectory)", f"{rmse:.4f}")
        c2.metric("R²", f"{r2:.3f}")

        past_x = np.arange(-100, 0)
        fut_x = np.arange(0, 500)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=past_x, y=seed_w, name="Seed (past 100)",
                                 line=dict(color="#888")))
        fig.add_trace(go.Scatter(x=fut_x, y=true, name="True future",
                                 line=dict(color="royalblue")))
        fig.add_trace(go.Scatter(x=fut_x, y=pred, name=f"{choice} forecast",
                                 line=dict(color="tomato", dash="dash")))
        fig.add_vline(x=0, line_dash="dot", line_color="white", opacity=0.4)
        fig.update_layout(template="plotly_dark", height=460,
                          xaxis_title="Step relative to forecast origin",
                          yaxis_title="Windward Cp",
                          title=f"{choice} — 500-step forecast from a real seed")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Smooth forecast = the spectral-collapse phenomenon: high R² / "
            "low RMSE, but the turbulent fluctuations are not reproduced. Good "
            "for the mean trend, not for fatigue / peak-factor use."
        )


# ===========================================================================
# TAB 3 — POR MODELO
# ===========================================================================
with tab_model:
    st.subheader(f"{selected_model_name}")

    folder    = selected_model_folder
    model_dir = AT / folder

    c_info, c_ckpt = st.columns(2)
    with c_info:
        st.write(f"**Directory:** `Agent_Test/{folder}/`")
        if model_has_checkpoint(folder):
            for c in (model_dir / "checkpoints").glob("*.pt"):
                st.success(f"Checkpoint: `{c.name}` ({c.stat().st_size / 1e6:.1f} MB)")
        else:
            st.info("Classical model - implicit weights (forecasts .npy already computed)")

    with c_ckpt:
        mc = load_csv(model_dir / "results" / "model_comparison.csv")
        if mc is not None and not mc.empty:
            r = mc.iloc[0]
            if isinstance(r.get("r2"), float):
                st.metric("R2 (h=1)",   f"{r['r2']:.5f}")
            if isinstance(r.get("rmse"), float):
                st.metric("RMSE (h=1)", f"{r['rmse']:.5f}")

    st.divider()
    st.subheader("Forecast Preview (.npy)")
    h_sel = st.select_slider("Horizon", HORIZONS, value=10, key="pm_h")
    npy_p = forecast_npy(folder, h_sel)
    if npy_p:
        arr = load_npy(npy_p)
        if arr is not None:
            n_show = st.slider(
                "Steps to show", 100, min(5000, len(arr)), 500, key="pm_n"
            )
            fig = go.Figure(
                go.Scatter(
                    y=arr[:n_show],
                    name=f"{folder} h={h_sel}",
                    line=dict(color=MODEL_COLORS.get(folder, "#888")),
                )
            )
            fig.update_layout(
                title=f"{selected_model_name} - h={h_sel}",
                xaxis_title="Step", yaxis_title="Cp",
                template="plotly_dark", height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"`{npy_p.relative_to(PROJ)}` - shape {arr.shape}")
    else:
        st.info(f"File `{folder}_h{h_sel}.npy` not found")

    st.divider()
    st.subheader("Model Plots")
    all_imgs = []
    for d in [model_dir / "results" / "plots",
              model_dir / "results" / "forecasts"]:
        if d.exists():
            all_imgs += sorted(d.glob("*.png"))
    if all_imgs:
        cols = st.columns(min(3, len(all_imgs)))
        for i, img in enumerate(all_imgs):
            with cols[i % 3]:
                st.image(str(img), caption=img.name, use_container_width=True)
    else:
        st.info("No .png plots found for this model.")


# ===========================================================================
# TAB 4 — LSTM-direct / PatchTST
# ===========================================================================
with tab_direct:
    st.subheader("Direct Multi-Step Models: LSTM-direct & PatchTST")
    st.caption(
        "Predict all horizons in a single forward pass, "
        "without autoregressive rollout. Avoids error accumulation."
    )

    df_lstm_d = load_csv(AT_RESULTS / "lstm_direct_metrics.csv")
    df_ptst   = load_csv(AT_RESULTS / "patchtst_metrics.csv")

    if df_lstm_d is not None:
        st.subheader("LSTM-direct")
        st.dataframe(df_lstm_d, use_container_width=True)

    if df_ptst is not None:
        st.subheader("PatchTST")
        st.dataframe(df_ptst, use_container_width=True)

    # Combined chart vs best autoreg
    st.divider()
    st.subheader("Direct vs best autoreg (XGBoost, GRU)")
    mh3 = load_csv(AT_RESULTS / "multi_horizon_metrics_round3.csv")
    frames = []
    for d in [df_lstm_d, df_ptst]:
        if d is not None:
            frames.append(d[["model", "horizon", "r2", "rmse"]])
    if mh3 is not None:
        frames.append(
            mh3[mh3["model"].isin(["xgboost", "gru"])][["model", "horizon", "r2", "rmse"]]
        )
    if frames:
        df_comb = pd.concat(frames, ignore_index=True)
        m_d = st.radio("Metric", ["r2", "rmse"], horizontal=True, key="d_m")
        fig = px.line(
            df_comb, x="horizon", y=m_d, color="model",
            markers=True,
            title=f"{m_d.upper()} - Direct vs Autoreg",
            color_discrete_map=MODEL_COLORS,
            template="plotly_dark",
            log_x=True,
        )
        if m_d == "r2":
            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Learning curves
    st.divider()
    st.subheader("Training Curves (RMSE)")
    curve_files = list(AT_RESULTS.glob("*rmse_curve.npy"))
    if curve_files:
        fig_c = go.Figure()
        for cf in curve_files:
            arr = load_npy(cf)
            if arr is not None:
                fig_c.add_trace(
                    go.Scatter(
                        y=arr,
                        name=cf.stem.replace("_rmse_curve", ""),
                        mode="lines",
                    )
                )
        fig_c.update_layout(
            title="Validation RMSE per Epoch",
            xaxis_title="Epoch", yaxis_title="RMSE",
            template="plotly_dark", height=400,
        )
        st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.info("No training curve .npy files found")

    # Spectral analysis
    st.divider()
    st.subheader("Spectral Analysis of Residuals")
    spec = load_csv(AT_RESULTS / "spectral_metrics.csv")
    if spec is not None:
        st.dataframe(spec, use_container_width=True)
    psd_img = AT_RESULTS / "plots_cross_round" / "psd_residuals.png"
    if psd_img.exists():
        st.image(str(psd_img), caption="PSD of residuals", use_container_width=True)


# ===========================================================================
# TAB 5 — DIEBOLD-MARIANO
# ===========================================================================
with tab_dm:
    st.subheader("Diebold-Mariano Test - Statistical Significance")
    st.caption(
        "Tests whether forecast error differences between models "
        "are statistically significant (p < 0.05). ⚠️ This DM table runs on a "
        "single 500-step trajectory; for the robust cross-trajectory test "
        "(paired Wilcoxon + win-rate over 680 trajectories) see the "
        "**Long-horizon (corrected)** tab."
    )

    dm_files = {
        "Round 3": AT_RESULTS / "dm_test_round3.csv",
        "All":     AT_RESULTS / "dm_test_all_rounds.csv",
    }
    dm_sel = st.selectbox("DM File", list(dm_files.keys()))
    df_dm  = load_csv(dm_files[dm_sel])

    if df_dm is None:
        st.warning("DM test file not found.")
    else:
        st.dataframe(df_dm, use_container_width=True)

        # Heatmap p-values
        if {"model_a", "model_b", "p_value"}.issubset(df_dm.columns):
            try:
                pivot_p = df_dm.pivot_table(
                    index="model_a", columns="model_b", values="p_value"
                )
                fig_hm = px.imshow(
                    pivot_p,
                    title="DM p-values (< 0.05 = significant difference)",
                    color_continuous_scale="RdYlGn_r",
                    zmin=0, zmax=0.1,
                    template="plotly_dark",
                    text_auto=".3f",
                )
                fig_hm.update_layout(height=450)
                st.plotly_chart(fig_hm, use_container_width=True)
            except Exception:
                pass

        # Winner bar chart
        if "winner" in df_dm.columns:
            wins = (
                df_dm[df_dm["winner"] != "tie"]["winner"]
                .value_counts()
                .reset_index()
            )
            wins.columns = ["model", "wins"]
            fig_w = px.bar(
                wins, x="model", y="wins",
                title="DM Test Wins (h=500)",
                color="model", color_discrete_map=MODEL_COLORS,
                template="plotly_dark",
            )
            fig_w.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_w, use_container_width=True)


# ===========================================================================
# TAB 6 — CROSS-RONDA
# ===========================================================================
with tab_cross:
    st.subheader("Cross-Round Evolution: Round 1 to 2 to 3")
    st.caption(
        "R1 = 1 building (sanity check) - "
        "R2 = 5 geometries x 2 roughness (85 series) - "
        "R3 = universal (340 series)"
    )

    dfs_r = {}
    for r in ROUNDS:
        d = load_csv(AT_RESULTS / f"multi_horizon_metrics_round{r}.csv")
        if d is not None:
            d["round"] = r
            dfs_r[r] = d

    if dfs_r:
        df_all = pd.concat(dfs_r.values(), ignore_index=True)
        m_cr = st.radio("Metric", ["r2", "rmse"], horizontal=True, key="cr_m")
        h_cr = st.select_slider("Fixed Horizon", HORIZONS, value=500, key="cr_h")

        df_h = df_all[df_all["horizon"] == h_cr]
        fig = px.line(
            df_h, x="round", y=m_cr, color="model",
            markers=True,
            title=f"{m_cr.upper()} at h={h_cr} ms per round",
            color_discrete_map=MODEL_COLORS,
            template="plotly_dark",
        )
        fig.update_xaxes(
            tickvals=[1, 2, 3],
            ticktext=["R1 (1 bldg.)", "R2 (5 bldg.)", "R3 (all)"],
        )
        if m_cr == "r2":
            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No round CSVs found.")

    # Cross-round images
    st.divider()
    cross_dir = AT_RESULTS / "plots_cross_round"
    if cross_dir.exists():
        imgs = sorted(cross_dir.glob("*.png"))
        filt = st.selectbox(
            "Filter by model",
            ["all"] + [f for f, _ in MODELS],
            key="cf",
        )
        shown = [i for i in imgs if filt == "all" or filt in i.name]
        if shown:
            cols = st.columns(min(3, len(shown)))
            for i, img in enumerate(shown):
                with cols[i % 3]:
                    st.image(str(img), caption=img.name, use_container_width=True)
        else:
            st.info("No images found for this filter.")


# ===========================================================================
# TAB 7 — GRAFICAS
# ===========================================================================
with tab_plots:
    st.subheader("Saved Plots")

    # Global round plots
    plot_dirs = {f"Round {r}": AT_RESULTS / f"plots_round{r}" for r in ROUNDS}
    plot_dirs["Cross-Round"] = AT_RESULTS / "plots_cross_round"

    sel_dir = plot_dirs[st.selectbox("Global folder", list(plot_dirs.keys()))]
    imgs = sorted(sel_dir.glob("*.png")) if sel_dir.exists() else []

    if imgs:
        ncols = st.slider("Columns", 1, 4, 2, key="gc")
        cols  = st.columns(ncols)
        for i, img in enumerate(imgs):
            with cols[i % ncols]:
                st.image(str(img), caption=img.name, use_container_width=True)
    else:
        st.info("No images in this folder.")

    # Per-model plots
    st.divider()
    st.subheader("Per-Model Plots")
    pm_sel = st.selectbox(
        "Model",
        [f for f, _ in MODELS],
        format_func=lambda f: dict(MODELS).get(f, f),
        key="pm_sel",
    )
    all_imgs = []
    for d in [AT / pm_sel / "results" / "plots",
              AT / pm_sel / "results" / "forecasts"]:
        if d.exists():
            all_imgs += sorted(d.glob("*.png"))

    if all_imgs:
        cols2 = st.columns(min(3, len(all_imgs)))
        for i, img in enumerate(all_imgs):
            with cols2[i % 3]:
                st.image(str(img), caption=img.name, use_container_width=True)
    else:
        st.info(f"No plots found for {pm_sel}.")


# ===========================================================================
# TAB 8 — AGENTE IA
# ===========================================================================
with tab_agent:
    st.subheader("AI Agent - Full Pipeline")
    st.warning(
        "This tab runs the AI agent (Anthropic + LangGraph) "
        "to search literature and train new models. "
        "**Do not use if you do not want to retrain.**"
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("API Keys")
        anthropic_key = st.text_input(
            "ANTHROPIC_API_KEY", type="password",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        serpapi_key = st.text_input(
            "SERPAPI_KEY", type="password",
            value=os.environ.get("SERPAPI_KEY", ""),
        )
    with c2:
        st.subheader("Configuration")
        mode           = st.selectbox("Mode", ["seed", "model", "research", "full"], index=0)
        max_iterations = st.slider("Max iterations", 1, 5, 2)
        target_r2      = st.slider("Target R2", 0.1, 0.9, 0.3, 0.05)
        max_models     = st.slider("Max models", 1, 30, 15)

    st.divider()
    st.subheader("Environment Status")
    ok_msgs, warn_msgs, err_msgs = [], [], []

    if anthropic_key:
        ok_msgs.append("ANTHROPIC_API_KEY set")
    elif mode != "seed":
        err_msgs.append("ANTHROPIC_API_KEY required for this mode")
    else:
        warn_msgs.append("ANTHROPIC_API_KEY not set (not required in seed mode)")

    if serpapi_key:
        ok_msgs.append("SERPAPI_KEY set")
    else:
        warn_msgs.append("SERPAPI_KEY not set (Scholar search disabled)")

    postprocess_dir = PROJ / "Data" / "Data_All_The_BDH_PostProcess"
    npy_files = list(postprocess_dir.rglob("*.npy")) if postprocess_dir.exists() else []
    if npy_files:
        ok_msgs.append(f"Data: {len(npy_files)} .npy files found")
    else:
        err_msgs.append("No .npy data files found")

    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            ok_msgs.append(f"GPU: {gpu} ({mem:.1f} GB)")
        else:
            warn_msgs.append("No GPU - training will be slow on CPU")
    except ImportError:
        warn_msgs.append("PyTorch not installed")

    for m in ok_msgs:   st.success(m)
    for m in warn_msgs: st.warning(m)
    for m in err_msgs:  st.error(m)

    can_run = not err_msgs or mode == "seed"
    run_btn = st.button(
        "Run AI Agent", type="primary",
        disabled=st.session_state.running or not can_run,
    )

    log_area = st.empty()
    if st.session_state.log_text:
        log_area.text_area("Last output", st.session_state.log_text, height=400)

    if run_btn and not st.session_state.running:
        st.session_state.running = True
        st.session_state.log_text = ""

        env = os.environ.copy()
        if anthropic_key:
            env["ANTHROPIC_API_KEY"] = anthropic_key
        if serpapi_key:
            env["SERPAPI_KEY"] = serpapi_key

        cmd = [
            sys.executable, str(ROOT / "run.py"),
            "--mode", mode,
            "--max-iterations", str(max_iterations),
            "--target-r2", str(target_r2),
            "--max-models", str(max_models),
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                env=env, cwd=str(ROOT),
            )
            log = f"[{datetime.now().strftime('%H:%M:%S')}] PID {proc.pid}\n"
            for line in proc.stdout:
                log += line
                log_area.text_area("Live output", log, height=400,
                                   key=f"ll_{len(log)}")
            proc.wait()
            log += (
                f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                f"Exit code: {proc.returncode}\n"
            )
            if proc.returncode == 0:
                st.success("Pipeline completed successfully.")
            else:
                st.error(f"Process exited with code {proc.returncode}.")
            st.session_state.log_text = log
        except Exception as exc:
            st.error(f"Error running the process: {exc}")
        finally:
            st.session_state.running = False
            st.rerun()

    # Literature results
    st.divider()
    st.subheader("Model Candidates (Literature)")
    lit_json = LIT_DIR / "model_candidates.json"
    if lit_json.exists():
        candidates = json.loads(lit_json.read_text(encoding="utf-8"))
        st.metric("Candidates found", len(candidates))
        for i, c in enumerate(candidates, 1):
            with st.expander(f"{i}. {c.get('name', 'N/A')}"):
                st.json(c)
    else:
        st.info("No literature results yet.")

    # Agent logs
    LOGS_DIR.mkdir(exist_ok=True)
    log_files = sorted(LOGS_DIR.glob("run_*.log"), reverse=True)
    if log_files:
        st.divider()
        st.subheader(f"Run History ({len(log_files)} logs)")
        sel_log = st.selectbox(
            "Log",
            log_files,
            format_func=lambda p: (
                f"{p.name}  "
                f"({datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})"
            ),
        )
        try:
            content = sel_log.read_text(encoding="utf-8", errors="replace")
            st.text_area("Contents", content, height=400)
            st.download_button(
                "Download log", data=content,
                file_name=sel_log.name, mime="text/plain",
            )
        except Exception as exc:
            st.error(f"Could not read log: {exc}")

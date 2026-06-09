"""
Wind Pressure Cp — Forecasting Dashboard
=========================================
Visualiza todos los modelos entrenados en Agent_Test sin reentrenar nada.
También incluye la pestaña del agente IA (AI_Agent).

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
    page_title="Wind Cp — Model Dashboard",
    page_icon="🌬️",
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
    st.title("🌬️ Wind Cp Dashboard")
    st.caption("Agent_Test · Todos los modelos")
    st.divider()

    model_options         = {name: folder for folder, name in MODELS}
    selected_model_name   = st.selectbox("Modelo (pestaña Por modelo)", list(model_options.keys()))
    selected_model_folder = model_options[selected_model_name]

    st.divider()
    selected_round = st.selectbox(
        "Ronda", ROUNDS, index=2, format_func=lambda r: f"Round {r}"
    )
    st.divider()
    st.caption(f"Proyecto: `{PROJ.name}`")
    st.caption(f"Python: `{sys.version.split()[0]}`")

# ---------------------------------------------------------------------------
# Title + Tabs
# ---------------------------------------------------------------------------
st.title("🌬️ Wind Pressure Cp — Forecasting Dashboard")
st.caption("Todos los modelos entrenados · TPU BDH Benchmark · Sin reentrenar")

(tab_overview, tab_horizon, tab_model,
 tab_direct, tab_dm, tab_cross, tab_plots, tab_agent) = st.tabs([
    "📋 Resumen",
    "📈 Multi-horizonte",
    "🔍 Por modelo",
    "⚡ LSTM-direct / PatchTST",
    "📊 Diebold-Mariano",
    "🔄 Cross-ronda",
    "🖼️ Gráficas",
    "🤖 Agente IA",
])

# ===========================================================================
# TAB 1 — RESUMEN
# ===========================================================================
with tab_overview:
    st.subheader(f"Comparación de modelos — Round {selected_round}")

    df_comp = load_csv(AT_RESULTS / f"model_comparison_round{selected_round}.csv")
    if df_comp is None:
        st.warning("No se encontró el CSV de comparación para esta ronda.")
    else:
        # Metric cards
        cols = st.columns(len(df_comp))
        for col, (_, row) in zip(cols, df_comp.iterrows()):
            with col:
                st.metric(
                    label=row["model"],
                    value=f"R²={row['r2']:.4f}",
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
                title=f"R² (h=1) — Round {selected_round}",
                color="model", color_discrete_map=MODEL_COLORS,
                template="plotly_dark",
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(
                df_comp.sort_values("rmse"),
                x="model", y="rmse",
                title=f"RMSE (h=1) — Round {selected_round}",
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
                title="Parámetros vs Tiempo de entrenamiento",
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
    st.subheader(f"Métricas multi-horizonte — Round {selected_round}")

    df_mh = load_csv(AT_RESULTS / f"multi_horizon_metrics_round{selected_round}.csv")
    if df_mh is None:
        st.warning("No se encontró el CSV multi-horizonte para esta ronda.")
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

        metric = st.radio("Métrica", ["r2", "rmse", "mae"], horizontal=True, key="mh_m")
        fig = px.line(
            df_mh, x="horizon", y=metric, color="model",
            markers=True,
            title=f"{metric.upper()} vs Horizonte — Round {selected_round}",
            color_discrete_map=MODEL_COLORS,
            template="plotly_dark",
            log_x=True,
        )
        if metric == "r2":
            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
        fig.update_layout(height=500, xaxis_title="Horizonte (ms)")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader(f"Tabla pivot — {metric.upper()}")
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
# TAB 3 — POR MODELO
# ===========================================================================
with tab_model:
    st.subheader(f"🔍 {selected_model_name}")

    folder    = selected_model_folder
    model_dir = AT / folder

    c_info, c_ckpt = st.columns(2)
    with c_info:
        st.write(f"**Directorio:** `Agent_Test/{folder}/`")
        if model_has_checkpoint(folder):
            for c in (model_dir / "checkpoints").glob("*.pt"):
                st.success(f"Checkpoint: `{c.name}` ({c.stat().st_size / 1e6:.1f} MB)")
        else:
            st.info("Modelo clásico — pesos implícitos (forecasts .npy ya calculados)")

    with c_ckpt:
        mc = load_csv(model_dir / "results" / "model_comparison.csv")
        if mc is not None and not mc.empty:
            r = mc.iloc[0]
            if isinstance(r.get("r2"), float):
                st.metric("R² (h=1)",   f"{r['r2']:.5f}")
            if isinstance(r.get("rmse"), float):
                st.metric("RMSE (h=1)", f"{r['rmse']:.5f}")

    st.divider()
    st.subheader("Vista previa del forecast (.npy)")
    h_sel = st.select_slider("Horizonte", HORIZONS, value=10, key="pm_h")
    npy_p = forecast_npy(folder, h_sel)
    if npy_p:
        arr = load_npy(npy_p)
        if arr is not None:
            n_show = st.slider(
                "Pasos a mostrar", 100, min(5000, len(arr)), 500, key="pm_n"
            )
            fig = go.Figure(
                go.Scatter(
                    y=arr[:n_show],
                    name=f"{folder} h={h_sel}",
                    line=dict(color=MODEL_COLORS.get(folder, "#888")),
                )
            )
            fig.update_layout(
                title=f"{selected_model_name} · h={h_sel}",
                xaxis_title="Paso", yaxis_title="Cp",
                template="plotly_dark", height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"`{npy_p.relative_to(PROJ)}` — shape {arr.shape}")
    else:
        st.info(f"No se encontró `{folder}_h{h_sel}.npy`")

    st.divider()
    st.subheader("Gráficas del modelo")
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
        st.info("No se encontraron gráficas .png para este modelo.")


# ===========================================================================
# TAB 4 — LSTM-direct / PatchTST
# ===========================================================================
with tab_direct:
    st.subheader("⚡ Modelos directos multi-paso: LSTM-direct & PatchTST")
    st.caption(
        "Predicen todos los horizontes en un solo forward pass, "
        "sin rollout autoregresivo. Evitan la acumulación de error."
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
    st.subheader("Directos vs mejores autoreg (XGBoost, GRU)")
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
        m_d = st.radio("Métrica", ["r2", "rmse"], horizontal=True, key="d_m")
        fig = px.line(
            df_comb, x="horizon", y=m_d, color="model",
            markers=True,
            title=f"{m_d.upper()} — Directos vs Autoreg",
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
    st.subheader("Curvas de entrenamiento (RMSE)")
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
            title="RMSE en validación por época",
            xaxis_title="Época", yaxis_title="RMSE",
            template="plotly_dark", height=400,
        )
        st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.info("No se encontraron curvas de entrenamiento .npy")

    # Spectral analysis
    st.divider()
    st.subheader("Análisis espectral de residuos")
    spec = load_csv(AT_RESULTS / "spectral_metrics.csv")
    if spec is not None:
        st.dataframe(spec, use_container_width=True)
    psd_img = AT_RESULTS / "plots_cross_round" / "psd_residuals.png"
    if psd_img.exists():
        st.image(str(psd_img), caption="PSD de residuos", use_container_width=True)


# ===========================================================================
# TAB 5 — DIEBOLD-MARIANO
# ===========================================================================
with tab_dm:
    st.subheader("Test Diebold-Mariano — Significancia estadística")
    st.caption(
        "Prueba si las diferencias de error de predicción entre modelos "
        "son estadísticamente significativas (p < 0.05)."
    )

    dm_files = {
        "Round 3": AT_RESULTS / "dm_test_round3.csv",
        "Todos":   AT_RESULTS / "dm_test_all_rounds.csv",
    }
    dm_sel = st.selectbox("Archivo DM", list(dm_files.keys()))
    df_dm  = load_csv(dm_files[dm_sel])

    if df_dm is None:
        st.warning("No se encontró el archivo de test DM.")
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
                    title="p-values DM (< 0.05 = diferencia significativa)",
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
            wins.columns = ["modelo", "victorias"]
            fig_w = px.bar(
                wins, x="modelo", y="victorias",
                title="Victorias en test DM (h=500)",
                color="modelo", color_discrete_map=MODEL_COLORS,
                template="plotly_dark",
            )
            fig_w.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_w, use_container_width=True)


# ===========================================================================
# TAB 6 — CROSS-RONDA
# ===========================================================================
with tab_cross:
    st.subheader("Evolución cross-ronda: Round 1 → 2 → 3")
    st.caption(
        "R1 = 1 edificio (sanity check) · "
        "R2 = 5 geometrías × 2 roughness (85 series) · "
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
        m_cr = st.radio("Métrica", ["r2", "rmse"], horizontal=True, key="cr_m")
        h_cr = st.select_slider("Horizonte fijo", HORIZONS, value=500, key="cr_h")

        df_h = df_all[df_all["horizon"] == h_cr]
        fig = px.line(
            df_h, x="round", y=m_cr, color="model",
            markers=True,
            title=f"{m_cr.upper()} en h={h_cr} ms por ronda",
            color_discrete_map=MODEL_COLORS,
            template="plotly_dark",
        )
        fig.update_xaxes(
            tickvals=[1, 2, 3],
            ticktext=["R1 (1 edif.)", "R2 (5 edif.)", "R3 (todos)"],
        )
        if m_cr == "r2":
            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No se encontraron CSVs de rondas.")

    # Cross-round images
    st.divider()
    cross_dir = AT_RESULTS / "plots_cross_round"
    if cross_dir.exists():
        imgs = sorted(cross_dir.glob("*.png"))
        filt = st.selectbox(
            "Filtrar por modelo",
            ["todos"] + [f for f, _ in MODELS],
            key="cf",
        )
        shown = [i for i in imgs if filt == "todos" or filt in i.name]
        if shown:
            cols = st.columns(min(3, len(shown)))
            for i, img in enumerate(shown):
                with cols[i % 3]:
                    st.image(str(img), caption=img.name, use_container_width=True)
        else:
            st.info("No hay imágenes para ese filtro.")


# ===========================================================================
# TAB 7 — GRAFICAS
# ===========================================================================
with tab_plots:
    st.subheader("Gráficas guardadas")

    # Global round plots
    plot_dirs = {f"Round {r}": AT_RESULTS / f"plots_round{r}" for r in ROUNDS}
    plot_dirs["Cross-ronda"] = AT_RESULTS / "plots_cross_round"

    sel_dir = plot_dirs[st.selectbox("Carpeta global", list(plot_dirs.keys()))]
    imgs = sorted(sel_dir.glob("*.png")) if sel_dir.exists() else []

    if imgs:
        ncols = st.slider("Columnas", 1, 4, 2, key="gc")
        cols  = st.columns(ncols)
        for i, img in enumerate(imgs):
            with cols[i % ncols]:
                st.image(str(img), caption=img.name, use_container_width=True)
    else:
        st.info("No hay imágenes en esta carpeta.")

    # Per-model plots
    st.divider()
    st.subheader("Gráficas por modelo individual")
    pm_sel = st.selectbox(
        "Modelo",
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
        st.info(f"No hay gráficas para {pm_sel}.")


# ===========================================================================
# TAB 8 — AGENTE IA
# ===========================================================================
with tab_agent:
    st.subheader("🤖 Agente IA — Pipeline completo")
    st.warning(
        "Esta pestaña ejecuta el agente de IA (Anthropic + LangGraph) "
        "para buscar literatura y entrenar modelos nuevos. "
        "**No la uses si no quieres reentrenar.**"
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔑 API Keys")
        anthropic_key = st.text_input(
            "ANTHROPIC_API_KEY", type="password",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        serpapi_key = st.text_input(
            "SERPAPI_KEY", type="password",
            value=os.environ.get("SERPAPI_KEY", ""),
        )
    with c2:
        st.subheader("Configuración")
        mode           = st.selectbox("Modo", ["seed", "model", "research", "full"], index=0)
        max_iterations = st.slider("Max iteraciones", 1, 5, 2)
        target_r2      = st.slider("Target R²", 0.1, 0.9, 0.3, 0.05)
        max_models     = st.slider("Max modelos", 1, 30, 15)

    st.divider()
    st.subheader("Estado del entorno")
    ok_msgs, warn_msgs, err_msgs = [], [], []

    if anthropic_key:
        ok_msgs.append("ANTHROPIC_API_KEY correcta")
    elif mode != "seed":
        err_msgs.append("ANTHROPIC_API_KEY requerida para este modo")
    else:
        warn_msgs.append("ANTHROPIC_API_KEY no configurada (no requerida en modo seed)")

    if serpapi_key:
        ok_msgs.append("SERPAPI_KEY correcta")
    else:
        warn_msgs.append("SERPAPI_KEY no configurada (búsqueda Scholar desactivada)")

    postprocess_dir = PROJ / "Data" / "Data_All_The_BDH_PostProcess"
    npy_files = list(postprocess_dir.rglob("*.npy")) if postprocess_dir.exists() else []
    if npy_files:
        ok_msgs.append(f"Datos: {len(npy_files)} archivos .npy encontrados")
    else:
        err_msgs.append("No se encontraron archivos .npy de datos")

    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            ok_msgs.append(f"GPU: {gpu} ({mem:.1f} GB)")
        else:
            warn_msgs.append("Sin GPU — entrenamiento lento en CPU")
    except ImportError:
        warn_msgs.append("PyTorch no instalado")

    for m in ok_msgs:   st.success(m)
    for m in warn_msgs: st.warning(m)
    for m in err_msgs:  st.error(m)

    can_run = not err_msgs or mode == "seed"
    run_btn = st.button(
        "▶ Iniciar Agente IA", type="primary",
        disabled=st.session_state.running or not can_run,
    )

    log_area = st.empty()
    if st.session_state.log_text:
        log_area.text_area("Última salida", st.session_state.log_text, height=400)

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
                log_area.text_area("Salida en vivo", log, height=400,
                                   key=f"ll_{len(log)}")
            proc.wait()
            log += (
                f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                f"Código de salida: {proc.returncode}\n"
            )
            if proc.returncode == 0:
                st.success("Pipeline completado exitosamente.")
            else:
                st.error(f"Proceso terminó con código {proc.returncode}.")
            st.session_state.log_text = log
        except Exception as exc:
            st.error(f"Error al ejecutar el proceso: {exc}")
        finally:
            st.session_state.running = False
            st.rerun()

    # Literature results
    st.divider()
    st.subheader("Candidatos de modelos (literatura)")
    lit_json = LIT_DIR / "model_candidates.json"
    if lit_json.exists():
        candidates = json.loads(lit_json.read_text(encoding="utf-8"))
        st.metric("Candidatos encontrados", len(candidates))
        for i, c in enumerate(candidates, 1):
            with st.expander(f"{i}. {c.get('name', 'N/A')}"):
                st.json(c)
    else:
        st.info("No hay resultados de literatura aún.")

    # Agent logs
    LOGS_DIR.mkdir(exist_ok=True)
    log_files = sorted(LOGS_DIR.glob("run_*.log"), reverse=True)
    if log_files:
        st.divider()
        st.subheader(f"Historial de ejecuciones ({len(log_files)} logs)")
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
            st.text_area("Contenido", content, height=400)
            st.download_button(
                "⬇ Descargar log", data=content,
                file_name=sel_log.name, mime="text/plain",
            )
        except Exception as exc:
            st.error(f"No se pudo leer el log: {exc}")

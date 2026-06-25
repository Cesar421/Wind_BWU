#!/usr/bin/env bash
# run_all_gpu.sh — reproduce the full pipeline on a GPU machine (Linux/Mac).
# Usage (from Agent_Test/, with the torch+CUDA env activated):
#   bash run_all_gpu.sh
set +e
export KMP_DUPLICATE_LIB_OK=TRUE
cd "$(dirname "$0")"   # Agent_Test/

step() { echo -e "\n==================== $1 ===================="; }

step "A) Autoregressive Round 3 (LSTM/GRU/TCN + Ridge/XGBoost/RF) -> creates checkpoints"
python train_all.py --scope all

step "B) Direct multi-step models (GPU) -> regenerates absent checkpoints + trues/preds"
python lstm/train_lstm_direct.py --scope all --horizon 500
python patchtst/train_patchtst.py --scope all --horizon 500

step "C) Multi-trajectory evaluation (incl. Random Forest re-fit, ~2 h CPU)"
python evaluate_long_horizon.py --classical --rf

step "D) Exact dense naive baseline"
python naive_dense_baseline.py

step "E) Spectral analysis (needs the direct trues/preds from step B)"
python spectral_analysis.py

step "F) Legacy single-trajectory tables (feed the Streamlit dashboard)"
python analyze_horizons.py --scope all
python diebold_mariano.py

echo -e "\nDONE. Review results/ then update docs + thesis and commit/push."

# run_all_gpu.ps1 — reproduce the full pipeline on a GPU machine.
# Usage (from Agent_Test/, with the torch+CUDA conda env activated):
#   $env:KMP_DUPLICATE_LIB_OK = "TRUE"
#   ./run_all_gpu.ps1
# Re-runnable: skip steps you don't need by commenting them out.

$ErrorActionPreference = "Continue"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Set-Location $PSScriptRoot   # Agent_Test/

function Step($msg) { Write-Host "`n==================== $msg ====================" -ForegroundColor Cyan }

Step "A) Autoregressive Round 3 (LSTM/GRU/TCN + Ridge/XGBoost/RF) -> creates checkpoints"
python train_all.py --scope all

Step "B) Direct multi-step models (GPU) -> regenerates absent checkpoints + trues/preds"
python lstm/train_lstm_direct.py --scope all --horizon 500
python patchtst/train_patchtst.py --scope all --horizon 500

Step "C) Multi-trajectory evaluation (incl. Random Forest re-fit, ~2 h CPU)"
python evaluate_long_horizon.py --classical --rf

Step "D) Exact dense naive baseline"
python naive_dense_baseline.py

Step "E) Spectral analysis (needs the direct trues/preds from step B)"
python spectral_analysis.py

Step "F) Legacy single-trajectory tables (feed the Streamlit dashboard)"
python analyze_horizons.py --scope all
python diebold_mariano.py

Write-Host "`nDONE. Review results/ then update docs + thesis and commit/push." -ForegroundColor Green

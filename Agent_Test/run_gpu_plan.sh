#!/usr/bin/env bash
# run_gpu_plan.sh — GPU plan, protocol B; bash twin of run_gpu_plan.ps1
# (Handoffs/2026-10-21_plan_GPU_ventana_y_restriccion_estadistica.md).
#
# Usage (from Agent_Test/, torch+CUDA env activated):
#   ./run_gpu_plan.sh                    # PYTHON=/path/to/python to override
#   SKIP_R3=1 ./run_gpu_plan.sh          # only the R2 phases
# Order: Phase 1 -> SEQ* -> Phase 2 -> Phases 4 + 5 (R3) -> controls. Finished runs
# are skipped (relaunch after a crash); a failing run is logged and the batch goes
# on, except in Phase 1.
set -uo pipefail
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8
PY=${PYTHON:-python}
OUT=results/gpu_plan_2026-10
LOG=$OUT/run_gpu_plan.log
mkdir -p "$OUT"
FAILED=()

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
train() {
    say "train_patchtst_v2.py $*"
    "$PY" -u patchtst/train_patchtst_v2.py "$@" || { say "  FAILED: $*"; FAILED+=("$*"); return 1; }
}

say "----- Phase 1: look-back window (MSE, R2) -----"
train --scope round2 --tag 1a --seq 100  --patch-len 16  --patch-stride 8  || exit 1
train --scope round2 --tag 1b --seq 1000 --patch-len 64  --patch-stride 32 || exit 1
train --scope round2 --tag 1c --seq 2000 --patch-len 128 --patch-stride 64 || exit 1

# SEQ* = lowest best val MSE among 1a/1b/1c
read -r SEQ PL PS < <("$PY" - <<'EOF'
import glob, json, sys
cfg = {"1a": (100, 16, 8), "1b": (1000, 64, 32), "1c": (2000, 128, 64)}
val = {}
for tag in cfg:
    (path,) = glob.glob(f"results/gpu_plan_2026-10/runs/{tag}_*/summary.json")
    val[tag] = json.load(open(path))["best_val_mse"]
    print(f"  {tag}: seq {cfg[tag][0]:>4}  best val MSE {val[tag]:.6f}", file=sys.stderr)
print(*cfg[min(val, key=val.get)])
EOF
)
say "SEQ* = $SEQ (patch $PL/$PS): lowest best val MSE"
S=(--seq "$SEQ" --patch-len "$PL" --patch-stride "$PS")

say "----- Phase 2: template-free StatLoss with SEQ* = $SEQ (R2) -----"
for spec in "2d 0" "2a 0.0003" "2b 0.003" "2c 0.03" "2f 0.1"; do
    set -- $spec
    train --scope round2 --tag "$1" "${S[@]}" --lam-psd "$2" --lam-std 1
done

if [ "${SKIP_R3:-0}" != 1 ]; then
    say "----- Phases 4 + 5: SEQ* MSE model on R3 (+ leave-buildings-out) -----"
    train --scope all --tag 4a "${S[@]}"
    train --scope all --tag 5a "${S[@]}" --exclude-ratios 1_1_4 3_1_3
fi

say "----- Controls -----"
train --scope round2 --tag 2e "${S[@]}" --lam-psd 0.003 --lam-std 1 --psd-bins linear
train --scope round2 --tag 2c-legacy "${S[@]}" --lam-psd 0.03 --lam-std 1 --legacy-statloss
train --scope round2 --tag 1a-orig --seq 100 --patch-len 16 --patch-stride 8 --align start

[ ${#FAILED[@]} -gt 0 ] && say "FAILED runs: ${FAILED[*]}"
say "ALL DONE. Next: python analyze_gpu_plan.py, then choose lambda."

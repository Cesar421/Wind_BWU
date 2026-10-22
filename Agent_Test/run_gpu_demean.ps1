# run_gpu_demean.ps1 — experiment 6: remove each series' own train-portion mean
# (--demean: in memory, causal; the data files are only read) + seed repeats.
#
# Usage (from Agent_Test/, in a PowerShell window of its own):
#   .\run_gpu_demean.ps1 -Python <path to the env's python.exe>
#
# All runs: full scope (R3), seq 2000, patches 128/64, MSE (the final model 4a).
#   6a          4a + --demean                                   (seed 42)
#   6b          6a, leave-buildings-out (no 1_1_4 / 3_1_3; their own train means)
#   6a, 4a      seeds 43 and 44, interleaved -> 3 seeds of each to compare
# Finished runs are skipped (relaunch after a crash); a failing run is logged and
# the batch goes on. Log: results/gpu_plan_2026-10/run_gpu_demean.log

param([string]$Python = "python")

$ErrorActionPreference = "Continue"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot   # Agent_Test/

$OUT = Join-Path $PSScriptRoot "results\gpu_plan_2026-10"
$LOG = Join-Path $OUT "run_gpu_demean.log"
$T_START = Get-Date
$FAILED = @()

function Say([string]$msg, [string]$color = "Cyan") {
    $line = "[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $msg
    Write-Host $line -ForegroundColor $color
    Add-Content -Path $LOG -Value $line -Encoding UTF8
}

function Train([string[]]$a) {
    Say ("train_patchtst_v2.py " + ($a -join " "))
    $t0 = Get-Date
    & $Python -u patchtst/train_patchtst_v2.py @a
    if ($LASTEXITCODE -ne 0) {
        Say "  FAILED (exit code $LASTEXITCODE): see that run's train.log" "Red"
        $script:FAILED += ($a -join " ")
        return
    }
    Say ("  done in {0:N1} min" -f ((Get-Date) - $t0).TotalMinutes) "Green"
}

$M = @("--scope", "all", "--seq", "2000", "--patch-len", "128", "--patch-stride", "64")
Say "===== Experiment 6: per-series train mean removed + seeds (python: $Python) =====" "Yellow"
Train (@("--tag", "6a", "--demean") + $M)
Train (@("--tag", "6b", "--demean", "--exclude-ratios", "1_1_4", "3_1_3") + $M)
foreach ($seed in @("43", "44")) {
    Train (@("--tag", "6a", "--demean", "--seed", $seed) + $M)
    Train (@("--tag", "4a", "--seed", $seed) + $M)
}

if ($FAILED.Count -gt 0) { Say ("FAILED runs:`n  " + ($FAILED -join "`n  ")) "Red" }
Say ("ALL DONE in {0:N1} h. Next: compare 4a vs 6a over the 3 seeds." -f `
     ((Get-Date) - $T_START).TotalHours) "Green"

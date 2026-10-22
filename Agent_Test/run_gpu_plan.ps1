# run_gpu_plan.ps1 â€” GPU plan, protocol B (overnight batch)
# (Handoffs/2026-10-21_plan_GPU_ventana_y_restriccion_estadistica.md).
#
# Usage (from Agent_Test/, in a PowerShell window of its own â€” not inside VS Code):
#   $env:KMP_DUPLICATE_LIB_OK = "TRUE"
#   .\run_gpu_plan.ps1 -Python <path to the env's python.exe>
#   .\run_gpu_plan.ps1 -SkipR3            # only the R2 phases
#
# Order (by importance):
#   Phase 1  look-back window, MSE, R2 (1a/1b/1c) -> SEQ* = lowest best val MSE
#   Phase 2  template-free StatLoss with SEQ*, R2 (2d, 2a, 2b, 2c, 2f)
#   Phase 4  SEQ* MSE model on the full scope (R3)            [independent of lambda]
#   Phase 5  same, leave-buildings-out (no 1_1_4 / 3_1_3)     [independent of lambda]
#   Controls 2e (linear bins), 2c-legacy (original StatLoss), 1a-orig (original patching)
# The training schedule (patience / LR patience / max epochs) defaults to the
# published runs' optimizer steps (train_patchtst_v2.py). Finished runs
# (summary.json present) are skipped: after a crash, just relaunch. A failing
# run is logged and the batch goes on, except in Phase 1 (SEQ* needs all three).
# One log per run in results/gpu_plan_2026-10/runs/<run>/train.log;
# commands, timings and decisions in run_gpu_plan.log.

param(
    [string]$Python = "python",
    [switch]$SkipR3
)

$ErrorActionPreference = "Continue"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $PSScriptRoot   # Agent_Test/

$OUT = Join-Path $PSScriptRoot "results\gpu_plan_2026-10"
$LOG = Join-Path $OUT "run_gpu_plan.log"
New-Item -ItemType Directory -Force $OUT | Out-Null
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

function Summary([string]$tag) {
    $f = @(Get-ChildItem (Join-Path $OUT "runs") -Directory -Filter "$($tag)_*" |
           ForEach-Object { Join-Path $_.FullName "summary.json" } |
           Where-Object { Test-Path $_ })
    if ($f.Count -ne 1) { return $null }
    Get-Content $f[0] -Raw | ConvertFrom-Json
}

Say "===== GPU plan, protocol B (python: $Python) =====" "Yellow"

# â”€â”€ Phase 1: look-back window, MSE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
$P1 = @(
    @{ tag = "1a"; seq = "100";  pl = "16";  ps = "8"  },
    @{ tag = "1b"; seq = "1000"; pl = "64";  ps = "32" },
    @{ tag = "1c"; seq = "2000"; pl = "128"; ps = "64" }
)
Say "----- Phase 1: look-back window (MSE, R2) -----" "Yellow"
foreach ($c in $P1) {
    Train @("--scope", "round2", "--tag", $c.tag, "--seq", $c.seq,
                    "--patch-len", $c.pl, "--patch-stride", $c.ps)
}

# SEQ* = lowest best val MSE (choose on validation, report on test)
$best = $null
foreach ($c in $P1) {
    $s = Summary $c.tag
    if ($null -eq $s) { Say "Phase 1 run $($c.tag) missing -> cannot choose SEQ*, stopping" "Red"; exit 1 }
    Say ("  {0}: seq {1,4}  best val MSE {2:N6}  (test RMSE@500 {3:N4}, mean over h {4:N4})" -f `
         $c.tag, $c.seq, [double]$s.best_val_mse, [double]$s.test.rmse_h500,
         [double]$s.test.rmse_mean) "White"
    if ($null -eq $best -or [double]$s.best_val_mse -lt $best.val_mse) {
        $best = @{ tag = $c.tag; seq = $c.seq; pl = $c.pl; ps = $c.ps;
                   val_mse = [double]$s.best_val_mse }
    }
}
Say ("SEQ* = {0} (run {1}, patch {2}/{3}): lowest best val MSE" -f `
     $best.seq, $best.tag, $best.pl, $best.ps) "Yellow"
$SEQ = @("--seq", $best.seq, "--patch-len", $best.pl, "--patch-stride", $best.ps)

# â”€â”€ Phase 2: template-free StatLoss with SEQ* â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Say "----- Phase 2: StatLoss (template-free) with SEQ* = $($best.seq) (R2) -----" "Yellow"
foreach ($c in @(
        @{ tag = "2d"; lpsd = "0" },
        @{ tag = "2a"; lpsd = "0.0003" },
        @{ tag = "2b"; lpsd = "0.003" },
        @{ tag = "2c"; lpsd = "0.03" },
        @{ tag = "2f"; lpsd = "0.1" })) {
    Train (@("--scope", "round2", "--tag", $c.tag) + $SEQ +
                   @("--lam-psd", $c.lpsd, "--lam-std", "1"))
}

# â”€â”€ Phases 4 + 5: SEQ* MSE model on the full scope (independent of lambda) â”€â”€â”€
if (-not $SkipR3) {
    Say "----- Phase 4: SEQ* MSE model on R3 -----" "Yellow"
    Train (@("--scope", "all", "--tag", "4a") + $SEQ)
    Say "----- Phase 5: leave-buildings-out (train without 1_1_4 and 3_1_3, test on them) -----" "Yellow"
    Train (@("--scope", "all", "--tag", "5a") + $SEQ +
                   @("--exclude-ratios", "1_1_4", "3_1_3"))
}

# â”€â”€ Controls â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Say "----- Controls: linear bins, original StatLoss, original patching (R2) -----" "Yellow"
Train (@("--scope", "round2", "--tag", "2e") + $SEQ +
               @("--lam-psd", "0.003", "--lam-std", "1", "--psd-bins", "linear"))
Train (@("--scope", "round2", "--tag", "2c-legacy") + $SEQ +
               @("--lam-psd", "0.03", "--lam-std", "1", "--legacy-statloss"))
Train @("--scope", "round2", "--tag", "1a-orig", "--seq", "100", "--patch-len", "16",
                "--patch-stride", "8", "--align", "start")

# â”€â”€ Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
$m = Import-Csv (Join-Path $OUT "metrics.csv") | Where-Object { $_.h -eq "500" }
$table = $m | Group-Object run | ForEach-Object {
    $v = $_.Group | Where-Object split -eq "val"
    $t = $_.Group | Where-Object split -eq "test"
    [pscustomobject]@{
        run           = $_.Name
        val_rmse_mean = "{0:N4}" -f [double]$v.rmse_mean
        test_rmse_mean = "{0:N4}" -f [double]$t.rmse_mean
        test_rmse500  = "{0:N4}" -f [double]$t.rmse
        power_ratio   = "{0:N3}" -f [double]$t.power_ratio_total
        tpl_share     = "{0:N3}" -f [double]$t.tpl_share
    }
}
Say ("Summary:`n" + ($table | Format-Table -AutoSize | Out-String).TrimEnd()) "White"
if ($FAILED.Count -gt 0) { Say ("FAILED runs:`n  " + ($FAILED -join "`n  ")) "Red" }
Say ("ALL DONE in {0:N1} h. Next: python analyze_gpu_plan.py, then choose lambda." -f `
     ((Get-Date) - $T_START).TotalHours) "Green"

"""
Spectral analysis (Section F) of the long-horizon forecasting residuals.

Hypothesis: if the long-horizon RMSE floor at h=500 (~0.175) is an irreducible
turbulent-noise property of the wind-tunnel signal — and NOT a modelling
limitation — then the residual r(t) = y_true(t) - y_pred(t) should look like
broadband noise whose power spectral density (PSD) approximates the PSD of
the high-frequency tail of the true signal itself. Both LSTM-direct and
PatchTST should produce statistically indistinguishable residual spectra.

Inputs (Round 3, H=500, denormalised Cp):
  results/lstm_direct_h500_trues.npy  (N, 500)
  results/lstm_direct_h500_preds.npy  (N, 500)
  results/patchtst_h500_r3_trues.npy  (N, 500)
  results/patchtst_h500_r3_preds.npy  (N, 500)

Outputs:
  results/plots_cross_round/psd_residuals.png
  results/spectral_metrics.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = RES / "plots_cross_round"
OUT.mkdir(parents=True, exist_ok=True)

FS = 1000.0   # 1 kHz sampling (BDH wind tunnel)
NPERSEG = 256 # 256/1000 = 256 ms ; df = 1000/256 ≈ 3.9 Hz
N_SAMPLE = 5000  # subsample of test windows for speed (146 880 total)


def avg_welch(arr: np.ndarray, fs: float, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Average Welch PSD across rows.
    arr: (N, T)  -> returns (freqs, psd_avg)
    Each row of length T is one trajectory of 500 samples.
    """
    rng = np.random.default_rng(0)
    if arr.shape[0] > N_SAMPLE:
        idx = rng.choice(arr.shape[0], N_SAMPLE, replace=False)
        arr = arr[idx]
    psds = []
    for row in arr:
        f, p = welch(row, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                     window="hann", detrend="constant", scaling="density")
        psds.append(p)
    return f, np.mean(psds, axis=0)


def spectral_summary(f, psd, label):
    """Total power, dominant frequency, and band-limited power summary."""
    total = np.trapezoid(psd, f)
    f_peak = f[np.argmax(psd)]
    lo = np.trapezoid(psd[(f < 20)], f[(f < 20)])
    mid = np.trapezoid(psd[(f >= 20) & (f < 100)], f[(f >= 20) & (f < 100)])
    hi = np.trapezoid(psd[f >= 100], f[f >= 100])
    print(f"{label:<25s} total={total:.4e}  f_peak={f_peak:6.1f} Hz  "
          f"P[<20]={lo:.3e}  P[20-100]={mid:.3e}  P[>100]={hi:.3e}")
    return {"model": label, "total_power": total, "f_peak_hz": f_peak,
            "P_lt_20hz": lo, "P_20_100hz": mid, "P_gt_100hz": hi}


def load_or_warn(p: Path) -> np.ndarray | None:
    if not p.exists():
        print(f"  [missing] {p.name}")
        return None
    return np.load(p)


print("Loading test arrays …")
lstm_true = load_or_warn(RES / "lstm_direct_h500_trues.npy")
lstm_pred = load_or_warn(RES / "lstm_direct_h500_preds.npy")
pt_true   = load_or_warn(RES / "patchtst_h500_r3_trues.npy")
pt_pred   = load_or_warn(RES / "patchtst_h500_r3_preds.npy")

assert lstm_true is not None and lstm_pred is not None, "LSTM-direct arrays missing"
assert pt_true   is not None and pt_pred   is not None, "PatchTST arrays missing"
print(f"  LSTM-direct: trues {lstm_true.shape}  preds {lstm_pred.shape}")
print(f"  PatchTST:    trues {pt_true.shape}    preds {pt_pred.shape}")

print(f"\nWelch parameters: fs={FS} Hz  nperseg={NPERSEG}  df={FS/NPERSEG:.2f} Hz  "
      f"subsample={N_SAMPLE}")

print("\nComputing PSDs (subsample of test windows) …")
f_t,  psd_true_l = avg_welch(lstm_true, FS, NPERSEG)
_,    psd_pred_l = avg_welch(lstm_pred, FS, NPERSEG)
_,    psd_res_l  = avg_welch(lstm_true - lstm_pred, FS, NPERSEG)
_,    psd_true_p = avg_welch(pt_true,   FS, NPERSEG)
_,    psd_pred_p = avg_welch(pt_pred,   FS, NPERSEG)
_,    psd_res_p  = avg_welch(pt_true - pt_pred, FS, NPERSEG)

print("\n-- Spectral summary --")
rows = []
rows.append(spectral_summary(f_t, psd_true_l, "true_LSTM-direct"))
rows.append(spectral_summary(f_t, psd_pred_l, "pred_LSTM-direct"))
rows.append(spectral_summary(f_t, psd_res_l,  "resid_LSTM-direct"))
rows.append(spectral_summary(f_t, psd_true_p, "true_PatchTST"))
rows.append(spectral_summary(f_t, psd_pred_p, "pred_PatchTST"))
rows.append(spectral_summary(f_t, psd_res_p,  "resid_PatchTST"))

pd.DataFrame(rows).to_csv(RES / "spectral_metrics.csv", index=False)
print(f"\nSpectral metrics saved to {RES/'spectral_metrics.csv'}")

# ─── Plot ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

ax = axes[0]
ax.loglog(f_t, psd_true_l, "-",  color="black",      lw=2.0, label="True $C_p$ signal")
ax.loglog(f_t, psd_pred_l, "--", color="#9400d3",    lw=2.0, label="LSTM-direct prediction")
ax.loglog(f_t, psd_res_l,  ":",  color="#d62728",    lw=2.0, label="Residual (true − pred)")
ax.set_xlabel("Frequency [Hz]", fontsize=11)
ax.set_ylabel("PSD [$C_p^2$/Hz]", fontsize=11)
ax.set_title("LSTM-direct (R3, h=500)", fontsize=12)
ax.legend(loc="lower left", fontsize=9, framealpha=0.92)
ax.grid(alpha=0.3, which="both")

ax = axes[1]
ax.loglog(f_t, psd_true_p, "-",  color="black",      lw=2.0, label="True $C_p$ signal")
ax.loglog(f_t, psd_pred_p, "--", color="#ff6f00",    lw=2.0, label="PatchTST prediction")
ax.loglog(f_t, psd_res_p,  ":",  color="#d62728",    lw=2.0, label="Residual (true − pred)")
ax.set_xlabel("Frequency [Hz]", fontsize=11)
ax.set_title("PatchTST (R3, h=500)", fontsize=12)
ax.legend(loc="lower left", fontsize=9, framealpha=0.92)
ax.grid(alpha=0.3, which="both")

fig.suptitle("Power spectral density — wind $C_p$ predictions vs. residuals "
             "(Round 3, direct multi-step)", fontsize=13, fontweight="bold")
fig.tight_layout()
out_path = OUT / "psd_residuals.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}")

# ─── Crossover analysis: at which freq does pred-PSD diverge from true-PSD? ──
def crossover(f, psd_true, psd_pred, ratio=0.5):
    """Lowest frequency at which predicted PSD falls below `ratio` × true PSD."""
    mask = (f > 0) & (psd_pred < ratio * psd_true)
    if not mask.any():
        return None
    return float(f[mask][0])

fc_l = crossover(f_t, psd_true_l, psd_pred_l)
fc_p = crossover(f_t, psd_true_p, psd_pred_p)
print(f"\nCrossover (pred PSD drops below 50% of true PSD):")
print(f"  LSTM-direct: f_c = {fc_l} Hz")
print(f"  PatchTST:    f_c = {fc_p} Hz")
print("Above this frequency the model produces a smoothed signal — that's "
      "where the residual power lives, and where the 0.175 RMSE comes from.")

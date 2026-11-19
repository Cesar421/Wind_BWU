"""
Gaussian AR(1) error layer ("bias") on the best PatchTST, run 4a — advisor's suggestion,
November 2026.

Why: the MSE-trained direct forecast is close to the conditional mean, so its spectrum
collapses (1-2 % of the true power), and putting the PSD into the loss (run 2f) buys power
with RMSE and no information. Here the variability goes into an error model and the point
forecast stays as it is:

    Cp(t+h) = 4a(t+h) + mu(h) + sigma(h) * z_h,          h = 1..500 (ms)
    z_h = rho * z_(h-1) + sqrt(1 - rho^2) * eps_h,      eps_h ~ N(0, 1),  z_1 ~ N(0, 1)

  mu(h), sigma(h)  mean and std of the validation errors y - 4a at each horizon
  rho, |rho| < 1   one constant -> exponential autocorrelation rho^k. Chosen on the
                   validation windows so that the spectrum of the sampled trajectories
                   matches the true one in octave bands: the "PSD constraint".

Fitted on validation only, scored on test (R3, 150 280 windows):
  - RMSE of the mean forecast (unchanged by construction) and of one trajectory;
  - spectrum of the trajectories vs the truth (same periodogram as train_patchtst_v2.py);
  - coverage of the 50/80/90/95 % intervals and the Gaussian CRPS, against the per-series
    climatology N(mu_s, sigma_s^2) fitted on each series' train part;
  - on the 85 R2 series (the same 37 570 windows run 2f was scored on): 4a + bias vs 2f.

Stages (predictions are cached as float32 .npy in <out>/big/, gitignored):
    python bias_gaussian_ar1.py                    # predict if not cached, analyze, gallery
    python bias_gaussian_ar1.py --stage predict    # ~9 min on a 4-thread CPU
    python bias_gaussian_ar1.py --stage analyze    # a few minutes
    python bias_gaussian_ar1.py --stage gallery [--angle 0] [--pct 50]
        # one figure per terrain roughness, one panel per building (B:D:H), seconds
    python bias_gaussian_ar1.py --every 40 --out <tmp dir>   # quick end-to-end check
Outputs: results/gpu_plan_2026-10/bias_ar1/
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.signal import lfilter  # noqa: E402
from scipy.special import ndtr, ndtri  # noqa: E402
from scipy.stats import kurtosis, skew  # noqa: E402

AGENT = Path(__file__).resolve().parent
sys.path[:0] = [str(AGENT), str(AGENT / "patchtst")]
from data_loader import list_series, load_all_buildings_ctx, load_faces  # noqa: E402
from train_utils import ROUND2_ALPHAS, ROUND2_RATIOS, WindowLoader  # noqa: E402
from models.patchtst import PatchTST  # noqa: E402

PLAN = AGENT / "results" / "gpu_plan_2026-10"
RUNS = PLAN / "runs"
OUT_DEFAULT = PLAN / "bias_ar1"
RUN_BEST = "4a_s2000_p128-64_mse_r3"
RUN_PSD = "2f_s2000_p128-64_lpsd0.1_lstd1_r2"
FS = 1000.0
TRAIN_R = 0.70
OCTAVE_EDGES = [2, 4, 8, 16, 32, 64, 128, 256, 501]   # Hz, the StatLoss bands of run 2f
COARSE_EDGES = [0, 10, 50, 501]                       # Hz, as in the 2f spectral table
LEVELS = (0.50, 0.80, 0.90, 0.95)
H_QQ = (10, 100, 500)
ACF_LAGS = 300
RHO_COARSE = (0.80, 0.85, 0.90, 0.93, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99,
              0.993, 0.996, 0.998)
CHUNK = 8192
SEED_FIT, SEED_TEST, SEED_FIG = 0, 1, 2

# Reference palette, categorical slots 1-3 in fixed order, one per entity in every figure
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_BIAS, C_2F, C_MEAN = "#2a78d6", "#eb6834", "#1baf7a"
SEQ_BLUES = ("#86b6ef", "#2a78d6", "#104281")          # ordinal ramp: h = 10, 100, 500
RC = {"font.size": 9, "axes.titlesize": 10, "axes.titlecolor": INK, "axes.labelcolor": INK2,
      "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False, "savefig.dpi": 150,
      "savefig.bbox": "tight"}


# ─────────────────────────────── predict ───────────────────────────────

def load_model(run: str):
    cfg = json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))
    m = cfg["model"]
    model = PatchTST(n_features=4, seq_len=cfg["seq"], horizon=cfg["horizon"],
                     patch_len=cfg["patch_len"], stride=cfg["patch_stride"],
                     d_model=m["d_model"], n_heads=m["n_heads"], d_ff=m["d_ff"],
                     n_layers=m["n_layers"], dropout=m["dropout"],
                     align_end=cfg["align"] == "end")
    ck = torch.load(AGENT / cfg["checkpoint"].replace("\\", "/"), map_location="cpu",
                    weights_only=True)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), cfg


def select_windows(data: dict, split: str, every: int):
    """Windows of a split (optionally every k-th per series) and their (key, target_start)."""
    idx = data[f"{split}_idx"]
    keys = np.array([f"{a}/{r}/{g}" for _, a, r, g in data["series_meta"]])
    meta = pd.DataFrame({"key": keys[idx[:, 0]],
                         "target_start": idx[:, 1] + data["seq_length"]})
    keep = (meta.groupby("key").cumcount() % every == 0).to_numpy()
    return idx[keep], meta[keep].reset_index(drop=True)


def run_model(model, series, idx, cfg, mu_w, sg_w, big: Path, name: str):
    H = cfg["horizon"]
    P = np.lib.format.open_memmap(big / f"{name}_pred.npy", mode="w+", dtype=np.float32,
                                  shape=(len(idx), H))
    Y = np.lib.format.open_memmap(big / f"{name}_true.npy", mode="w+", dtype=np.float32,
                                  shape=(len(idx), H))
    i, t0 = 0, time.time()
    with torch.no_grad():
        for xb, yb in WindowLoader(series, idx, cfg["seq"], H, 512):
            k = len(xb)
            P[i:i + k] = (model(xb).double() * sg_w + mu_w).numpy()
            Y[i:i + k] = (yb.double() * sg_w + mu_w).numpy()
            i += k
    P.flush()
    Y.flush()
    print(f"  {name}: {len(idx)} windows in {time.time() - t0:.0f} s")
    return P, Y


def check_rmse(P, Y, run: str) -> None:
    """The cached predictions must reproduce the run's stored test RMSE curve."""
    se = np.zeros(P.shape[1])
    for s in range(0, len(P), CHUNK):
        se += ((P[s:s + CHUNK].astype(np.float64) - Y[s:s + CHUNK]) ** 2).sum(0)
    stored = np.load(RUNS / run / "rmse_curve.npy").ravel()
    diff = float(np.abs(np.sqrt(se / len(P)) - stored).max())
    print(f"  {run}: max |RMSE(h) - stored curve| = {diff:.1e}")
    if diff > 1e-4:
        raise RuntimeError(f"{run} does not reproduce its stored test RMSE curve")


def stage_predict(big: Path, every: int) -> None:
    big.mkdir(parents=True, exist_ok=True)
    print(f"Predict: torch {torch.__version__}, {torch.get_num_threads()} CPU threads, "
          f"every {every} window(s) per series")
    jobs = [(RUN_BEST, "4a", {}, ("val", "test")),
            (RUN_PSD, "2f", dict(alphas=ROUND2_ALPHAS, ratios=ROUND2_RATIOS), ("test",))]
    for run, tag, scope_kw, splits in jobs:
        model, cfg = load_model(run)
        data = load_all_buildings_ctx(seq_length=cfg["seq"], step=cfg["step"],
                                      horizon=cfg["horizon"], test_step=cfg["test_step"],
                                      **scope_kw)
        if not (np.allclose(data["mu"], cfg["mu"]) and np.allclose(data["sigma"], cfg["sigma"])):
            raise RuntimeError(f"{run}: normalisation differs from its config.json")
        mu_w, sg_w = float(data["mu"][0]), float(data["sigma"][0])
        series = torch.from_numpy(data["series"])
        for split in splits:
            idx, meta = select_windows(data, split, every)
            meta.to_csv(big / f"{tag}_{split}_meta.csv", index=False)
            P, Y = run_model(model, series, idx, cfg, mu_w, sg_w, big, f"{tag}_{split}")
            if every == 1 and split == "test":
                check_rmse(P, Y, run)
            del P, Y
        del series, data, model
    (big / "predict_done.json").write_text(json.dumps({"every": every,
                                                       "finished": time.ctime()}))


# ─────────────────────────────── analyze ───────────────────────────────

def hann(H: int) -> np.ndarray:
    """Periodic Hann window, identical to torch.hann_window(H) in train_patchtst_v2.py."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(H) / H)


def periodogram_sum(x: np.ndarray, win: np.ndarray) -> np.ndarray:
    """Sum over windows of |rfft(hann * (x - window mean))|^2, as in evaluate()."""
    d = x - x.mean(1, keepdims=True)
    return (np.abs(np.fft.rfft(win * d, axis=1)) ** 2).sum(0)


def simulate_z(n: int, H: int, rho: float, rng) -> np.ndarray:
    """n stationary AR(1) paths with unit variance: z_1 = eps_1, z_h = rho z_(h-1) + c eps_h."""
    eps = rng.standard_normal((n, H))
    c = np.sqrt(1.0 - rho ** 2)
    eps[:, 0] /= c
    return lfilter([c], [1.0, -rho], eps, axis=1)


def crps_gauss(y, m, s):
    """CRPS of N(m, s^2) at y (closed form, Gneiting & Raftery 2007)."""
    w = (y - m) / s
    return s * (w * (2.0 * ndtr(w) - 1.0) + 2.0 * np.exp(-0.5 * w * w) / np.sqrt(2 * np.pi)
                - 1.0 / np.sqrt(np.pi))


def band_db(P, T, f, edges) -> np.ndarray:
    """10 log10 of the forecast/true power in each band (DC excluded)."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (f > 0) & (f >= lo) & (f < hi)
        out.append(10 * np.log10(P[b].sum() / T[b].sum()))
    return np.array(out)


def spectral_cols(P, T, f) -> dict:
    m = f > 0
    db = 10 * np.log10(P[m] / T[m])
    out = {"power_ratio": float(P[m].sum() / T[m].sum())}
    for (lo, hi), v in zip(zip(COARSE_EDGES[:-1], COARSE_EDGES[1:]), band_db(P, T, f, COARSE_EDGES)):
        out[f"dB_{lo}_{hi}Hz"] = float(v)
    out["LSD_below_50Hz_dB"] = float(np.sqrt(np.mean(db[f[m] < 50] ** 2)))
    return out


def series_stats() -> pd.DataFrame:
    """Per series, train part only: climatology (mean, std) and the exponential fit of the
    autocorrelation, rho_tau = exp(-1/tau) with tau the integral time scale."""
    rows = []
    for a, r, g in list_series():
        w = load_faces(a, r, g)[:, 0].astype(np.float64)
        x = w[:int(len(w) * TRAIN_R)]
        d = x - x.mean()
        F = np.fft.rfft(d, 2 * len(d))
        acf = np.fft.irfft(F * np.conj(F))[:len(d)]
        acf /= acf[0]
        k0 = int(np.argmax(acf < 0)) if (acf < 0).any() else len(d)
        tau = float(acf[:k0].sum())                            # samples = ms at 1 kHz
        rows.append({"key": f"{a}/{r}/{g}", "mu_s": x.mean(), "sigma_s": x.std(),
                     "tau_ms": tau, "rho_tau": float(np.exp(-1.0 / tau)),
                     "lag1": float(acf[1])})
    return pd.DataFrame(rows).set_index("key")


def stage_analyze(out: Path, big: Path, every: int) -> None:
    t0 = time.time()

    def mm(name):
        return np.load(big / f"{name}.npy", mmap_mode="r")

    Pv, Yv, Pt, Yt, P2, Y2 = (mm(n) for n in ("4a_val_pred", "4a_val_true", "4a_test_pred",
                                              "4a_test_true", "2f_test_pred", "2f_test_true"))
    meta_t = pd.read_csv(big / "4a_test_meta.csv")
    meta_2 = pd.read_csv(big / "2f_test_meta.csv")
    nv, nt, H = len(Pv), len(Pt), Pv.shape[1]
    f = np.fft.rfftfreq(H, 1 / FS)
    win = hann(H)
    stats = series_stats()
    print(f"Analyze: {nv} val / {nt} test / {len(P2)} 2f windows, H = {H}")

    # 1. Validation errors: mu(h), sigma(h), normality, autocorrelation
    s1, s2 = np.zeros(H), np.zeros(H)
    for s in range(0, nv, CHUNK):
        d = Yv[s:s + CHUNK].astype(np.float64) - Pv[s:s + CHUNK]
        s1 += d.sum(0)
        s2 += (d * d).sum(0)
    mu_h = s1 / nv
    sigma_h = np.sqrt(s2 / nv - mu_h ** 2)
    lag1, acf_end = np.zeros(H - 1), np.zeros(ACF_LAGS + 1)
    qq = {h: [] for h in H_QQ}
    psd_true_v = np.zeros(len(f))
    lags = H - 1 - np.arange(ACF_LAGS + 1)
    for s in range(0, nv, CHUNK):
        y = Yv[s:s + CHUNK].astype(np.float64)
        z = (y - Pv[s:s + CHUNK] - mu_h) / sigma_h
        lag1 += (z[:, 1:] * z[:, :-1]).sum(0)
        acf_end += (z[:, [H - 1]] * z[:, lags]).sum(0)
        for h in H_QQ:
            qq[h].append(z[:, h - 1].copy())
        psd_true_v += periodogram_sum(y, win)
    lag1 /= nv
    acf_end /= nv
    psd_true_v /= nv
    qq = {h: np.concatenate(v) for h, v in qq.items()}
    normality = {h: {"skew": float(skew(v)), "kurtosis": float(kurtosis(v, fisher=False))}
                 for h, v in qq.items()}

    # 2. rho from the PSD constraint (octave bands, validation windows, common random numbers)
    def traj_psd(P, rows, rho, seed):
        rng, acc = np.random.default_rng(seed), np.zeros(len(f))
        for s in range(0, len(rows), CHUNK):
            r = rows[s:s + CHUNK]
            traj = P[r].astype(np.float64) + mu_h + sigma_h * simulate_z(len(r), H, rho, rng)
            acc += periodogram_sum(traj, win)
        return acc / len(rows)

    search = []

    def try_rho(rho):
        db = band_db(traj_psd(Pv, np.arange(nv), rho, SEED_FIT), psd_true_v, f, OCTAVE_EDGES)
        search.append((rho, float(np.sqrt(np.mean(db ** 2))), *db))

    for rho in RHO_COARSE:
        try_rho(rho)
    best = min(search, key=lambda t: t[1])[0]
    for rho in np.arange(best - 0.004, best + 0.0045, 0.001):
        rho = round(float(rho), 4)
        if 0 < rho < 1 and min(abs(rho - t[0]) for t in search) > 1e-6:
            try_rho(rho)
    search.sort()
    rho = min(search, key=lambda t: t[1])[0]
    print(f"  rho (PSD constraint, validation) = {rho}  [{time.time() - t0:.0f} s]")

    # 3. Test: mean forecast, one trajectory per window, coverage, CRPS, spectra
    def evaluate(P, Y, rows, keys, seed, window_rmse=False):
        cm_all = stats.loc[keys, "mu_s"].to_numpy()
        cs_all = stats.loc[keys, "sigma_s"].to_numpy()
        rng, acc, wr = np.random.default_rng(seed), defaultdict(float), []
        qs = {lv: ndtri(0.5 + lv / 2) for lv in LEVELS}
        for s in range(0, len(rows), CHUNK):
            r = rows[s:s + CHUNK]
            p, y = P[r].astype(np.float64), Y[r].astype(np.float64)
            m = p + mu_h
            cm, cs = cm_all[s:s + CHUNK, None], cs_all[s:s + CHUNK, None]
            traj = m + sigma_h * simulate_z(len(r), H, rho, rng)
            acc["se_4a"] += ((p - y) ** 2).sum(0)
            acc["se_mean"] += ((m - y) ** 2).sum(0)
            acc["ae_mean"] += np.abs(m - y).sum(0)
            acc["se_traj"] += ((traj - y) ** 2).sum(0)
            acc["se_clim"] += ((cm - y) ** 2).sum(0)
            acc["crps"] += crps_gauss(y, m, sigma_h).sum(0)
            acc["crps_clim"] += crps_gauss(y, cm, cs).sum(0)
            w = np.abs(y - m) / sigma_h
            for lv, q in qs.items():
                acc[f"cov{round(lv * 100)}"] += (w <= q).sum(0)
            acc["psd_true"] += periodogram_sum(y, win)
            acc["psd_mean"] += periodogram_sum(m, win)
            acc["psd_traj"] += periodogram_sum(traj, win)
            if window_rmse:
                wr.append(np.sqrt(((p - y) ** 2).mean(1)))
        res = {k: v / len(rows) for k, v in acc.items()}
        for k in ("se_4a", "se_mean", "se_traj", "se_clim"):
            res["rmse" + k[2:]] = np.sqrt(res.pop(k))
        if window_rmse:
            res["window_rmse"] = np.concatenate(wr)
        return res

    ev3 = evaluate(Pt, Yt, np.arange(nt), meta_t["key"].to_numpy(), SEED_TEST)
    print(f"  test R3 scored  [{time.time() - t0:.0f} s]")

    # 4. R2: the same windows as run 2f (deterministic -> its CRPS is its MAE)
    j = (meta_2.reset_index().rename(columns={"index": "row_2f"})
         .merge(meta_t.reset_index().rename(columns={"index": "row_4a"}),
                on=["key", "target_start"], how="left"))
    if j["row_4a"].isna().any():
        raise RuntimeError("some 2f test windows are not among the 4a test windows")
    rows_r2 = j["row_4a"].to_numpy(int)
    ev2 = evaluate(Pt, Yt, rows_r2, j["key"].to_numpy(), SEED_TEST, window_rmse=True)
    se2, ae2, psd2, ydiff = 0.0, 0.0, 0.0, 0.0
    for s in range(0, len(j), CHUNK):
        p, y = P2[s:s + CHUNK].astype(np.float64), Y2[s:s + CHUNK].astype(np.float64)
        ydiff = max(ydiff, float(np.abs(y - Yt[rows_r2[s:s + CHUNK]]).max()))
        se2 = se2 + ((p - y) ** 2).sum(0)
        ae2 = ae2 + np.abs(p - y).sum(0)
        psd2 = psd2 + periodogram_sum(p, win)
    if ydiff > 1e-4:
        raise RuntimeError(f"2f and 4a targets differ on the shared windows ({ydiff:.1e})")
    rmse_2f, mae_2f, psd_2f = np.sqrt(se2 / len(j)), ae2 / len(j), psd2 / len(j)
    if every == 1:
        z = np.load(RUNS / RUN_PSD / "psd.npz")
        print(f"  2f spectra vs stored psd.npz: max rel. diff pred "
              f"{np.max(np.abs(psd_2f / z['test_pred'] - 1)):.1e}, true "
              f"{np.max(np.abs(ev2['psd_true'] / z['test_true'] - 1)):.1e}")
    print(f"  R2 vs 2f scored  [{time.time() - t0:.0f} s]")

    # 5. Tables
    out.mkdir(parents=True, exist_ok=True)
    octave_cols = [f"dB_{lo}_{hi}Hz" for lo, hi in zip(OCTAVE_EDGES[:-1], OCTAVE_EDGES[1:])]
    pd.DataFrame(search, columns=["rho", "rms_octave_dB", *octave_cols]) \
        .to_csv(out / "rho_search.csv", index=False)
    by_h = pd.DataFrame({
        "h": np.arange(1, H + 1), "mu_val": mu_h, "sigma_val": sigma_h,
        "lag1_corr_val": np.r_[np.nan, lag1],
        "rmse_4a": ev3["rmse_4a"], "rmse_mean": ev3["rmse_mean"],
        "rmse_one_trajectory": ev3["rmse_traj"], "rmse_climatology": ev3["rmse_clim"],
        "crps": ev3["crps"], "crps_climatology": ev3["crps_clim"], "mae_mean": ev3["ae_mean"],
        **{f"coverage_{round(lv * 100)}": ev3[f"cov{round(lv * 100)}"] for lv in LEVELS}})
    by_h.to_csv(out / "metrics_by_h_test_r3.csv", index=False)

    def rows_for(ev, extra=None):
        mean_row = {"forecast": "4a mean (point forecast)",
                    "rmse_mean_h": ev["rmse_mean"].mean(), "rmse_h500": ev["rmse_mean"][-1],
                    "crps_mean_h": ev["ae_mean"].mean(),
                    **spectral_cols(ev["psd_mean"], ev["psd_true"], f)}
        bias_row = {"forecast": "4a + Gaussian AR(1) bias",
                    "rmse_mean_h": ev["rmse_mean"].mean(), "rmse_h500": ev["rmse_mean"][-1],
                    "rmse_one_trajectory_mean_h": ev["rmse_traj"].mean(),
                    "crps_mean_h": ev["crps"].mean(),
                    **{f"coverage_{round(lv * 100)}": ev[f"cov{round(lv * 100)}"].mean()
                       for lv in LEVELS},
                    **spectral_cols(ev["psd_traj"], ev["psd_true"], f)}
        clim_row = {"forecast": "climatology N(mu_s, sigma_s^2)",
                    "rmse_mean_h": ev["rmse_clim"].mean(), "rmse_h500": ev["rmse_clim"][-1],
                    "crps_mean_h": ev["crps_clim"].mean(), "power_ratio": 0.0}
        return pd.DataFrame([bias_row, mean_row] + ([extra] if extra else []) + [clim_row])

    row_2f = {"forecast": "2f: PSD in the loss (lambda_psd 0.1)",
              "rmse_mean_h": rmse_2f.mean(), "rmse_h500": rmse_2f[-1],
              "crps_mean_h": mae_2f.mean(), **spectral_cols(psd_2f, ev2["psd_true"], f)}
    tab3, tab2 = rows_for(ev3), rows_for(ev2, row_2f)
    tab3.to_csv(out / "comparison_test_r3.csv", index=False)
    tab2.to_csv(out / "comparison_test_r2_vs_2f.csv", index=False)

    worse = np.nonzero(ev3["crps"] > ev3["crps_clim"])[0]
    summary = {
        "windows": {"val": nv, "test_r3": nt, "test_r2": len(j)},
        "every": every,
        "rho_psd_constraint": rho,
        "rho_lag1_errors_h500": float(lag1[-1]),
        "rho_tau_train_series_median": float(stats["rho_tau"].median()),
        "tau_ms_train_series_median": float(stats["tau_ms"].median()),
        "mu_val_abs_max": float(np.abs(mu_h).max()),
        "sigma_val_h1_h500": [float(sigma_h[0]), float(sigma_h[-1])],
        "normality_val": {str(h): v for h, v in normality.items()},
        "coverage_test_r3": {str(round(lv * 100)): float(ev3[f"cov{round(lv * 100)}"].mean())
                             for lv in LEVELS},
        "crps_test_r3": {"4a_bias": float(ev3["crps"].mean()),
                         "climatology": float(ev3["crps_clim"].mean()),
                         "first_h_worse_than_climatology": int(worse[0] + 1) if len(worse) else None},
        "rmse_test_r3": {"4a": float(ev3["rmse_4a"].mean()), "mean": float(ev3["rmse_mean"].mean()),
                         "one_trajectory": float(ev3["rmse_traj"].mean()),
                         "climatology": float(ev3["rmse_clim"].mean())},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 6. Figures
    plt.rcParams.update(RC)
    fig_psd(out / "psd_4a_bias_vs_2f.png", f, ev2, psd_2f, len(j))
    fig_diagnostics(out / "diagnostics.png", qq, normality, acf_end, rho, lag1[-1], ev3)
    fig_examples(out / "forecast_examples.png", j, ev2["window_rmse"], rows_r2, Pt, Yt, P2,
                 mu_h, sigma_h, rho)

    pd.set_option("display.width", 220)
    print(f"\nTest R3 ({nt} windows):")
    print(tab3.round(4).to_string(index=False))
    print(f"\nTest R2, the {len(j)} windows of run 2f:")
    print(tab2.round(4).to_string(index=False))
    print(f"\nrho: PSD constraint {rho} | lag-1 of the errors at h=500 {lag1[-1]:.4f} | "
          f"train-series exp(-1/tau) median {stats['rho_tau'].median():.4f}")
    print("normality (val):", {h: {k: round(v, 3) for k, v in d.items()} for h, d in normality.items()})
    print(f"done in {time.time() - t0:.0f} s -> {out}")


def stage_gallery(out: Path, big: Path, angle: int, pct: float) -> None:
    """One figure per terrain roughness (alpha) with one panel per building (B:D:H), all at
    the same wind angle: the test window whose 4a error (RMSE over h = 1..500) is at the
    `pct` percentile of that building's windows. Uses the layer fitted by stage_analyze.
    The CSV lists the windows shown and, per building, the median window RMSE against the
    std of the series' train part (sigma_s), to compare buildings on a common scale."""
    if not (out / "summary.json").exists():
        raise FileNotFoundError(f"run --stage analyze first (no {out / 'summary.json'})")
    by_h = pd.read_csv(out / "metrics_by_h_test_r3.csv")
    mu_h, sigma_h = by_h["mu_val"].to_numpy(), by_h["sigma_val"].to_numpy()
    rho = json.loads((out / "summary.json").read_text(encoding="utf-8"))["rho_psd_constraint"]
    Pt, Yt = (np.load(big / f"4a_test_{n}.npy", mmap_mode="r") for n in ("pred", "true"))
    meta = pd.read_csv(big / "4a_test_meta.csv")
    keys = meta["key"].to_numpy()
    H, back = len(mu_h), 300
    rng = np.random.default_rng(SEED_FIG)
    panels, table = {}, []                               # "Alpha1_4/2_1_3" -> window data
    print(f"Gallery: wind angle {angle} deg, {_ordinal(pct)} 4a error per building")
    for geo in meta["key"].str.rsplit("/", n=1).str[0].unique():
        rows = np.flatnonzero(keys == f"{geo}/{angle}")
        if not len(rows):
            continue
        err = np.sqrt(((Pt[rows].astype(np.float64) - Yt[rows]) ** 2).mean(1))
        k = np.argsort(err, kind="stable")[int(pct / 100 * (len(rows) - 1))]
        i, ts = rows[k], int(meta.loc[rows[k], "target_start"])
        w = _load_windward(f"{geo}/{angle}")
        hist, truth = _cut_window(w, ts, back, H, Yt[i])
        m = Pt[i].astype(np.float64) + mu_h
        panels[geo] = (hist, truth, m, m + sigma_h * simulate_z(2, H, rho, rng))
        sigma_s = float(w[:int(len(w) * TRAIN_R)].std())
        table.append({"alpha": geo.split("/")[0], "ratio_BDH": geo.split("/")[1],
                      "angle": angle, "target_start": ts, "window_rmse": err[k],
                      "building_median_rmse": np.median(err), "sigma_s": sigma_s,
                      "median_rmse_over_sigma_s": np.median(err) / sigma_s})
        print(f"  {geo:16s} target_start {ts:6d}  window RMSE {err[k]:.4f}  building median "
              f"{np.median(err):.4f}  sigma_s {sigma_s:.4f}")
    if not panels:
        raise ValueError(f"no test windows at wind angle {angle} deg")
    suffix = "" if pct == 50 else f"_p{pct:g}"
    table = pd.DataFrame(table)
    table.to_csv(out / f"forecast_gallery_{angle}deg{suffix}.csv", index=False)
    print(f"  corr(building median RMSE, sigma_s) = "
          f"{np.corrcoef(table['building_median_rmse'], table['sigma_s'])[0, 1]:.2f}; "
          f"median RMSE / sigma_s by alpha: "
          f"{table.groupby('alpha')['median_rmse_over_sigma_s'].mean().round(3).to_dict()}")
    lo = min(min(h.min(), t.min(), (m - 1.96 * sigma_h).min(), tr.min())
             for h, t, m, tr in panels.values())
    hi = max(max(h.max(), t.max(), (m + 1.96 * sigma_h).max(), tr.max())
             for h, t, m, tr in panels.values())
    pad = 0.03 * (hi - lo)
    plt.rcParams.update(RC)
    for alpha in sorted({g.split("/")[0] for g in panels}):
        mine = {g.split("/")[1]: v for g, v in panels.items() if g.startswith(alpha + "/")}
        path = out / f"forecast_gallery_{alpha.lower()}_{angle}deg{suffix}.png"
        fig_gallery(path, alpha, mine, sigma_h, angle, pct, (lo - pad, hi + pad))
        print(f"  -> {path.name}")


# ─────────────────────────────── figures ───────────────────────────────

def _style(ax, grid_minor=False):
    ax.grid(True, which="major", color=GRID, lw=0.6)
    if grid_minor:
        ax.grid(True, which="minor", color=GRID, lw=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)


def fig_psd(path, f, ev2, psd_2f, n_windows):
    m = f > 0
    pr = {k: v[m].sum() / ev2["psd_true"][m].sum()
          for k, v in (("traj", ev2["psd_traj"]), ("2f", psd_2f), ("mean", ev2["psd_mean"]))}
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.loglog(f[m], ev2["psd_true"][m], color=INK, lw=1.8, label="truth", zorder=5)
    ax.loglog(f[m], ev2["psd_traj"][m], color=C_BIAS, lw=1.3, zorder=4,
              label=f"4a + Gaussian AR(1) bias, one trajectory per window (power {pr['traj']:.2f})")
    ax.loglog(f[m], psd_2f[m], color=C_2F, lw=1.3, zorder=3,
              label=f"2f: PSD in the loss, λ_psd = 0.1 (power {pr['2f']:.2f})")
    ax.loglog(f[m], ev2["psd_mean"][m], color=C_MEAN, lw=1.3, zorder=2,
              label=f"4a mean forecast (power {pr['mean']:.2f})")
    k = int(np.argmin(np.abs(f - 30)))
    ax.annotate("4a mean", (f[k], ev2["psd_mean"][k]), xytext=(0, 9),
                textcoords="offset points", color=INK2, fontsize=8, ha="center")
    _style(ax, grid_minor=True)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("mean periodogram (hann, 500 steps)")
    n_txt = f"{n_windows:,}".replace(",", " ")
    ax.set_title(f"Spectrum of the 500-step forecasts vs. the truth — R2 test, "
                 f"the {n_txt} windows of run 2f", loc="left")
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def fig_diagnostics(path, qq, normality, acf_end, rho, rho_lag1, ev3):
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.6))
    H = len(ev3["crps"])
    h = np.arange(1, H + 1)

    ax = axs[0, 0]
    probs = (np.arange(1, 201) - 0.5) / 200
    theo = ndtri(probs)
    ax.plot([-4, 4], [-4, 4], color=INK2, ls="--", lw=1, label="normal")
    for hq, col in zip(H_QQ, SEQ_BLUES):
        nm = normality[hq]
        ax.plot(theo, np.quantile(qq[hq], probs), "o", ms=4, color=col, mec="white", mew=0.4,
                label=f"h = {hq} ms  (skew {nm['skew']:+.2f}, kurtosis {nm['kurtosis']:.2f})")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel("N(0, 1) quantile")
    ax.set_ylabel("standardised error quantile")
    ax.set_title("(a) Normal distribution: QQ plot of the validation errors", loc="left")
    ax.legend(loc="upper left", fontsize=8)

    ax = axs[0, 1]
    k = np.arange(len(acf_end))
    ax.axhline(0, color=INK2, lw=0.6)
    ax.plot(k, acf_end, color=C_BIAS, lw=1.6, label="errors at h = 500 ms (validation)")
    ax.plot(k, rho ** k, color=INK, ls="--", lw=1.2, label=f"ρ^k, ρ = {rho:.3f} (PSD constraint)")
    ax.plot(k, rho_lag1 ** k, color=INK2, ls=":", lw=1.2,
            label=f"ρ^k, ρ = {rho_lag1:.3f} (lag-1 correlation of the errors)")
    ax.set_xlabel("lag k [ms]")
    ax.set_ylabel("correlation")
    ax.set_title("(b) Exponential term: error autocorrelation vs. ρ^k", loc="left")
    ax.legend(loc="upper right", fontsize=8)

    ax = axs[1, 0]
    for lv, col in ((0.50, SEQ_BLUES[0]), (0.90, SEQ_BLUES[2])):
        c = ev3[f"cov{round(lv * 100)}"]
        ax.axhline(lv, color=INK2, ls="--", lw=1)
        ax.plot(h, c, color=col, lw=1.6, label=f"{lv:.0%} interval: mean coverage {c.mean():.1%}")
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("horizon h [ms]")
    ax.set_ylabel("share of true values inside")
    ax.set_title("(c) Interval coverage on test (dashed = nominal)", loc="left")
    ax.legend(loc="lower right", fontsize=8)

    ax = axs[1, 1]
    ax.plot(h, ev3["crps"], color=C_BIAS, lw=1.6,
            label=f"4a + bias (mean {ev3['crps'].mean():.4f})")
    ax.plot(h, ev3["ae_mean"], color=C_MEAN, lw=1.3,
            label=f"4a point forecast, CRPS = MAE (mean {ev3['ae_mean'].mean():.4f})")
    ax.plot(h, ev3["crps_clim"], color=INK2, ls="--", lw=1.2,
            label=f"climatology N(μ_s, σ_s²) (mean {ev3['crps_clim'].mean():.4f})")
    ax.set_xlabel("horizon h [ms]")
    ax.set_ylabel("CRPS [Cp]")
    ax.set_title("(d) CRPS on test, lower is better", loc="left")
    ax.legend(loc="lower right", fontsize=8)

    for ax in axs.flat:
        _style(ax)
    fig.suptitle("Gaussian AR(1) bias on 4a — R3 (fitted on validation, scored on test)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_examples(path, j, window_rmse, rows_r2, Pt, Yt, P2, mu_h, sigma_h, rho):
    order = np.argsort(window_rmse)
    picks = [(order[int(0.5 * (len(order) - 1))], "median 4a error"),
             (order[int(0.9 * (len(order) - 1))], "90th-percentile 4a error")]
    H, back = len(mu_h), 300
    rng = np.random.default_rng(SEED_FIG)
    fig, axs = plt.subplots(len(picks), 1, figsize=(10, 6.6), sharex=True)
    for ax, (i, what) in zip(axs, picks):
        key, ts = j.loc[i, "key"], int(j.loc[i, "target_start"])
        hist, truth = _cut_window(_load_windward(key), ts, back, H, Yt[rows_r2[i]])
        m = Pt[rows_r2[i]].astype(np.float64) + mu_h
        _draw_forecast(ax, hist, truth, m, sigma_h, m + sigma_h * simulate_z(2, H, rho, rng),
                       other=P2[i])
        ax.set_title(f"{key.replace('/', '  ')}°  —  window with the {what}", loc="left")
        ax.set_ylabel("windward Cp")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    axs[-1].set_xlabel("time from the forecast origin [ms]")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_gallery(path, alpha, panels, sigma_h, angle, pct, ylim):
    """Rows: building height H, columns: plan B:D. `panels` maps "2_1_3" -> window data."""
    plans = sorted({r.rsplit("_", 1)[0] for r in panels})
    heights = sorted({int(r.rsplit("_", 1)[1]) for r in panels})
    nr, nc = len(heights), len(plans)
    fig, axs = plt.subplots(nr, nc, figsize=(4.3 * nc, 2.2 * nr + 1.0), sharex=True,
                            sharey=True, squeeze=False)
    first = None
    for ri, height in enumerate(heights):
        for ci, plan in enumerate(plans):
            ax, r = axs[ri, ci], f"{plan}_{height}"
            if r not in panels:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "not measured", transform=ax.transAxes, ha="center",
                        color=INK2)
                continue
            hist, truth, m, traj = panels[r]
            _draw_forecast(ax, hist, truth, m, sigma_h, traj)
            ax.set_title(f"B:D:H = {r.replace('_', ':')}", loc="left")
            if first is None:
                first = ax
        axs[ri, 0].set_ylabel("windward Cp")
    for ax in axs[-1]:
        ax.set_xlabel("time from the forecast origin [ms]")
    axs[0, 0].set_ylim(*ylim)
    axs[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]))
    handles, labels = first.get_legend_handles_labels()
    ncol = len(labels) if nc >= 3 else 3
    fig.legend(handles, labels, loc="lower center", ncol=ncol, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Terrain roughness α = {_alpha_label(alpha)}, wind angle {angle}°: "
                 f"4a + Gaussian AR(1) bias\nEach panel is one building, showing its test "
                 f"window with the {_ordinal(pct)} 4a error", x=0.01, ha="left", color=INK,
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.03 * -(-len(labels) // ncol), 1, 1))
    fig.savefig(path)
    plt.close(fig)


def _draw_forecast(ax, hist, truth, m, sigma_h, traj, other=None):
    """History, truth, 4a mean, the bias layer's 50/95 % bands and sampled trajectories, and
    optionally the 2f forecast, for one window (time 0 = forecast origin, 1 sample = 1 ms)."""
    hh = np.arange(1, len(m) + 1)
    ax.fill_between(hh, m - 1.96 * sigma_h, m + 1.96 * sigma_h, color=C_BIAS, alpha=0.12,
                    lw=0, label="4a + bias: 95 % band")
    ax.fill_between(hh, m - 0.674 * sigma_h, m + 0.674 * sigma_h, color=C_BIAS, alpha=0.25,
                    lw=0, label="4a + bias: 50 % band")
    for n_traj, t in enumerate(traj):
        ax.plot(hh, t, color=C_BIAS, lw=0.9,
                label="4a + bias: sampled trajectories" if n_traj == 0 else None)
    ax.plot(np.arange(-len(hist) + 1, 1), hist, color=INK, lw=1.0)
    ax.plot(hh, truth, color=INK, lw=1.4, label="truth")
    if other is not None:
        ax.plot(hh, other, color=C_2F, lw=1.3, label="2f: PSD in the loss")
    ax.plot(hh, m, color=C_MEAN, lw=1.8, label="4a mean forecast")
    ax.axvline(0, color=INK2, ls=":", lw=1)
    _style(ax)


def _load_windward(key):
    a, r, g = key.split("/")
    return load_faces(a, r, int(g))[:, 0].astype(np.float64)


def _cut_window(w, ts, back, H, cached):
    """History (last `back` samples) and truth of one window, checked against the cache."""
    truth = w[ts:ts + H]
    if np.abs(truth - cached).max() > 1e-4:
        raise RuntimeError(f"window at {ts} does not match the cached targets")
    return w[ts - back:ts], truth


def _alpha_label(alpha):
    return alpha.replace("Alpha", "").replace("_", "/")          # "Alpha1_4" -> "1/4"


def _ordinal(pct):
    if pct == 50:
        return "median"
    n = f"{pct:g}"
    end = "th" if n.endswith(("11", "12", "13")) else {"1": "st", "2": "nd", "3": "rd"}.get(
        n[-1], "th")
    return f"{n}{end}-percentile"


# ─────────────────────────────── main ───────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Gaussian AR(1) bias layer on PatchTST 4a")
    p.add_argument("--stage", choices=["all", "predict", "analyze", "gallery"], default="all")
    p.add_argument("--every", type=int, default=1,
                   help="keep every k-th window per series (quick end-to-end check)")
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--angle", type=int, default=0, help="gallery: wind angle [deg]")
    p.add_argument("--pct", type=float, default=50.0,
                   help="gallery: percentile of the 4a window error shown (50 = median)")
    args = p.parse_args()
    big = args.out / "big"
    cached = (big / "predict_done.json").exists() and \
        json.loads((big / "predict_done.json").read_text())["every"] == args.every
    if args.stage == "predict" or (args.stage == "all" and not cached):
        stage_predict(big, args.every)
    if args.stage in ("all", "analyze"):
        stage_analyze(args.out, big, args.every)
    if args.stage in ("all", "gallery"):
        stage_gallery(args.out, big, args.angle, args.pct)


if __name__ == "__main__":
    main()

"""
StatLoss — MSE plus spectral and variance constraints for direct forecasts
(GPU plan, Handoffs/2026-10-21, section 4.1).

MSE-trained direct multi-step forecasts collapse towards the conditional mean
(PatchTST R3: within-window std 0.059 vs 0.291 for the truth, ~63x less
spectral power). StatLoss adds

    lam_psd * mean_band (log P_pred - log P_true)^2   batch-averaged periodogram
    lam_std * mean_window (std_pred - std_true)^2

on the normalised direct output (B, H), fs = 1000 Hz. The PSD term compares
batch-averaged spectra (one 500-sample periodogram is too noisy); the std term
stops the energy from piling up in a few windows. Octave bands by default:
with linear bins, 90 % of the bins sit above 50 Hz where only ~1.5 % of the
energy is. Calibration on the collapsed R3 predictions: lam_psd* = 0.0031
makes the octave term equal the MSE, lam_std* ~ 2.1.

detemplate (default): both statistics are computed on the within-window
deviations MINUS their batch-mean shape. Without it (protocol A of the plan)
the model met the constraint with an input-independent template — the same
dip-and-jump at h~240 in every window, 17 % of its variance at lam_psd=0.03 —
which raises the batch power for free. With it, a shape shared by all windows
adds nothing to either statistic, so only input-dependent variability counts.

Build it with ``StatLoss(...).to(DEVICE)`` so the buffers live on the GPU.
"""
import numpy as np
import torch
import torch.nn.functional as F

OCTAVE_EDGES = [2, 4, 8, 16, 32, 64, 128, 256, 501]   # Hz


class StatLoss(torch.nn.Module):
    """MSE + lam_psd * log-PSD in octave bands (batch level)
             + lam_std * (std_pred - std_true)^2 per window.
    Works in normalised space on the direct output (B, H); fs = 1000 Hz.
    bins="linear" uses every rfft bin as its own band (plan run 2e);
    detemplate=False is the original version (templates allowed)."""

    def __init__(self, H=500, lam_psd=0.0, lam_std=0.0, eps=1e-8, fs=1000.0,
                 bins="octave", detemplate=True):
        super().__init__()
        self.lam_psd, self.lam_std, self.eps = lam_psd, lam_std, eps
        self.detemplate = detemplate
        self.register_buffer("win", torch.hann_window(H))
        f = np.fft.rfftfreq(H, 1 / fs)[1:]                      # no DC
        if bins == "octave":
            band = torch.as_tensor(np.digitize(f, OCTAVE_EDGES) - 1)
        elif bins == "linear":
            band = torch.arange(len(f))
        else:
            raise ValueError(f"bins must be 'octave' or 'linear', got {bins!r}")
        self.register_buffer("M", F.one_hot(band).float())     # (bins, bands)

    def _dev(self, x):
        """Within-window deviations, minus the batch-mean shape if detemplate."""
        d = x - x.mean(-1, keepdim=True)
        if self.detemplate and x.shape[0] > 1:
            d = d - d.mean(0, keepdim=True)
        return d

    def _psd(self, d):
        P = torch.fft.rfft(self.win * d).abs().pow(2)[:, 1:]
        return P @ self.M                                       # (B, bands)

    def forward(self, pred, true):
        loss = F.mse_loss(pred, true)
        if self.lam_psd == 0 and self.lam_std == 0:
            return loss
        dp, dt = self._dev(pred), self._dev(true)
        if self.lam_psd > 0:
            Pp, Pt = self._psd(dp).mean(0), self._psd(dt).mean(0)
            loss = loss + self.lam_psd * (torch.log(Pp + self.eps)
                                          - torch.log(Pt + self.eps)).pow(2).mean()
        if self.lam_std > 0:
            loss = loss + self.lam_std * (dp.std(-1) - dt.std(-1)).pow(2).mean()
        return loss

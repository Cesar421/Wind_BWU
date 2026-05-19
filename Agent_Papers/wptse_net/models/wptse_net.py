"""
WPTSE-Net — Wind Pressure Time Series Extend Network.

Reference: Tong B., Liang Y., Song J., Hu G., Kareem A. (2024).
"Deep learning-based extension of wind pressure time series."
JWEIA 254, 105909.

Architecture (per Section 2.2.1 + Fig. 3 of the paper):

    Encoder (deterministic, no learnable parameters):
        Per slice of length L, compute (mean, variance, skewness, kurtosis)
        and concatenate one normal-noise scalar (×0.1) → latent vector ∈ R^5.

    Decoder:
        Input  : (B, 5)
        Block × 4 :
            Dense(128) → ReLU
            Dense(128) → ReLU
            BatchNorm1d(128)                    # regularization
            multiplicative translation  (γ)     # learnable per-feature scale
            additive scale              (β)     # learnable per-feature shift
        Final linear FC : Dense(L)              # L = slice length (paper: 32)

    Loss : Huber (δ = 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ────────────────────────────────────────────────────────────────────────────
# Encoder — deterministic statistical-feature extractor + noise injection
# ────────────────────────────────────────────────────────────────────────────

def encode_slice(slices: torch.Tensor, noise_scale: float = 0.1) -> torch.Tensor:
    """Compute (mean, var, skewness, kurtosis, noise) per slice.

    Parameters
    ----------
    slices       : (B, L) tensor of Cp slices.
    noise_scale  : multiplicative factor for the standard-normal noise scalar
                   (paper: 0.1).

    Returns
    -------
    z : (B, 5) latent tensor.
    """
    mu = slices.mean(dim=1)
    var = slices.var(dim=1, unbiased=False)
    centered = slices - mu.unsqueeze(1)
    std = torch.sqrt(var.clamp_min(1e-12))
    m3 = (centered ** 3).mean(dim=1)
    m4 = (centered ** 4).mean(dim=1)
    skew = m3 / std.pow(3).clamp_min(1e-12)
    kurt = m4 / var.clamp_min(1e-12).pow(2)         # raw (non-excess) kurtosis
    noise = torch.randn_like(mu) * noise_scale
    return torch.stack([mu, var, skew, kurt, noise], dim=1)


# ────────────────────────────────────────────────────────────────────────────
# Decoder iteration block
# ────────────────────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """Two Dense+ReLU, BatchNorm, then per-feature multiplicative translation
    and additive scale layers (Dinh et al., 2016 affine coupling style)."""

    def __init__(self, in_features: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        # multiplicative translation layer (learnable γ initialised to 1)
        self.gamma = nn.Parameter(torch.ones(hidden))
        # additive scale layer (learnable β initialised to 0)
        self.beta = nn.Parameter(torch.zeros(hidden))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.bn(x)
        x = x * self.gamma + self.beta
        return x


# ────────────────────────────────────────────────────────────────────────────
# Full WPTSE-Net
# ────────────────────────────────────────────────────────────────────────────

class WPTSENet(nn.Module):
    """WPTSE-Net decoder. Encoder is deterministic (see ``encode_slice``).

    Parameters
    ----------
    slice_length : output dimension (samples per generated slice; paper: 32).
    latent_dim   : encoder output dim (default 5 — paper).
    hidden       : hidden width of dense layers (paper: 128).
    n_blocks     : decoder iteration count (paper: 4).
    """

    def __init__(self, slice_length: int = 32, latent_dim: int = 5,
                 hidden: int = 128, n_blocks: int = 4):
        super().__init__()
        blocks = []
        in_dim = latent_dim
        for _ in range(n_blocks):
            blocks.append(DecoderBlock(in_dim, hidden))
            in_dim = hidden
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, slice_length)
        self.slice_length = slice_length
        self.latent_dim = latent_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z : (B, latent_dim) → (B, slice_length)."""
        h = self.blocks(z)
        return self.head(h)

    @torch.no_grad()
    def predict(self, z: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(z)

    @torch.no_grad()
    def generate_from_slices(self, ref_slices: torch.Tensor,
                             noise_scale: float = 0.1) -> torch.Tensor:
        """Compute encoder stats from reference slices and decode."""
        self.eval()
        z = encode_slice(ref_slices, noise_scale=noise_scale)
        return self.forward(z)

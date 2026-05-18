"""
PatchTST — Time-series Transformer with patch tokenisation.

Reference:
  Nie, Nguyen, Sinthong, Kalagnanam (ICLR 2023)
  "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"

Adapted for the wind-Cp pipeline:
  - Input  : (B, seq_len=100, 4 channels)  channel-mixed (windward + 3 sides)
  - Output : (B, horizon)                   direct multi-step head
  - Patching with overlap (patch_len, stride)
  - Standard pre-norm Transformer encoder
"""
import math
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Slice (B, L, C) into overlapping patches and project to d_model."""

    def __init__(self, n_features, seq_len, patch_len=16, stride=8, d_model=128):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.proj = nn.Linear(patch_len * n_features, d_model)
        # Learnable positional embedding
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):                        # x: (B, L, C)
        # unfold: (B, n_patches, patch_len, C)
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        x = x.permute(0, 1, 3, 2).contiguous()   # (B, n_patches, patch_len, C)
        x = x.flatten(2)                          # (B, n_patches, patch_len*C)
        x = self.proj(x) + self.pos              # (B, n_patches, d_model)
        return x


class TransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder block."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class PatchTST(nn.Module):
    """
    Channel-mixed PatchTST with direct multi-step head.

    Parameters
    ----------
    n_features : int
        Number of input channels (4 for windward + 3 side faces).
    seq_len : int
        Length of the input window (100 for wind-Cp pipeline).
    horizon : int
        Number of future steps to predict in one shot.
    patch_len, stride : int
        Patching geometry.
    d_model, n_heads, d_ff, n_layers, dropout : transformer hyper-params.
    """

    def __init__(
        self,
        n_features=4,
        seq_len=100,
        horizon=1,
        patch_len=16,
        stride=8,
        d_model=128,
        n_heads=4,
        d_ff=256,
        n_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        self.embed = PatchEmbedding(n_features, seq_len, patch_len, stride, d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        n_patches = self.embed.n_patches
        # Flatten all patch tokens and project to the full horizon
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(n_patches * d_model, horizon),
        )

    def forward(self, x):                        # x: (B, L, C)
        x = self.embed(x)                        # (B, P, D)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        out = self.head(x)                       # (B, horizon)
        if out.shape[1] == 1:
            out = out.squeeze(-1)                # match single-step convention
        return out


if __name__ == "__main__":
    # Sanity check
    m = PatchTST(n_features=4, seq_len=100, horizon=500)
    x = torch.randn(8, 100, 4)
    y = m(x)
    n = sum(p.numel() for p in m.parameters())
    print(f"out shape  : {tuple(y.shape)}")
    print(f"params     : {n:,}")

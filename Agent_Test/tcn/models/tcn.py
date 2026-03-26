"""
Temporal Convolutional Network (TCN) for Cp Time-Series Forecasting
====================================================================
Source: Kareem ML/JWEIA 2024 — TCN with causal dilated convolutions.

Architecture:
  TemporalBlock(4→64,   dilation=1)  — residual
  TemporalBlock(64→128,  dilation=2)  — residual
  TemporalBlock(128→256, dilation=4)  — residual
  → last time-step →
  Linear(256→128) → ReLU → Dropout → Linear(128→horizon)
"""

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """Conv1d with left-padding so output length == input length (causal)."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=self.pad, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # trim right padding


class TemporalBlock(nn.Module):
    """Two causal-conv layers + residual connection."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel_size, dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.downsample = (nn.Conv1d(in_ch, out_ch, 1)
                           if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        return self.net(x) + self.downsample(x)


class TCN(nn.Module):
    def __init__(self, n_features=4, num_channels=(64, 128, 256),
                 kernel_size=3, dropout=0.2, fc_hidden=128, horizon=1):
        super().__init__()
        layers = []
        in_ch = n_features
        for i, out_ch in enumerate(num_channels):
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size,
                                        dilation=2 ** i, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, horizon),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, T, F)
        x = x.permute(0, 2, 1)     # (B, F, T)
        x = self.tcn(x)             # (B, C, T)
        x = x[:, :, -1]             # last time step → (B, C)
        return self.fc(x)           # (B, horizon)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

"""
Transformer Encoder for Cp Time-Series Forecasting
====================================================
Source: Kareem ML/JWEIA 2024 — Self-attention for multi-horizon wind pressure.

Architecture:
  Linear(4→128) input projection
  Sinusoidal positional encoding
  TransformerEncoder(3 layers, 8 heads, d_ff=256, dropout=0.1)
  → last time-step →
  Linear(128→64) → ReLU → Dropout → Linear(64→horizon)
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerForecaster(nn.Module):
    def __init__(self, n_features=4, d_model=128, nhead=8,
                 num_layers=3, dim_feedforward=256, dropout=0.1,
                 horizon=1, seq_length=100):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length,
                                               dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, horizon),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, T, n_features)
        x = self.input_proj(x)       # (B, T, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)      # (B, T, d_model)
        x = x[:, -1, :]              # last time step
        return self.fc(x)            # (B, horizon)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

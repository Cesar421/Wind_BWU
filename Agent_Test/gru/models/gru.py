"""
Pure GRU Model for Cp Time-Series Forecasting
==============================================
Simpler recurrent baseline (Chung et al. 2014) — fewer parameters than LSTM.

Architecture:
  GRU(4→256, 2 layers, dropout=0.2)  → last hidden
  Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→horizon)
"""

import torch
import torch.nn as nn


class PureGRU(nn.Module):
    def __init__(self, n_features=4, hidden_size=256, num_layers=2,
                 dropout=0.2, fc_hidden=128, fc_dropout=0.3, horizon=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden, horizon),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                for name, p in m.named_parameters():
                    if "weight_ih" in name: nn.init.xavier_uniform_(p)
                    elif "weight_hh" in name: nn.init.orthogonal_(p)
                    elif "bias" in name: nn.init.zeros_(p)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

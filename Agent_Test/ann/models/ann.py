"""
Feedforward ANN for Cp Prediction
===================================
Source: Aldoum & Stathopoulos 2025 — 4-hidden-layer ANN [100, 500, 500, 100].

Architecture:
  Flatten(seq_len × n_features)
  Linear(400→100) → ReLU → Dropout(0.3)
  Linear(100→500) → ReLU → Dropout(0.3)
  Linear(500→500) → ReLU → Dropout(0.3)
  Linear(500→100) → ReLU → Dropout(0.3)
  Linear(100→horizon)
"""

import torch
import torch.nn as nn


class FeedforwardANN(nn.Module):
    def __init__(self, n_features=4, seq_length=100,
                 hidden_layers=(100, 500, 500, 100),
                 dropout=0.3, horizon=1):
        super().__init__()
        self.seq_length = seq_length
        input_size = n_features * seq_length

        layers = []
        prev = input_size
        for h in hidden_layers:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, horizon))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, seq_len, n_features)
        x = x.reshape(x.size(0), -1)   # flatten → (B, seq_len*n_features)
        return self.net(x)              # (B, horizon)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

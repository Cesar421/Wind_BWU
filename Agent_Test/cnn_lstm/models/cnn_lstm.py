"""
CNN-LSTM Hybrid Model for Cp Time-Series Forecasting
=====================================================
Source: "A hybrid machine learning framework for wind pressure prediction
on buildings with constrained sensor networks" (MICE 2025)

Architecture:
  Conv1D(4→64) → BN → ReLU
  Conv1D(64→128) → BN → ReLU → MaxPool(2)
  Conv1D(128→256) → BN → ReLU
  LSTM(256→256, 2 layers, dropout=0.2)  → last hidden
  Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→horizon)
"""

import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 pool: bool = False):
        super().__init__()
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNNLSTM(nn.Module):
    def __init__(self, n_features=4, seq_length=100, horizon=1,
                 cnn_channels=(64, 128, 256), lstm_hidden=256,
                 lstm_layers=2, lstm_dropout=0.2,
                 fc_hidden=128, dropout=0.3):
        super().__init__()
        self.n_features = n_features
        self.seq_length = seq_length
        self.horizon = horizon
        c1, c2, c3 = cnn_channels

        self.cnn = nn.Sequential(
            CNNBlock(n_features, c1, kernel=3, pool=False),
            CNNBlock(c1, c2, kernel=3, pool=True),
            CNNBlock(c2, c3, kernel=3, pool=False),
        )
        self.lstm = nn.LSTM(c3, lstm_hidden, num_layers=lstm_layers,
                            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
                            batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, horizon),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if "weight_ih" in name: nn.init.xavier_uniform_(p)
                    elif "weight_hh" in name: nn.init.orthogonal_(p)
                    elif "bias" in name: nn.init.zeros_(p)

    def forward(self, x):
        x = x.permute(0, 2, 1)        # (B, feat, T)
        x = self.cnn(x)                # (B, c3, T')
        x = x.permute(0, 2, 1)        # (B, T', c3)
        out, _ = self.lstm(x)          # (B, T', H)
        x = out[:, -1, :]             # last step
        return self.fc(x)              # (B, horizon)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

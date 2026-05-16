"""
CNN-LSTM Hybrid Model for Wind Pressure Coefficient (Cp) Time-Series Forecasting
=================================================================================

Architecture based on:
    "A hybrid machine learning framework for wind pressure prediction on
     buildings with constrained sensor networks" (MICE 2025, doi:10.1111/mice.13488)

Design:
  ┌──────────────────────────────────────────────────────────────────┐
  │  Input  (batch, seq_len, n_features=4)                           │
  │  ↓                                                               │
  │  CNN Encoder  ──  extracts local temporal patterns               │
  │    Conv1D(4→64,  k=3, p=1) → BN → ReLU                         │
  │    Conv1D(64→128, k=3, p=1) → BN → ReLU → MaxPool(2)           │
  │    Conv1D(128→256, k=3, p=1) → BN → ReLU                       │
  │  ↓                                                               │
  │  LSTM Decoder  ──  captures long-range temporal dependencies     │
  │    LSTM(256→256, num_layers=2, dropout=0.2, batch_first=True)   │
  │  ↓  (take last hidden state)                                     │
  │  FC Head                                                         │
  │    Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→horizon) │
  │  ↓                                                               │
  │  Output (batch, horizon)                                         │
  └──────────────────────────────────────────────────────────────────┘

Input shape : (batch, seq_len, n_features)
Output shape: (batch, horizon)   — Cp windward face (normalised)
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    """Conv1D → BatchNorm1d → ReLU, with optional MaxPool."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model for Cp time-series single- or multi-step forecasting.

    Parameters
    ----------
    n_features   : number of input channels (4 façade faces)
    seq_length   : look-back window length
    horizon      : forecast horizon (default 1 — single-step)
    cnn_channels : list of [c1, c2, c3] output channels for the 3 CNN blocks
    lstm_hidden  : LSTM hidden size
    lstm_layers  : number of stacked LSTM layers
    lstm_dropout : dropout between LSTM layers (only if lstm_layers > 1)
    fc_hidden    : size of intermediate FC layer
    dropout      : dropout before the final output layer
    """

    def __init__(
        self,
        n_features: int = 4,
        seq_length: int = 100,
        horizon: int = 1,
        cnn_channels: tuple = (64, 128, 256),
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
        fc_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_length = seq_length
        self.horizon = horizon
        self.lstm_hidden = lstm_hidden

        c1, c2, c3 = cnn_channels

        # ── CNN Encoder (operates on (batch, channels, time))
        self.cnn = nn.Sequential(
            CNNBlock(n_features, c1, kernel=3, pool=False),
            CNNBlock(c1, c2, kernel=3, pool=True),   # seq_len → seq_len // 2
            CNNBlock(c2, c3, kernel=3, pool=False),
        )

        # ── LSTM Decoder
        self.lstm = nn.LSTM(
            input_size=c3,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        # ── FC Head
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, horizon),
        )

        self._init_weights()

    # ── weight initialisation ──────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    # ── forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, n_features)
        returns : (batch, horizon)
        """
        # CNN expects (batch, channels, time)
        x = x.permute(0, 2, 1)           # → (batch, n_features, seq_len)
        x = self.cnn(x)                   # → (batch, c3, seq_len')
        x = x.permute(0, 2, 1)           # → (batch, seq_len', c3)

        # LSTM
        lstm_out, _ = self.lstm(x)        # → (batch, seq_len', hidden)
        x = lstm_out[:, -1, :]            # last time step → (batch, hidden)

        # FC
        return self.fc(x)                 # → (batch, horizon)

    # ── inference helper ──────────────────────────────────────────────────

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward with torch.no_grad()."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    # ── parameter count ───────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── serialisation ────────────────────────────────────────────────────

    def get_config(self) -> Dict[str, Any]:
        return {
            "model": "CNNLSTM",
            "n_features": self.n_features,
            "seq_length": self.seq_length,
            "horizon": self.horizon,
            "lstm_hidden": self.lstm_hidden,
            "total_parameters": self.count_parameters(),
        }

    def save(self, path: str, extra: Optional[Dict] = None):
        payload = {
            "state_dict": self.state_dict(),
            "config": self.get_config(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "CNNLSTM":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ckpt["config"]
        model = cls(
            n_features=cfg["n_features"],
            seq_length=cfg["seq_length"],
            horizon=cfg["horizon"],
            lstm_hidden=cfg["lstm_hidden"],
        )
        model.load_state_dict(ckpt["state_dict"])
        return model

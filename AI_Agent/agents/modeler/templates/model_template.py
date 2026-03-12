"""
Base Wind Model Template
All agent-generated models must inherit from this class.
Provides a standardized interface for training, evaluation, and serialization.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseWindModel(nn.Module, ABC):
    """
    Abstract base class for all wind pressure Cp forecasting models.
    Agent-generated models MUST inherit from this class.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model_name = config.get("name", self.__class__.__name__)
        self.input_size = config.get("input_size", 1)
        self.output_size = config.get("output_size", 1)
        self.seq_length = config.get("seq_length", 100)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size) for single-step
            or (batch_size, horizon, output_size) for multi-step
        """
        pass

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference (no gradients)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for logging/serialization."""
        return {
            "model_name": self.model_name,
            "class": self.__class__.__name__,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "seq_length": self.seq_length,
            "total_parameters": self.count_parameters(),
            "config": self.config,
        }

    def get_model_summary(self) -> str:
        """Return a human-readable model summary."""
        lines = [
            f"Model: {self.model_name}",
            f"Class: {self.__class__.__name__}",
            f"Parameters: {self.count_parameters():,}",
            f"Input: (batch, {self.seq_length}, {self.input_size})",
            f"Output: (batch, {self.output_size})",
            "",
            "Architecture:",
        ]
        for name, module in self.named_children():
            lines.append(f"  {name}: {module}")
        return "\n".join(lines)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseWindModel":
        """Instantiate model from config dict."""
        return cls(config)

    def save(self, path: str):
        """Save model weights and config."""
        torch.save({
            "state_dict": self.state_dict(),
            "config": self.config,
            "class_name": self.__class__.__name__,
        }, path)

    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> "BaseWindModel":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model


# ---------------------------------------------------------------------------
# Seed model implementations (always available, not agent-generated)
# ---------------------------------------------------------------------------

class SeedLSTM(BaseWindModel):
    """Baseline LSTM — included as a seed model for comparison."""

    def __init__(self, config: Dict[str, Any]):
        config.setdefault("name", "SeedLSTM")
        config.setdefault("hidden_size", 64)
        config.setdefault("num_layers", 2)
        config.setdefault("dropout", 0.2)
        super().__init__(config)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"] if config["num_layers"] > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Linear(config["hidden_size"], self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class SeedBiLSTM(BaseWindModel):
    """Bidirectional LSTM seed model."""

    def __init__(self, config: Dict[str, Any]):
        config.setdefault("name", "SeedBiLSTM")
        config.setdefault("hidden_size", 64)
        config.setdefault("num_layers", 2)
        config.setdefault("dropout", 0.2)
        super().__init__(config)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"] if config["num_layers"] > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(config["hidden_size"] * 2, self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class SeedTCN(BaseWindModel):
    """Temporal Convolutional Network seed model."""

    def __init__(self, config: Dict[str, Any]):
        config.setdefault("name", "SeedTCN")
        config.setdefault("num_channels", [32, 64, 64])
        config.setdefault("kernel_size", 3)
        config.setdefault("dropout", 0.2)
        super().__init__(config)

        channels = config["num_channels"]
        k = config["kernel_size"]
        layers = []
        in_ch = self.input_size

        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            padding = (k - 1) * dilation
            layers.append(nn.Conv1d(in_ch, out_ch, k, dilation=dilation, padding=padding))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config["dropout"]))
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features) -> (batch, features, seq) for Conv1d
        out = self.tcn(x.transpose(1, 2))
        # Take last timestep
        return self.fc(out[:, :, -1])


class SeedCNNLSTM(BaseWindModel):
    """CNN-LSTM Hybrid seed model."""

    def __init__(self, config: Dict[str, Any]):
        config.setdefault("name", "SeedCNNLSTM")
        config.setdefault("cnn_filters", 64)
        config.setdefault("kernel_size", 3)
        config.setdefault("hidden_size", 64)
        config.setdefault("num_layers", 1)
        config.setdefault("dropout", 0.2)
        super().__init__(config)

        self.conv1 = nn.Conv1d(self.input_size, config["cnn_filters"], config["kernel_size"], padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(
            input_size=config["cnn_filters"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            batch_first=True,
        )
        self.fc = nn.Linear(config["hidden_size"], self.output_size)
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN part: (batch, seq, feat) -> (batch, feat, seq)
        out = self.conv1(x.transpose(1, 2))
        out = self.relu(out)
        out = self.pool(out)
        # Back to (batch, seq/2, filters)
        out = out.transpose(1, 2)
        # LSTM part
        lstm_out, _ = self.lstm(out)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)


# Registry of seed models
SEED_MODEL_REGISTRY = {
    "SeedLSTM": SeedLSTM,
    "SeedBiLSTM": SeedBiLSTM,
    "SeedTCN": SeedTCN,
    "SeedCNNLSTM": SeedCNNLSTM,
}


def get_seed_model(name: str, config: Dict[str, Any]) -> BaseWindModel:
    """Instantiate a seed model by name."""
    if name not in SEED_MODEL_REGISTRY:
        raise ValueError(f"Unknown seed model: {name}. Available: {list(SEED_MODEL_REGISTRY.keys())}")
    return SEED_MODEL_REGISTRY[name](config)

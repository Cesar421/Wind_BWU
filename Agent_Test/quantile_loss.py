"""
PinballLoss — multi-quantile direct forecasts (experiment 7: uncertainty bands).

The model outputs (B, Q*H), quantile-major: [q_0 h=1..H, q_1 h=1..H, ...].
Training minimises the mean pinball loss over quantiles and horizons; at
inference the Q values of each step are sorted (no quantile crossing) and the
median is the point forecast. The pinball loss of a quantile q on error
e = y - y_q is max(q e, (q - 1) e): an over-forecast costs (1 - q), an
under-forecast q, so each output settles on its quantile of the conditional
distribution.
"""
import torch


class PinballLoss(torch.nn.Module):
    def __init__(self, quantiles, H=500):
        super().__init__()
        if 0.5 not in quantiles:
            raise ValueError("the quantiles must include 0.5 (the point forecast)")
        self.quantiles = sorted(quantiles)
        self.H = H
        self.register_buffer("q", torch.tensor(self.quantiles, dtype=torch.float32)[None, :, None])

    def split(self, out):
        """(B, Q*H) -> (B, Q, H), sorted along Q."""
        return out.view(out.shape[0], len(self.quantiles), self.H).sort(dim=1).values

    def point(self, out):
        """Median forecast (B, H)."""
        return self.split(out)[:, self.quantiles.index(0.5)]

    def forward(self, out, y):
        e = y[:, None, :] - out.view(out.shape[0], len(self.quantiles), self.H)
        return torch.maximum(self.q * e, (self.q - 1) * e).mean()

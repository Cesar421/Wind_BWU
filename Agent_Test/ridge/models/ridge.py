"""
Ridge Regression baseline (sklearn).

A flat linear model fed with the concatenated sliding window
(seq_length * n_features) → 1 scalar Cp prediction.
"""

from sklearn.linear_model import Ridge


def build():
    # alpha is the L2 strength; small value keeps it close to OLS but stable.
    return Ridge(alpha=1.0, solver="auto", random_state=42)

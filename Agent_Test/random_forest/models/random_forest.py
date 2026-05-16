"""
Random Forest baseline (sklearn).

Nonlinear tree ensemble on flattened sliding windows.
Modest size — 200 trees, no max depth cap, parallel across CPU cores.
"""

from sklearn.ensemble import RandomForestRegressor


def build():
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )

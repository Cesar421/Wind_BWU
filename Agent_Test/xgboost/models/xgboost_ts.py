"""
XGBoost baseline (gradient boosting).

Uses GPU histogram method when CUDA is available (XGBoost ≥2.0 API:
device="cuda" + tree_method="hist"). Falls back to CPU automatically
if CUDA is not present.
"""

import xgboost as xgb


def _gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def build():
    device = "cuda" if _gpu_available() else "cpu"
    return xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        device=device,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )

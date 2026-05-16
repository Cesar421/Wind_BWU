"""
Training Script — XGBoost (gradient-boosting baseline, GPU if available).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# IMPORTANT: append (not insert) so pip-installed `xgboost` takes priority
# over the local folder `Agent_Test/xgboost/`.
sys.path.append(str(ROOT.parent))
sys.path.append(str(ROOT))

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import numpy as np

from classical_utils import run_classical
from models.xgboost_ts import build

np.random.seed(42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", type=str, default="Alpha1_4/2_1_3",
                        help='"Alpha1_4/2_1_3" (Round 1) | "round2" | "all"')
    args = parser.parse_args()

    run_classical(
        model_name="xgboost",
        build_estimator=build,
        root=ROOT,
        seq_length=100,
        step=10,
        horizon=1,
        horizons=[1, 10, 50, 100, 500],
        scope=args.scope,
    )


if __name__ == "__main__":
    main()

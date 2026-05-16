"""
Training Script — Random Forest (tree-ensemble baseline, CPU).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import numpy as np

from classical_utils import run_classical
from models.random_forest import build

np.random.seed(42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--building", type=str, default=None,
                        help="alpha/ratio e.g. Alpha1_4/2_1_3 for Round 1")
    args = parser.parse_args()

    run_classical(
        model_name="random_forest",
        build_estimator=build,
        root=ROOT,
        seq_length=100,
        step=10,
        horizon=1,
        horizons=[1, 10, 50, 100, 500],
        building=args.building,
    )


if __name__ == "__main__":
    main()

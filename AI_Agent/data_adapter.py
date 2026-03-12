"""
Data Integration Adapter — BDH Dataset
Loads Wind_pressure_coefficients from the Data_All_The_BDH .mat files
(TPU Aerodynamic Database) organized by alpha profile and building ratio.

Folder structure:
    Data/Data_All_The_BDH/
        Alpha1_4/
            1_1_2/  -> T112_4_000_1.mat, T112_4_005_1.mat, ...
            1_1_3/  -> T113_4_000_1.mat, ...
            ...
        Alpha1_6/
            ...
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io

logger = logging.getLogger(__name__)

# Project root (parent of AI_Agent/)
PROJECT_ROOT = Path(__file__).parent.parent

# Default BDH data directory
BDH_DATA_DIR = PROJECT_ROOT / "Data" / "Data_All_The_BDH"

# Building ratio -> number of pressure taps (for reference)
TAPS_BY_RATIO = {
    "1_1_2": 200, "1_1_3": 300, "1_1_4": 400, "1_1_5": 500,
    "2_1_2": 240, "2_1_3": 360, "2_1_4": 450, "2_1_5": 510,
    "3_1_2": 200, "3_1_3": 320, "3_1_4": 440, "3_1_5": 480,
}

# Which angles are available per building type
ANGLES_TYPE1 = list(range(0, 55, 5))     # 0-50 (11 angles) for 1:1:X
ANGLES_TYPE23 = list(range(0, 105, 5))   # 0-100 (21 angles) for 2:1:X, 3:1:X

# Valid alpha profiles
VALID_ALPHAS = ["Alpha1_4", "Alpha1_6"]

# Valid building ratios
VALID_RATIOS = list(TAPS_BY_RATIO.keys())


def _ratio_prefix(ratio: str) -> str:
    """Convert ratio folder name to filename prefix, e.g. '1_1_2' -> 'T112'."""
    return "T" + ratio.replace("_", "")


def _angles_for_ratio(ratio: str) -> List[int]:
    """Return valid angles based on building type."""
    if ratio.startswith("1_"):
        return ANGLES_TYPE1
    return ANGLES_TYPE23


class WindDataAdapter:
    """
    Unified data interface for loading BDH wind pressure .mat files.
    Each .mat file contains:
        - Wind_pressure_coefficients: (32768, num_taps), float32
        - Building_breadth, Building_depth, Building_height: scalar
        - Location_of_measured_points: (4, num_taps)
        - Sample_frequency: scalar (1000 Hz)
        - Wind_direction_angle: scalar
    """

    def __init__(
        self,
        bdh_dir: Optional[str] = None,
        alpha: str = "Alpha1_4",
        building_ratio: str = "1_1_3",
    ):
        self.bdh_dir = Path(bdh_dir) if bdh_dir else BDH_DATA_DIR
        self.alpha = alpha
        self.building_ratio = building_ratio
        self._cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Core loading
    # ------------------------------------------------------------------

    def _mat_path(self, alpha: str, ratio: str, angle: int) -> Path:
        """Build the path to a specific .mat file."""
        prefix = _ratio_prefix(ratio)
        alpha_num = alpha.replace("Alpha1_", "")  # "4" or "6"
        fname = f"{prefix}_{alpha_num}_{angle:03d}_1.mat"
        return self.bdh_dir / alpha / ratio / fname

    def load_mat(
        self,
        angle: int,
        alpha: Optional[str] = None,
        ratio: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load a single .mat file and return its contents.

        Returns:
            Dict with keys: 'cp' (32768, num_taps), 'frequency', 'angle',
            'building_breadth', 'building_depth', 'building_height',
            'tap_locations' (4, num_taps)
        """
        alpha = alpha or self.alpha
        ratio = ratio or self.building_ratio

        mat_path = self._mat_path(alpha, ratio, angle)
        if not mat_path.exists():
            raise FileNotFoundError(f"MAT file not found: {mat_path}")

        cache_key = f"{alpha}_{ratio}_{angle}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        raw = scipy.io.loadmat(str(mat_path))
        result = {
            "cp": raw["Wind_pressure_coefficients"],  # (32768, num_taps)
            "frequency": int(raw["Sample_frequency"][0, 0]),
            "angle": int(raw["Wind_direction_angle"][0, 0]),
            "building_breadth": float(raw["Building_breadth"][0, 0]),
            "building_depth": float(raw["Building_depth"][0, 0]),
            "building_height": float(raw["Building_height"][0, 0]),
            "tap_locations": raw["Location_of_measured_points"],
        }
        self._cache[cache_key] = result
        logger.info(f"Loaded {mat_path.name}: cp={result['cp'].shape}, "
                     f"B={result['building_breadth']}, D={result['building_depth']}, "
                     f"H={result['building_height']}")
        return result

    def load_cp(
        self,
        angle: int,
        alpha: Optional[str] = None,
        ratio: Optional[str] = None,
        tap: Optional[int] = None,
    ) -> np.ndarray:
        """
        Load Cp time series for a given angle.

        Args:
            angle: Wind direction angle in degrees
            alpha: Alpha profile (default: self.alpha)
            ratio: Building ratio (default: self.building_ratio)
            tap: Specific tap index. If None, returns mean across all taps.

        Returns:
            1D array of shape (32768,)
        """
        mat_data = self.load_mat(angle, alpha, ratio)
        cp = mat_data["cp"]  # (32768, num_taps)

        if tap is not None:
            if tap >= cp.shape[1]:
                raise ValueError(f"Tap {tap} out of range (max={cp.shape[1]-1})")
            return cp[:, tap].astype(np.float64)
        else:
            return cp.mean(axis=1).astype(np.float64)

    def get_available_angles(
        self,
        alpha: Optional[str] = None,
        ratio: Optional[str] = None,
    ) -> List[int]:
        """List angles with .mat files present on disk."""
        alpha = alpha or self.alpha
        ratio = ratio or self.building_ratio
        folder = self.bdh_dir / alpha / ratio
        if not folder.exists():
            return []
        angles = []
        for f in sorted(folder.glob("*.mat")):
            parts = f.stem.split("_")
            if len(parts) >= 3:
                angles.append(int(parts[2]))
        return angles

    # ------------------------------------------------------------------
    # Sequence creation
    # ------------------------------------------------------------------

    def get_training_data(
        self,
        angle: int = 0,
        seq_length: int = 100,
        tap: Optional[int] = None,
        normalize: bool = True,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        alpha: Optional[str] = None,
        ratio: Optional[str] = None,
        # Legacy params (ignored, kept for backward compatibility)
        face: str = "windward",
        column: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Get ready-to-use training data for a model.

        Args:
            angle: Wind direction angle
            seq_length: Input sequence length
            tap: Specific tap (None = mean of all taps)
            normalize: Normalize to [-1, 1]
            train_ratio: Training set proportion
            val_ratio: Validation set proportion
            alpha: Alpha profile override
            ratio: Building ratio override

        Returns:
            Dict with X_train, y_train, X_val, y_val, X_test, y_test,
            test_seed, y_future, scaler_params
        """
        data = self.load_cp(angle, alpha=alpha, ratio=ratio, tap=tap)

        # Normalize to [-1, 1]
        scaler_params = {}
        if normalize:
            data_min = data.min()
            data_max = data.max()
            data_range = data_max - data_min
            if data_range > 0:
                data = 2 * (data - data_min) / data_range - 1
            scaler_params = {"min": float(data_min), "max": float(data_max)}

        # Create sequences
        X, y = self._create_sequences(data, seq_length)

        # Time-series split (chronological)
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        # True forecasting seed
        test_ts_start = val_end + seq_length
        test_seed = data[test_ts_start - seq_length:test_ts_start]
        y_future = data[test_ts_start:test_ts_start + 500]

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "test_seed": test_seed,
            "y_future": y_future,
            "scaler_params": scaler_params,
            "data_length": len(data),
            "seq_length": seq_length,
        }

    def get_multi_angle_data(
        self,
        angles: Optional[List[int]] = None,
        seq_length: int = 100,
        tap: Optional[int] = None,
        normalize: bool = True,
        alpha: Optional[str] = None,
        ratio: Optional[str] = None,
        # Legacy params
        face: str = "windward",
    ) -> Dict[str, np.ndarray]:
        """
        Get training data pooled across multiple angles.
        Useful for training a single model that generalizes across angles.
        """
        alpha = alpha or self.alpha
        ratio = ratio or self.building_ratio
        angles = angles or self.get_available_angles(alpha, ratio)

        all_X_train, all_y_train = [], []
        all_X_val, all_y_val = [], []
        all_X_test, all_y_test = [], []

        for angle in angles:
            try:
                data_dict = self.get_training_data(
                    angle=angle, seq_length=seq_length, tap=tap,
                    normalize=normalize, alpha=alpha, ratio=ratio,
                )
                all_X_train.append(data_dict["X_train"])
                all_y_train.append(data_dict["y_train"])
                all_X_val.append(data_dict["X_val"])
                all_y_val.append(data_dict["y_val"])
                all_X_test.append(data_dict["X_test"])
                all_y_test.append(data_dict["y_test"])
            except FileNotFoundError:
                logger.debug(f"No data for alpha={alpha}, ratio={ratio}, angle={angle}")
                continue

        if not all_X_train:
            raise FileNotFoundError(
                f"No .mat files found for alpha={alpha}, ratio={ratio}"
            )

        return {
            "X_train": np.concatenate(all_X_train, axis=0),
            "y_train": np.concatenate(all_y_train, axis=0),
            "X_val": np.concatenate(all_X_val, axis=0),
            "y_val": np.concatenate(all_y_val, axis=0),
            "X_test": np.concatenate(all_X_test, axis=0),
            "y_test": np.concatenate(all_y_test, axis=0),
        }

    def get_multi_ratio_data(
        self,
        ratios: Optional[List[str]] = None,
        alpha: Optional[str] = None,
        seq_length: int = 100,
        tap: Optional[int] = None,
        normalize: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Get training data pooled across multiple building ratios and all
        their available angles. Useful for large-scale cross-building training.
        """
        alpha = alpha or self.alpha
        ratios = ratios or VALID_RATIOS

        all_X_train, all_y_train = [], []
        all_X_val, all_y_val = [], []
        all_X_test, all_y_test = [], []

        for ratio in ratios:
            try:
                data_dict = self.get_multi_angle_data(
                    seq_length=seq_length, tap=tap, normalize=normalize,
                    alpha=alpha, ratio=ratio,
                )
                all_X_train.append(data_dict["X_train"])
                all_y_train.append(data_dict["y_train"])
                all_X_val.append(data_dict["X_val"])
                all_y_val.append(data_dict["y_val"])
                all_X_test.append(data_dict["X_test"])
                all_y_test.append(data_dict["y_test"])
                logger.info(f"Loaded ratio {ratio}: "
                            f"train={data_dict['X_train'].shape[0]} sequences")
            except FileNotFoundError:
                logger.warning(f"Skipping ratio {ratio} (no data for alpha={alpha})")
                continue

        if not all_X_train:
            raise FileNotFoundError(
                f"No data found for any ratio with alpha={alpha}"
            )

        return {
            "X_train": np.concatenate(all_X_train, axis=0),
            "y_train": np.concatenate(all_y_train, axis=0),
            "X_val": np.concatenate(all_X_val, axis=0),
            "y_val": np.concatenate(all_y_val, axis=0),
            "X_test": np.concatenate(all_X_test, axis=0),
            "y_test": np.concatenate(all_y_test, axis=0),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding-window sequences."""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        X = np.array(X).reshape(-1, seq_length, 1)  # (N, seq, 1)
        y = np.array(y)
        return X, y

    def summarize(self) -> str:
        """Print a summary of available data."""
        lines = ["Wind Pressure BDH Data Summary", "=" * 50]
        for alpha in VALID_ALPHAS:
            lines.append(f"\n{alpha}:")
            for ratio in VALID_RATIOS:
                angles = self.get_available_angles(alpha, ratio)
                if angles:
                    # Load one file to get tap count
                    try:
                        mat = self.load_mat(angles[0], alpha, ratio)
                        n_taps = mat["cp"].shape[1]
                        lines.append(
                            f"  {ratio}: {len(angles)} angles, "
                            f"{n_taps} taps, "
                            f"B={mat['building_breadth']:.2f}, "
                            f"D={mat['building_depth']:.2f}, "
                            f"H={mat['building_height']:.2f}"
                        )
                    except Exception as e:
                        lines.append(f"  {ratio}: error loading - {e}")
                else:
                    lines.append(f"  {ratio}: NO DATA")
        return "\n".join(lines)

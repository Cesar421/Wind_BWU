"""
Data Integration Adapter — BDH Dataset
Loads Wind_pressure_coefficients from:
  1. Raw .mat files in Data_All_The_BDH/ (original per-tap time series)
  2. Postprocessed .npy files in Data_All_The_BDH_PostProcess/
     (face-averaged: windward, leeward, sideleft, sideright)

Folder structure (postprocessed):
    Data/Data_All_The_BDH_PostProcess/
        summary_all_buildings.csv
        Alpha1_4/
            1_1_2/Data/
                windward_avg_angle_0.npy   # shape (32768,)
                leeward_avg_angle_0.npy
                sideleft_avg_angle_0.npy
                sideright_avg_angle_0.npy
                statistics_all_angles.csv
                ...
            ...
        Alpha1_6/
            ...
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io

logger = logging.getLogger(__name__)

# Project root (parent of AI_Agent/)
PROJECT_ROOT = Path(__file__).parent.parent

# Default BDH data directory (raw .mat)
BDH_DATA_DIR = PROJECT_ROOT / "Data" / "Data_All_The_BDH"

# Postprocessed data directory (face-averaged .npy)
POSTPROCESS_DIR = PROJECT_ROOT / "Data" / "Data_All_The_BDH_PostProcess"

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

# Four building faces in the postprocessed data
FACES = ["windward", "leeward", "sideleft", "sideright"]


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


# ======================================================================
# Postprocessed Data Adapter — Face-averaged .npy files
# ======================================================================

class PostProcessDataAdapter:
    """
    Load postprocessed wind pressure data (face-averaged Cp time series).

    Each .npy file is shape (32768,) representing the spatial average of Cp
    across all taps on one face (windward, leeward, sideleft, sideright)
    for a specific building, alpha, and angle.

    The adapter can produce:
      - Univariate: single face, shape (N, seq, 1)
      - Multivariate: 4 faces stacked, shape (N, seq, 4)
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        alpha: str = "Alpha1_4",
        building_ratio: str = "1_1_3",
    ):
        self.data_dir = Path(data_dir) if data_dir else POSTPROCESS_DIR
        self.alpha = alpha
        self.building_ratio = building_ratio
        self._cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Core loading
    # ------------------------------------------------------------------

    def _npy_path(self, alpha: str, ratio: str, face: str, angle: int) -> Path:
        """Build the path to a specific .npy file."""
        return self.data_dir / alpha / ratio / "Data" / f"{face}_avg_angle_{angle}.npy"

    def load_face(
        self,
        face: str,
        angle: int,
        alpha: Optional[str] = None,
        ratio: Optional[str] = None,
    ) -> np.ndarray:
        """
        Load a single face time series.

        Returns:
            1D array of shape (32768,)
        """
        alpha = alpha or self.alpha
        ratio = ratio or self.building_ratio

        cache_key = f"{alpha}_{ratio}_{face}_{angle}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        npy_path = self._npy_path(alpha, ratio, face, angle)
        if not npy_path.exists():
            raise FileNotFoundError(f"Postprocess file not found: {npy_path}")

        arr = np.load(str(npy_path)).astype(np.float32)
        self._cache[cache_key] = arr
        logger.debug(f"Loaded {npy_path.name}: shape={arr.shape}")
        return arr

    def load_multivariate(
        self,
        angle: int,
        faces: Optional[List[str]] = None,
        alpha: Optional[str] = None,
        ratio: Optional[str] = None,
    ) -> np.ndarray:
        """
        Load multiple faces as a multivariate time series.

        Args:
            angle: Wind direction angle
            faces: List of faces to load (default: all 4)

        Returns:
            2D array of shape (32768, num_faces)
        """
        faces = faces or FACES
        arrays = []
        for face in faces:
            arr = self.load_face(face, angle, alpha, ratio)
            arrays.append(arr)
        return np.column_stack(arrays)

    def get_available_angles(
        self,
        alpha: Optional[str] = None,
        ratio: Optional[str] = None,
    ) -> List[int]:
        """List angles with postprocessed data present on disk."""
        alpha = alpha or self.alpha
        ratio = ratio or self.building_ratio
        folder = self.data_dir / alpha / ratio / "Data"
        if not folder.exists():
            return []
        angles = set()
        for f in folder.glob("windward_avg_angle_*.npy"):
            angle_str = f.stem.split("_avg_angle_")[-1]
            angles.add(int(angle_str))
        return sorted(angles)

    def get_available_buildings(self) -> List[Dict[str, str]]:
        """List all alpha/ratio combinations with postprocessed data."""
        buildings = []
        for alpha_dir in sorted(self.data_dir.iterdir()):
            if not alpha_dir.is_dir() or not alpha_dir.name.startswith("Alpha"):
                continue
            for ratio_dir in sorted(alpha_dir.iterdir()):
                if not ratio_dir.is_dir():
                    continue
                data_dir = ratio_dir / "Data"
                if data_dir.exists() and list(data_dir.glob("*.npy")):
                    buildings.append({
                        "alpha": alpha_dir.name,
                        "ratio": ratio_dir.name,
                    })
        return buildings

    def load_summary_csv(self) -> pd.DataFrame:
        """Load the global summary statistics CSV."""
        csv_path = self.data_dir / "summary_all_buildings.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Summary CSV not found: {csv_path}")
        return pd.read_csv(csv_path)

    def load_statistics(
        self,
        alpha: Optional[str] = None,
        ratio: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load the per-building statistics CSV."""
        alpha = alpha or self.alpha
        ratio = ratio or self.building_ratio
        csv_path = self.data_dir / alpha / ratio / "Data" / "statistics_all_angles.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Statistics CSV not found: {csv_path}")
        return pd.read_csv(csv_path)

    # ------------------------------------------------------------------
    # Sequence creation
    # ------------------------------------------------------------------

    @staticmethod
    def _create_sequences(
        data: np.ndarray, seq_length: int, step: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding-window sequences from a time series.

        Args:
            data: 1D (T,) or 2D (T, features) array
            step: Stride between consecutive windows (1=fully overlapping)

        Returns:
            X: (N, seq_length, features)
            y: (N, features)  — next-step target for each feature
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        T, F = data.shape
        indices = range(0, T - seq_length, step)
        N = len(indices)
        X = np.zeros((N, seq_length, F), dtype=np.float32)
        y = np.zeros((N, F), dtype=np.float32)
        for idx, i in enumerate(indices):
            X[idx] = data[i : i + seq_length]
            y[idx] = data[i + seq_length]
        return X, y

    def get_training_data(
        self,
        angle: int = 0,
        seq_length: int = 100,
        step: int = 1,
        faces: Optional[List[str]] = None,
        normalize: bool = True,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        alpha: Optional[str] = None,
        ratio: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get ready-to-use training data for a model.

        Args:
            angle: Wind direction angle in degrees
            seq_length: Input sequence length
            step: Stride for sliding window (increase to reduce memory)
            faces: List of faces to use (default: all 4 → multivariate)
            normalize: Normalize each feature to [-1, 1]
            train_ratio: Training set proportion
            val_ratio: Validation set proportion

        Returns:
            Dict with X_train, y_train, X_val, y_val, X_test, y_test,
            test_seed, y_future, scaler_params, feature_names
        """
        faces = faces or FACES
        data = self.load_multivariate(angle, faces, alpha, ratio)  # (T, F)

        # Normalize each feature to [-1, 1]
        scaler_params = {}
        if normalize:
            normed = np.zeros_like(data)
            for i, face in enumerate(faces):
                col = data[:, i]
                cmin, cmax = col.min(), col.max()
                crange = cmax - cmin
                if crange > 0:
                    normed[:, i] = 2 * (col - cmin) / crange - 1
                else:
                    normed[:, i] = 0.0
                scaler_params[face] = {"min": float(cmin), "max": float(cmax)}
            data = normed

        # Create sequences: X (N, seq, F), y (N, F)
        X, y = self._create_sequences(data, seq_length, step=step)

        # Time-series split (chronological)
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        # True forecasting seed (multivariate)
        test_ts_start = val_end + seq_length
        test_seed = data[test_ts_start - seq_length : test_ts_start]  # (seq, F)
        y_future = data[test_ts_start : test_ts_start + 500]          # (500, F)

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
            "feature_names": faces,
            "data_length": data.shape[0],
            "num_features": data.shape[1],
            "seq_length": seq_length,
        }

    def get_multi_angle_data(
        self,
        angles: Optional[List[int]] = None,
        seq_length: int = 100,
        step: int = 1,
        faces: Optional[List[str]] = None,
        normalize: bool = True,
        alpha: Optional[str] = None,
        ratio: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get training data pooled across multiple angles for one building.
        """
        alpha = alpha or self.alpha
        ratio = ratio or self.building_ratio
        angles = angles or self.get_available_angles(alpha, ratio)
        faces = faces or FACES

        all_X_train, all_y_train = [], []
        all_X_val, all_y_val = [], []
        all_X_test, all_y_test = [], []

        for angle in angles:
            try:
                d = self.get_training_data(
                    angle=angle, seq_length=seq_length, step=step,
                    faces=faces, normalize=normalize,
                    alpha=alpha, ratio=ratio,
                )
                all_X_train.append(d["X_train"])
                all_y_train.append(d["y_train"])
                all_X_val.append(d["X_val"])
                all_y_val.append(d["y_val"])
                all_X_test.append(d["X_test"])
                all_y_test.append(d["y_test"])
            except FileNotFoundError:
                logger.debug(f"No postprocess data for {alpha}/{ratio}/angle={angle}")
                continue

        if not all_X_train:
            raise FileNotFoundError(
                f"No postprocess data found for alpha={alpha}, ratio={ratio}"
            )

        return {
            "X_train": np.concatenate(all_X_train, axis=0),
            "y_train": np.concatenate(all_y_train, axis=0),
            "X_val": np.concatenate(all_X_val, axis=0),
            "y_val": np.concatenate(all_y_val, axis=0),
            "X_test": np.concatenate(all_X_test, axis=0),
            "y_test": np.concatenate(all_y_test, axis=0),
            "feature_names": faces,
            "num_features": len(faces),
        }

    def get_multi_building_data(
        self,
        alphas: Optional[List[str]] = None,
        ratios: Optional[List[str]] = None,
        seq_length: int = 100,
        step: int = 10,
        faces: Optional[List[str]] = None,
        normalize: bool = True,
    ) -> Dict[str, Any]:
        """
        Get training data pooled across multiple buildings and all their angles.
        This is the recommended method for training a generalizable model
        using ALL postprocessed data.
        """
        alphas = alphas or VALID_ALPHAS
        faces = faces or FACES

        all_X_train, all_y_train = [], []
        all_X_val, all_y_val = [], []
        all_X_test, all_y_test = [], []

        buildings_loaded = 0
        for alpha in alphas:
            # Determine which ratios are available for this alpha
            available = []
            if ratios:
                available = ratios
            else:
                alpha_dir = self.data_dir / alpha
                if alpha_dir.exists():
                    available = [
                        d.name for d in sorted(alpha_dir.iterdir())
                        if d.is_dir() and (d / "Data").exists()
                    ]

            for ratio in available:
                try:
                    d = self.get_multi_angle_data(
                        seq_length=seq_length, step=step, faces=faces,
                        normalize=normalize, alpha=alpha, ratio=ratio,
                    )
                    all_X_train.append(d["X_train"])
                    all_y_train.append(d["y_train"])
                    all_X_val.append(d["X_val"])
                    all_y_val.append(d["y_val"])
                    all_X_test.append(d["X_test"])
                    all_y_test.append(d["y_test"])
                    buildings_loaded += 1
                    logger.info(
                        f"  Loaded {alpha}/{ratio}: "
                        f"train={d['X_train'].shape[0]} sequences"
                    )
                except FileNotFoundError:
                    logger.warning(f"  Skipping {alpha}/{ratio} (no data)")
                    continue

        if not all_X_train:
            raise FileNotFoundError("No postprocessed data found")

        result = {
            "X_train": np.concatenate(all_X_train, axis=0),
            "y_train": np.concatenate(all_y_train, axis=0),
            "X_val": np.concatenate(all_X_val, axis=0),
            "y_val": np.concatenate(all_y_val, axis=0),
            "X_test": np.concatenate(all_X_test, axis=0),
            "y_test": np.concatenate(all_y_test, axis=0),
            "feature_names": faces,
            "num_features": len(faces),
            "buildings_loaded": buildings_loaded,
        }

        logger.info(
            f"Multi-building data: {buildings_loaded} buildings, "
            f"train={result['X_train'].shape}, "
            f"val={result['X_val'].shape}, "
            f"test={result['X_test'].shape}"
        )
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def summarize(self) -> str:
        """Print a summary of available postprocessed data."""
        lines = ["Wind Pressure PostProcessed Data Summary", "=" * 50]
        buildings = self.get_available_buildings()
        for b in buildings:
            alpha, ratio = b["alpha"], b["ratio"]
            angles = self.get_available_angles(alpha, ratio)
            lines.append(
                f"  {alpha}/{ratio}: {len(angles)} angles, "
                f"faces={FACES}"
            )
        lines.append(f"\nTotal: {len(buildings)} buildings")

        # Load summary stats
        try:
            df = self.load_summary_csv()
            lines.append(f"Summary CSV: {df.shape[0]} rows, {df.shape[1]} cols")
        except FileNotFoundError:
            pass

        return "\n".join(lines)

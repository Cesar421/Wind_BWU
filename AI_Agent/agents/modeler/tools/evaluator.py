"""
Model Evaluator Tool
Comprehensive evaluation of models with interpolation & true forecasting metrics.
Supports TimeSeriesSplit cross-validation and comparison against baselines.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of correct direction predictions."""
    if len(y_true) < 2:
        return 0.0
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(true_dir == pred_dir) * 100)


class ModelEvaluator:
    """Evaluate wind pressure models comprehensively."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_interpolation(
        self,
        model: nn.Module,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate one-step-ahead interpolation (teacher forcing).

        Args:
            model: Trained model
            X_test: Test input (N, seq_len, features)
            y_test: Test target (N,) or (N, 1)

        Returns:
            Dict of metric_name -> value
        """
        model.eval()
        model = model.to(self.device)

        X_t = torch.FloatTensor(X_test)
        if X_t.ndim == 2:
            X_t = X_t.unsqueeze(2)

        with torch.no_grad():
            # Process in batches to avoid OOM
            preds = []
            batch_size = 256
            for i in range(0, len(X_t), batch_size):
                batch = X_t[i:i + batch_size].to(self.device)
                out = model(batch)
                preds.append(out.cpu().numpy())

        y_pred = np.concatenate(preds, axis=0).squeeze()
        y_true = y_test.squeeze()

        return {
            "rmse": rmse(y_true, y_pred),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "mape": mape(y_true, y_pred),
            "directional_accuracy": directional_accuracy(y_true, y_pred),
        }

    def evaluate_true_forecasting(
        self,
        model: nn.Module,
        initial_sequence: np.ndarray,
        y_true_future: np.ndarray,
        horizons: List[int] = [10, 50, 100, 500],
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate TRUE multi-step forecasting (autoregressive, no teacher forcing).

        The model recursively feeds its own predictions back as input.

        Args:
            model: Trained model
            initial_sequence: Initial input (seq_len, features) — the seed
            y_true_future: Ground truth for future timesteps
            horizons: List of forecast horizons to evaluate

        Returns:
            Dict of horizon -> metrics
        """
        model.eval()
        model = model.to(self.device)

        seq_len = initial_sequence.shape[0]
        features = initial_sequence.shape[1] if initial_sequence.ndim > 1 else 1

        # Build running sequence
        if initial_sequence.ndim == 1:
            running = initial_sequence.copy().reshape(-1, 1)
        else:
            running = initial_sequence.copy()

        max_horizon = min(max(horizons), len(y_true_future))
        predictions = []

        with torch.no_grad():
            for step in range(max_horizon):
                # Take last seq_len values
                input_seq = running[-seq_len:]
                x = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)  # (1, seq, feat)

                pred = model(x)  # (1, output_size)
                pred_val = pred.cpu().numpy().squeeze()

                predictions.append(pred_val)

                # Append prediction to running sequence
                if running.ndim == 2:
                    running = np.vstack([running, [[pred_val]]])
                else:
                    running = np.append(running, pred_val)

        predictions = np.array(predictions)
        results = {}

        for h in horizons:
            if h > max_horizon:
                continue
            y_pred_h = predictions[:h]
            y_true_h = y_true_future[:h].squeeze()

            if len(y_pred_h) != len(y_true_h):
                continue

            results[f"horizon_{h}"] = {
                "rmse": rmse(y_true_h, y_pred_h),
                "mae": float(mean_absolute_error(y_true_h, y_pred_h)),
                "r2": float(r2_score(y_true_h, y_pred_h)),
                "mape": mape(y_true_h, y_pred_h),
                "directional_accuracy": directional_accuracy(y_true_h, y_pred_h),
            }

        return results

    def cross_validate(
        self,
        model_class: type,
        model_config: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        trainer=None,
    ) -> Dict[str, Any]:
        """
        Time series cross-validation.

        Args:
            model_class: Model class to instantiate
            model_config: Config dict for model
            X: Full input data
            y: Full target data
            n_splits: Number of CV folds
            trainer: ModelTrainer instance

        Returns:
            Dict with fold results and aggregated metrics
        """
        from agents.modeler.tools.trainer import ModelTrainer

        if trainer is None:
            from config import get_config
            cfg = get_config()
            trainer = ModelTrainer(cfg.training_config, self.device)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []

        logger.info(f"Cross-validation con {n_splits} folds...")

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"  Fold {fold + 1}/{n_splits}...")

            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            # Split train into train/val (90/10)
            val_size = max(int(len(X_train) * 0.1), 1)
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train = X_train[:-val_size]
            y_train = y_train[:-val_size]

            # Create fresh model
            model = model_class(model_config.copy())

            # Train
            history = trainer.train(
                model, X_train, y_train, X_val, y_val,
                model_name=f"CV_fold_{fold}",
            )

            # Evaluate
            metrics = self.evaluate_interpolation(model, X_test, y_test)
            metrics["fold"] = fold
            metrics["train_size"] = len(X_train)
            metrics["test_size"] = len(X_test)
            metrics["best_val_loss"] = history["best_val_loss"]
            fold_results.append(metrics)

            logger.info(f"    Fold {fold + 1}: RMSE={metrics['rmse']:.6f}, R²={metrics['r2']:.4f}")

        # Aggregate
        aggregated = {}
        for metric in ["rmse", "mae", "r2", "mape", "directional_accuracy"]:
            values = [f[metric] for f in fold_results if metric in f]
            if values:
                aggregated[f"{metric}_mean"] = float(np.mean(values))
                aggregated[f"{metric}_std"] = float(np.std(values))

        return {
            "folds": fold_results,
            "aggregated": aggregated,
            "n_splits": n_splits,
        }

    def compare_with_baselines(
        self,
        model_results: Dict[str, float],
        baseline_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compare model results against existing baselines.

        Args:
            model_results: Dict with at least 'rmse' and 'r2' keys
            baseline_file: Path to CSV with baseline results

        Returns:
            Comparison dict
        """
        # Hardcoded baselines from the existing project
        baselines = {
            "Ridge Regression": {"rmse_interp": 0.0391, "r2_interp": 0.9893, "rmse_forecast": 0.3872, "r2_forecast": -0.051},
            "Linear Regression": {"rmse_interp": 0.0392, "r2_interp": 0.9892, "rmse_forecast": 0.3872, "r2_forecast": -0.051},
            "Random Forest": {"rmse_interp": 0.0411, "r2_interp": 0.9882, "rmse_forecast": 0.6330, "r2_forecast": -1.810},
            "Naive Persistence": {"rmse_interp": 0.0473, "r2_interp": 0.9843, "rmse_forecast": 0.4551, "r2_forecast": -0.453},
        }

        # Try loading from file
        if baseline_file:
            try:
                import pandas as pd
                df = pd.read_csv(baseline_file)
                for _, row in df.iterrows():
                    name = row.get("model", row.get("Model", ""))
                    if name:
                        baselines[name] = {
                            "rmse_interp": row.get("rmse", row.get("RMSE", None)),
                            "r2_interp": row.get("r2", row.get("R2", None)),
                        }
            except Exception as e:
                logger.warning(f"Could not load baselines from {baseline_file}: {e}")

        comparison = {
            "model_metrics": model_results,
            "baselines": baselines,
            "rankings": {},
        }

        # Rank by RMSE (interpolation)
        if "rmse" in model_results:
            all_rmse = {name: b["rmse_interp"] for name, b in baselines.items() if b.get("rmse_interp")}
            all_rmse["Current Model"] = model_results["rmse"]
            sorted_models = sorted(all_rmse.items(), key=lambda x: x[1])
            comparison["rankings"]["rmse_rank"] = {
                name: rank + 1 for rank, (name, _) in enumerate(sorted_models)
            }

        return comparison

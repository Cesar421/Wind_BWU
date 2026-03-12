"""
Hyperparameter Tuner Tool
Optuna-based hyperparameter optimization for BaseWindModel instances.
"""

import logging
from typing import Any, Callable, Dict, Optional, Type

import numpy as np

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("Optuna not installed. HPO disabled. Run: pip install optuna")


class HyperparameterTuner:
    """Optuna-based hyperparameter optimization."""

    def __init__(self, config: Dict[str, Any]):
        if not HAS_OPTUNA:
            raise ImportError("Optuna required. Install with: pip install optuna")

        self.n_trials = config.get("n_trials", 50)
        self.timeout = config.get("timeout_per_trial", 600)
        self.pruner_type = config.get("pruner", "median")
        self.sampler_type = config.get("sampler", "tpe")

    def optimize(
        self,
        model_class: Type,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        search_space: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            model_class: Model class to optimize
            X_train, y_train: Training data
            X_val, y_val: Validation data
            search_space: Optional custom search space
            device: Device to train on

        Returns:
            Dict with best params, best value, and study summary
        """
        import torch
        from agents.modeler.tools.trainer import ModelTrainer

        if search_space is None:
            search_space = self._default_search_space()

        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            config = {}
            for param_name, param_def in search_space.items():
                ptype = param_def["type"]
                if ptype == "int":
                    config[param_name] = trial.suggest_int(
                        param_name, param_def["low"], param_def["high"],
                        log=param_def.get("log", False),
                    )
                elif ptype == "float":
                    config[param_name] = trial.suggest_float(
                        param_name, param_def["low"], param_def["high"],
                        log=param_def.get("log", False),
                    )
                elif ptype == "categorical":
                    config[param_name] = trial.suggest_categorical(
                        param_name, param_def["choices"]
                    )

            # Add fixed params
            config["input_size"] = X_train.shape[2] if X_train.ndim == 3 else 1
            config["output_size"] = 1
            config["seq_length"] = X_train.shape[1] if X_train.ndim >= 2 else 100

            try:
                model = model_class(config)
                trainer_config = {
                    "default_epochs": 50,  # Shorter for HPO
                    "batch_size": config.get("batch_size", 64),
                    "learning_rate": config.get("learning_rate", 0.001),
                    "weight_decay": config.get("weight_decay", 0.0001),
                    "gradient_clip": 1.0,
                    "early_stopping_patience": 10,
                    "lr_scheduler_patience": 3,
                    "lr_scheduler_factor": 0.5,
                    "mixed_precision": device == "cuda",
                }
                trainer = ModelTrainer(trainer_config, device)
                history = trainer.train(model, X_train, y_train, X_val, y_val, epochs=50)

                val_loss = history["best_val_loss"]

                # Report intermediate values for pruning
                for epoch_loss in history["val_loss"]:
                    trial.report(epoch_loss, step=len(history["val_loss"]))
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                return val_loss

            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float("inf")

        # Create study
        sampler = TPESampler(seed=42) if self.sampler_type == "tpe" else None
        pruner = MedianPruner() if self.pruner_type == "median" else None

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )

        logger.info(f"Starting HPO: {self.n_trials} trials...")
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout * self.n_trials)

        best = study.best_trial
        logger.info(f"Best trial: val_loss={best.value:.6f}")
        logger.info(f"   Params: {best.params}")

        return {
            "best_params": best.params,
            "best_value": best.value,
            "n_trials_completed": len(study.trials),
            "n_trials_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        }

    @staticmethod
    def _default_search_space() -> Dict[str, Any]:
        """Default search space for wind pressure models."""
        return {
            "hidden_size": {"type": "int", "low": 32, "high": 256, "log": False},
            "num_layers": {"type": "int", "low": 1, "high": 4},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "batch_size": {"type": "categorical", "choices": [32, 64, 128, 256]},
            "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
        }

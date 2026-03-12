"""
Model Trainer Tool
Generic training pipeline that works with any BaseWindModel.
Supports early stopping, LR scheduling, mixed precision, and MLflow tracking.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


class EarlyStopping:
    """Early stopping to halt training when val loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class ModelTrainer:
    """
    Generic trainer for BaseWindModel instances.
    Handles the full training loop with best practices.
    """

    def __init__(self, config: Dict[str, Any], device: Optional[str] = None):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Training hyperparameters
        self.epochs = config.get("default_epochs", 200)
        self.batch_size = config.get("batch_size", 64)
        self.lr = config.get("learning_rate", 0.001)
        self.weight_decay = config.get("weight_decay", 0.0001)
        self.grad_clip = config.get("gradient_clip", 1.0)
        self.patience = config.get("early_stopping_patience", 15)
        self.lr_patience = config.get("lr_scheduler_patience", 5)
        self.lr_factor = config.get("lr_scheduler_factor", 0.5)
        self.use_amp = config.get("mixed_precision", True) and self.device == "cuda"
        self.num_workers = config.get("num_workers", 4)

    def train(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: Optional[int] = None,
        model_name: Optional[str] = None,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train a model with full pipeline.

        Args:
            model: PyTorch model (BaseWindModel)
            X_train: Training input, shape (N, seq_len, features)
            y_train: Training target, shape (N,) or (N, output_size)
            X_val: Validation input
            y_val: Validation target
            epochs: Override default epochs
            model_name: Name for logging
            save_dir: Directory to save best checkpoint

        Returns:
            Dict with training history and best metrics
        """
        epochs = epochs or self.epochs
        model_name = model_name or getattr(model, "model_name", model.__class__.__name__)

        logger.info(f"Training {model_name} on {self.device} ({epochs} epochs)")
        logger.info(f"   Train: {X_train.shape}, Val: {X_val.shape}")

        model = model.to(self.device)

        # Prepare data loaders
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        # Optimizer & scheduler
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self.lr_factor, patience=self.lr_patience
        )
        criterion = nn.MSELoss()
        early_stop = EarlyStopping(patience=self.patience)
        scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "best_epoch": 0,
            "best_val_loss": float("inf"),
            "training_time_sec": 0,
        }

        best_state = None
        start_time = time.time()

        for epoch in range(epochs):
            # --- Train ---
            model.train()
            train_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        pred = model(X_batch)
                        loss = criterion(pred.squeeze(), y_batch.squeeze())
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(X_batch)
                    loss = criterion(pred.squeeze(), y_batch.squeeze())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                    optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            avg_train_loss = train_loss / max(n_batches, 1)

            # --- Validate ---
            model.eval()
            val_loss = 0.0
            n_val = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    if self.use_amp:
                        with torch.amp.autocast("cuda"):
                            pred = model(X_batch)
                            loss = criterion(pred.squeeze(), y_batch.squeeze())
                    else:
                        pred = model(X_batch)
                        loss = criterion(pred.squeeze(), y_batch.squeeze())

                    val_loss += loss.item()
                    n_val += 1

            avg_val_loss = val_loss / max(n_val, 1)

            # Update scheduler
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # Record history
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["lr"].append(current_lr)

            # Check for best model
            if avg_val_loss < history["best_val_loss"]:
                history["best_val_loss"] = avg_val_loss
                history["best_epoch"] = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Log progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(
                    f"  Epoch {epoch:>3d}/{epochs} | "
                    f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
                    f"LR: {current_lr:.2e} | Best: {history['best_val_loss']:.6f} (ep {history['best_epoch']})"
                )

            # Early stopping
            if early_stop.step(avg_val_loss):
                logger.info(f"  Early stopping at epoch {epoch} (patience: {self.patience})")
                break

        elapsed = time.time() - start_time
        history["training_time_sec"] = round(elapsed, 2)
        history["final_epoch"] = epoch

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
            model = model.to(self.device)

        # Save best model
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = save_path / f"{model_name}_{timestamp}_best.pth"
            torch.save({
                "state_dict": model.state_dict(),
                "config": getattr(model, "config", {}),
                "history": history,
                "model_name": model_name,
            }, checkpoint_path)
            history["checkpoint_path"] = str(checkpoint_path)
            logger.info(f"  Model saved: {checkpoint_path}")

        logger.info(
            f"{model_name} trained in {elapsed:.1f}s | "
            f"Best val loss: {history['best_val_loss']:.6f} (epoch {history['best_epoch']})"
        )

        return history

    def quick_train(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 5,
    ) -> Tuple[bool, float]:
        """
        Quick sanity check: train for a few epochs to verify convergence.

        Returns:
            Tuple of (converging: bool, final_loss: float)
        """
        model = model.to(self.device)
        train_loader = self._make_loader(X_train[:1000], y_train[:1000], shuffle=True)
        val_loader = self._make_loader(X_val[:500], y_val[:500], shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        losses = []
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred.squeeze(), y_batch.squeeze())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n += 1
            losses.append(epoch_loss / max(n, 1))

        # Check if loss is decreasing
        converging = len(losses) >= 2 and losses[-1] < losses[0]
        return converging, losses[-1]

    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool
    ) -> DataLoader:
        """Create a DataLoader from numpy arrays."""
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y)

        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(1)
        if X_t.ndim == 2:
            X_t = X_t.unsqueeze(2)  # Add feature dimension

        dataset = TensorDataset(X_t, y_t)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Use 0 for Windows compatibility
            pin_memory=(self.device == "cuda"),
        )

"""
Training infrastructure for DKO and baseline models.

Implements:
- Training loop with early stopping
- AdamW optimizer with cosine annealing
- Mixed precision training (FP16)
- Gradient clipping
- Checkpointing best models
- W&B + local logging

Research plan specifications:
- Optimizer: AdamW with base_lr=1e-4, weight_decay=1e-5
- Scheduler: Cosine annealing from 1e-4 to 1e-6
- Max epochs: 300
- Early stopping: patience=30 on validation loss
- Batch size: 32
- Gradient clipping: max_norm=1.0
- Mixed precision: FP16 enabled
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger("trainer")


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.

    Research plan specification: patience=30 epochs
    """

    def __init__(
        self,
        patience: int = 30,
        min_delta: float = 0.0,
        mode: str = 'min',
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy/AUC
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation metric

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class Trainer:
    """
    Trainer for DKO and baseline models.

    Specifications from research plan:
    - AdamW optimizer (lr=1e-4, weight_decay=1e-5)
    - Cosine annealing to 1e-6
    - Early stopping (patience=30)
    - Gradient clipping (max_norm=1.0)
    - Mixed precision (FP16)
    - Checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        task: str = 'regression',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 300,
        early_stopping_patience: int = 30,
        gradient_clip_max_norm: float = 1.0,
        device: Optional[str] = None,
        use_mixed_precision: bool = True,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
        verbose: bool = True,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            task: 'regression' or 'classification'
            learning_rate: Base learning rate (default: 1e-4)
            weight_decay: Weight decay for AdamW (default: 1e-5)
            max_epochs: Maximum number of epochs (default: 300)
            early_stopping_patience: Patience for early stopping (default: 30)
            gradient_clip_max_norm: Max norm for gradient clipping (default: 1.0)
            device: Device to train on (auto-detect if None)
            use_mixed_precision: Whether to use FP16 training (default: True)
            checkpoint_dir: Directory to save checkpoints
            use_wandb: Whether to use W&B logging
            wandb_project: W&B project name
            wandb_run_name: W&B run name
            wandb_config: Additional W&B config
            verbose: Whether to print progress
        """
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = model.to(device)
        self.task = task
        self.device = device
        self.max_epochs = max_epochs
        self.gradient_clip_max_norm = gradient_clip_max_norm
        self.use_mixed_precision = use_mixed_precision and device == 'cuda'
        self.verbose = verbose

        # Optimizer: AdamW as specified in research plan
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Scheduler: Cosine annealing to eta_min=1e-6 as specified
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
            eta_min=1e-6
        )

        # Loss function
        if task == 'regression':
            self.criterion = nn.MSELoss()
        elif task == 'classification':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown task: {task}. Use 'regression' or 'classification'")

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='min'  # Minimize validation loss
        )

        # Mixed precision scaler
        self.scaler = GradScaler('cuda') if self.use_mixed_precision else None

        # Checkpointing
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

        # W&B logging
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_initialized = False
        if self.use_wandb:
            try:
                config = {
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'max_epochs': max_epochs,
                    'early_stopping_patience': early_stopping_patience,
                    'task': task,
                    'device': device,
                    'mixed_precision': use_mixed_precision,
                }
                if wandb_config:
                    config.update(wandb_config)

                wandb.init(
                    project=wandb_project or "dko-training",
                    name=wandb_run_name,
                    config=config,
                    reinit=True,
                )
                self.wandb_initialized = True
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
                self.use_wandb = False

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }

        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.current_epoch = 0

    def _get_batch_data(self, batch: Dict) -> Tuple:
        """
        Extract data from batch, handling both DKO and baseline formats.

        Args:
            batch: Batch dictionary

        Returns:
            Tuple of (data_dict, labels) where data_dict contains model inputs
        """
        labels = batch.get('label', batch.get('labels'))
        if labels is not None:
            labels = labels.to(self.device)

        # Check for DKO format (mu, sigma)
        if 'mu' in batch and 'sigma' in batch:
            return {
                'format': 'dko',
                'mu': batch['mu'].to(self.device),
                'sigma': batch['sigma'].to(self.device),
            }, labels

        # Check for baseline format with conformer features
        if 'features' in batch:
            mask = batch.get('mask')
            if mask is not None:
                mask = mask.to(self.device)

            # Check for Boltzmann weights (DeepSets)
            weights = batch.get('weights', batch.get('boltzmann_weights'))
            if weights is not None:
                weights = weights.to(self.device)

            return {
                'format': 'baseline',
                'features': batch['features'].to(self.device),
                'mask': mask,
                'weights': weights,
            }, labels

        raise ValueError("Unknown batch format. Expected 'mu'/'sigma' or 'features'")

    def _forward_pass(
        self,
        data: Dict,
        fit_pca: bool = False
    ) -> torch.Tensor:
        """
        Perform forward pass for any model format.

        Args:
            data: Data dictionary from _get_batch_data
            fit_pca: Whether to fit PCA (for DKO)

        Returns:
            Model predictions
        """
        if data['format'] == 'dko':
            # DKO model with mu, sigma
            outputs = self.model(
                data['mu'],
                data['sigma'],
                fit_pca=fit_pca
            )
        else:
            # Baseline model with conformer features
            features = data['features']
            mask = data.get('mask')
            weights = data.get('weights')

            # Handle different baseline model signatures
            if weights is not None:
                # DeepSets with Boltzmann weights
                outputs = self.model(features, weights, mask=mask)
            elif mask is not None:
                outputs = self.model(features, mask=mask)
            else:
                outputs = self.model(features)

            # Handle tuple output (output, attention_info)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

        return outputs

    def train_epoch(
        self,
        train_loader: DataLoader,
        fit_pca: bool = False,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            fit_pca: Whether to fit PCA (first epoch for DKO)

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        # Progress bar
        if self.verbose and TQDM_AVAILABLE:
            pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        else:
            pbar = train_loader

        for batch_idx, batch in enumerate(pbar):
            # Extract data
            data, labels = self._get_batch_data(batch)

            # Fit PCA on first batch of first epoch (for DKO)
            do_fit_pca = fit_pca and batch_idx == 0

            # Forward pass with optional mixed precision
            self.optimizer.zero_grad()

            if self.use_mixed_precision:
                with autocast('cuda'):
                    outputs = self._forward_pass(data, fit_pca=do_fit_pca)
                    loss = self.criterion(outputs.squeeze(), labels.squeeze())

                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_max_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self._forward_pass(data, fit_pca=do_fit_pca)
                loss = self.criterion(outputs.squeeze(), labels.squeeze())

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_max_norm
                )
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if self.verbose and TQDM_AVAILABLE:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss}

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validate on validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            # Extract data
            data, labels = self._get_batch_data(batch)

            # Forward pass
            outputs = self._forward_pass(data, fit_pca=False)
            loss = self.criterion(outputs.squeeze(), labels.squeeze())

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Training history
        """
        if self.verbose:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"Starting training on {self.device}")
            print(f"Model parameters: {n_params:,}")
            print(f"Max epochs: {self.max_epochs}")

        # Reset early stopping
        self.early_stopping.reset()

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train (fit PCA only on first epoch)
            fit_pca = (epoch == 0)
            train_metrics = self.train_epoch(train_loader, fit_pca=fit_pca)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step()

            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['learning_rate'].append(current_lr)

            epoch_time = time.time() - epoch_start

            if self.verbose:
                print(f"Epoch {epoch+1}/{self.max_epochs} ({epoch_time:.1f}s) - "
                      f"train_loss: {train_metrics['loss']:.4f}, "
                      f"val_loss: {val_metrics['loss']:.4f}, "
                      f"lr: {current_lr:.2e}")

            # W&B logging
            if self.use_wandb and self.wandb_initialized:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'learning_rate': current_lr,
                })

            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch + 1
                if self.checkpoint_dir:
                    self.save_checkpoint('best_model.pt')
                if self.verbose:
                    print(f"  -> New best model (val_loss: {self.best_val_loss:.4f})")

            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                if self.verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(f"Best epoch: {self.best_epoch}, Best val_loss: {self.best_val_loss:.4f}")
                break

        # Load best model
        if self.checkpoint_dir and (self.checkpoint_dir / 'best_model.pt').exists():
            self.load_checkpoint('best_model.pt')

        return self.history

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'history': self.history,
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename: str) -> bool:
        """
        Load model checkpoint.

        Args:
            filename: Checkpoint filename

        Returns:
            True if loaded successfully
        """
        if self.checkpoint_dir is None:
            return False

        checkpoint_path = self.checkpoint_dir / filename
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint {checkpoint_path} not found")
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        self.history = checkpoint.get('history', self.history)
        self.current_epoch = checkpoint.get('epoch', 0)

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if self.verbose:
            print(f"Loaded checkpoint from epoch {self.best_epoch}")

        return True

    def finish(self) -> None:
        """Cleanup after training (close W&B, etc.)."""
        if self.use_wandb and self.wandb_initialized:
            wandb.finish()


def create_trainer(
    model: nn.Module,
    config: Dict[str, Any],
    **kwargs,
) -> Trainer:
    """
    Factory function to create a Trainer from config.

    Args:
        model: Model to train
        config: Configuration dictionary with training parameters
        **kwargs: Additional arguments to override config

    Returns:
        Configured Trainer instance
    """
    # Extract config values with defaults matching research plan
    training_config = config.get('training', config)

    return Trainer(
        model=model,
        task=kwargs.get('task', training_config.get('task', 'regression')),
        learning_rate=kwargs.get('learning_rate', training_config.get('base_learning_rate', 1e-4)),
        weight_decay=kwargs.get('weight_decay', training_config.get('weight_decay', 1e-5)),
        max_epochs=kwargs.get('max_epochs', training_config.get('max_epochs', 300)),
        early_stopping_patience=kwargs.get(
            'early_stopping_patience',
            training_config.get('early_stopping_patience', 30)
        ),
        gradient_clip_max_norm=kwargs.get(
            'gradient_clip_max_norm',
            training_config.get('gradient_clip_max_norm', 1.0)
        ),
        device=kwargs.get('device', training_config.get('device')),
        use_mixed_precision=kwargs.get(
            'use_mixed_precision',
            training_config.get('mixed_precision', True)
        ),
        checkpoint_dir=kwargs.get('checkpoint_dir', training_config.get('checkpoint_dir')),
        use_wandb=kwargs.get('use_wandb', training_config.get('use_wandb', False)),
        wandb_project=kwargs.get('wandb_project', training_config.get('wandb_project')),
        wandb_run_name=kwargs.get('wandb_run_name', training_config.get('wandb_run_name')),
        verbose=kwargs.get('verbose', training_config.get('verbose', True)),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: str = "cuda",
    experiment_name: str = "experiment",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train a model using the Trainer class.

    Convenience wrapper function for training models.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dictionary
        device: Device to train on
        experiment_name: Name for the experiment

    Returns:
        Tuple of (trained model, training results dict)
    """
    # Create trainer from config
    trainer = Trainer(
        model=model,
        task=config.get('task_type', config.get('task', 'regression')),
        learning_rate=config.get('learning_rate', config.get('base_learning_rate', 1e-4)),
        weight_decay=config.get('weight_decay', 1e-5),
        max_epochs=config.get('max_epochs', 300),
        early_stopping_patience=config.get('early_stopping_patience', 30),
        gradient_clip_max_norm=config.get('gradient_clip_max_norm', 1.0),
        device=device,
        use_mixed_precision=config.get('mixed_precision', True),
        checkpoint_dir=config.get('checkpoint_dir'),
        use_wandb=config.get('use_wandb', False),
        wandb_project=config.get('wandb_project'),
        wandb_run_name=experiment_name,
        verbose=config.get('verbose', True),
    )

    # Train
    history = trainer.fit(train_loader, val_loader)

    # Prepare results
    results = {
        'history': history,
        'best_epoch': trainer.best_epoch,
        'best_val_loss': trainer.best_val_loss,
        'total_epochs': len(history.get('train_loss', [])),
    }

    return model, results


if __name__ == "__main__":
    # Test trainer
    print("Testing Trainer...")

    from dko.models.dko import DKO
    from torch.utils.data import TensorDataset

    # Create dummy data
    batch_size = 16
    n_samples = 100
    D = 50

    mu_train = torch.randn(n_samples, D)
    sigma_train = torch.randn(n_samples, D, D)
    sigma_train = torch.bmm(sigma_train, sigma_train.transpose(1, 2))
    labels_train = torch.randn(n_samples, 1)

    mu_val = torch.randn(50, D)
    sigma_val = torch.randn(50, D, D)
    sigma_val = torch.bmm(sigma_val, sigma_val.transpose(1, 2))
    labels_val = torch.randn(50, 1)

    # Create datasets
    train_dataset = TensorDataset(mu_train, sigma_train, labels_train)
    val_dataset = TensorDataset(mu_val, sigma_val, labels_val)

    # Custom collate function
    def collate_fn(batch):
        mu, sigma, labels = zip(*batch)
        return {
            'mu': torch.stack(mu),
            'sigma': torch.stack(sigma),
            'label': torch.stack(labels),
        }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Create model
    model = DKO(feature_dim=D, output_dim=1, verbose=False)

    # Create trainer
    trainer = Trainer(
        model=model,
        task='regression',
        max_epochs=5,
        early_stopping_patience=3,
        use_wandb=False,
        checkpoint_dir=Path('./checkpoints_test'),
        verbose=True,
    )

    # Train
    history = trainer.fit(train_loader, val_loader)

    print("\n[OK] Trainer test passed!")
    print(f"  Training history: {len(history['train_loss'])} epochs")
    print(f"  Best val loss: {trainer.best_val_loss:.4f} at epoch {trainer.best_epoch}")

    # Cleanup test directory
    import shutil
    if Path('./checkpoints_test').exists():
        shutil.rmtree('./checkpoints_test')

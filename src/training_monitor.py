"""
Training monitoring utilities for debugging CNN convergence issues.

This module provides tools to monitor gradients, loss progression, and
feature representations during CNN training for the 2D SNPE pipeline.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from pathlib import Path


class TrainingMonitor:
    """Monitor training progress and CNN behavior for debugging."""

    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize training monitor.

        Parameters:
        -----------
        log_dir : str, optional
            Directory to save monitoring logs and plots
        """
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_norms = []
        self.losses = []
        self.feature_stats = []
        self.epoch_count = 0

    def log_gradients(self, model: nn.Module, epoch: int):
        """
        Log gradient norms for all model parameters.

        Parameters:
        -----------
        model : nn.Module
            Model to monitor
        epoch : int
            Current epoch number
        """
        total_norm = 0.0
        layer_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                layer_norms[name] = param_norm.item()

        total_norm = total_norm ** (1. / 2)

        gradient_info = {
            'epoch': epoch,
            'total_norm': total_norm,
            'layer_norms': layer_norms
        }

        self.gradient_norms.append(gradient_info)

        # Log critical gradient issues
        if total_norm > 10.0:
            print(f"⚠️  Warning: Large gradient norm at epoch {epoch}: {total_norm:.4f}")
        elif total_norm < 1e-6:
            print(f"⚠️  Warning: Vanishing gradients at epoch {epoch}: {total_norm:.6f}")

    def log_loss(self, loss: float, epoch: int, validation_loss: Optional[float] = None):
        """
        Log training and validation losses.

        Parameters:
        -----------
        loss : float
            Training loss
        epoch : int
            Current epoch
        validation_loss : float, optional
            Validation loss
        """
        loss_info = {
            'epoch': epoch,
            'train_loss': loss,
            'val_loss': validation_loss
        }
        self.losses.append(loss_info)

    def log_feature_stats(self, features: torch.Tensor, layer_name: str, epoch: int):
        """
        Log statistics of feature activations.

        Parameters:
        -----------
        features : torch.Tensor
            Feature tensor to analyze
        layer_name : str
            Name of the layer
        epoch : int
            Current epoch
        """
        with torch.no_grad():
            stats = {
                'epoch': epoch,
                'layer_name': layer_name,
                'mean': features.mean().item(),
                'std': features.std().item(),
                'min': features.min().item(),
                'max': features.max().item(),
                'zero_fraction': (features == 0).float().mean().item()
            }

        self.feature_stats.append(stats)

        # Warn about dead neurons
        if stats['zero_fraction'] > 0.5:
            print(f"⚠️  Warning: {stats['zero_fraction']:.1%} dead neurons in {layer_name} at epoch {epoch}")

    def plot_training_progress(self):
        """Create plots showing training progress and issues."""
        if not self.log_dir:
            return

        # Plot gradient norms
        if self.gradient_norms:
            epochs = [g['epoch'] for g in self.gradient_norms]
            total_norms = [g['total_norm'] for g in self.gradient_norms]

            plt.figure(figsize=(10, 6))
            plt.subplot(2, 2, 1)
            plt.plot(epochs, total_norms)
            plt.xlabel('Epoch')
            plt.ylabel('Gradient Norm')
            plt.title('Total Gradient Norm')
            plt.yscale('log')

        # Plot losses
        if self.losses:
            epochs = [l['epoch'] for l in self.losses]
            train_losses = [l['train_loss'] for l in self.losses]
            val_losses = [l['val_loss'] for l in self.losses if l['val_loss'] is not None]

            plt.subplot(2, 2, 2)
            plt.plot(epochs, train_losses, label='Training Loss')
            if val_losses and len(val_losses) == len(epochs):
                plt.plot(epochs, val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.yscale('log')

        # Plot feature statistics
        if self.feature_stats:
            plt.subplot(2, 2, 3)
            for layer_name in set(s['layer_name'] for s in self.feature_stats):
                layer_stats = [s for s in self.feature_stats if s['layer_name'] == layer_name]
                epochs = [s['epoch'] for s in layer_stats]
                zero_fractions = [s['zero_fraction'] for s in layer_stats]
                plt.plot(epochs, zero_fractions, label=f'{layer_name} dead neurons')
            plt.xlabel('Epoch')
            plt.ylabel('Fraction of Dead Neurons')
            plt.title('Dead Neuron Analysis')
            plt.legend()

        plt.tight_layout()
        plt.savefig(self.log_dir / "training_monitor.png", dpi=150, bbox_inches='tight')
        plt.close()

    def save_logs(self):
        """Save monitoring logs to disk."""
        if not self.log_dir:
            return

        import pickle

        logs = {
            'gradient_norms': self.gradient_norms,
            'losses': self.losses,
            'feature_stats': self.feature_stats
        }

        with open(self.log_dir / "training_logs.pkl", 'wb') as f:
            pickle.dump(logs, f)

    def check_convergence_issues(self) -> List[str]:
        """
        Analyze logs for common convergence issues.

        Returns:
        --------
        List[str]
            List of detected issues
        """
        issues = []

        # Check gradient issues
        if self.gradient_norms:
            recent_norms = [g['total_norm'] for g in self.gradient_norms[-10:]]
            if np.mean(recent_norms) > 10.0:
                issues.append("Exploding gradients detected (avg norm > 10)")
            elif np.mean(recent_norms) < 1e-5:
                issues.append("Vanishing gradients detected (avg norm < 1e-5)")

        # Check loss progression
        if len(self.losses) > 20:
            recent_losses = [l['train_loss'] for l in self.losses[-20:]]
            if abs(recent_losses[-1] - recent_losses[-10]) < 1e-6:
                issues.append("Training loss plateaued - no improvement")

        # Check dead neurons
        if self.feature_stats:
            recent_stats = self.feature_stats[-10:]
            avg_dead_fraction = np.mean([s['zero_fraction'] for s in recent_stats])
            if avg_dead_fraction > 0.3:
                issues.append(f"High dead neuron rate: {avg_dead_fraction:.1%}")

        return issues


def create_cnn_hook_monitor(model: nn.Module, monitor: TrainingMonitor, epoch: int):
    """
    Create forward hooks to monitor CNN layer activations.

    Parameters:
    -----------
    model : nn.Module
        CNN model to monitor
    monitor : TrainingMonitor
        Monitor instance
    epoch : int
        Current epoch
    """
    def create_hook(layer_name: str):
        def hook(module, input, output):
            monitor.log_feature_stats(output, layer_name, epoch)
        return hook

    hooks = []

    # Add hooks to key CNN layers
    if hasattr(model, 'conv1'):
        hooks.append(model.conv1.register_forward_hook(create_hook('conv1')))
    if hasattr(model, 'conv2'):
        hooks.append(model.conv2.register_forward_hook(create_hook('conv2')))
    if hasattr(model, 'conv3'):
        hooks.append(model.conv3.register_forward_hook(create_hook('conv3')))
    if hasattr(model, 'conv4'):
        hooks.append(model.conv4.register_forward_hook(create_hook('conv4')))

    return hooks


def remove_hooks(hooks: List):
    """Remove forward hooks."""
    for hook in hooks:
        hook.remove()
"""
Factory for creating and configuring optimizers.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau


class OptimizerConfig:
    """Configuration for optimizer and learning rate scheduling."""

    def __init__(
        self,
        optimizer_type: str = "adamw",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        gradient_clip_val: float = 1.0,
        # Learning rate scheduler
        scheduler_type: Optional[str] = None,
        scheduler_params: Optional[Dict[str, Any]] = None,
    ):
        self.optimizer_type = optimizer_type.lower()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.gradient_clip_val = gradient_clip_val
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params or {}


def create_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
    checkpoint_path: Optional[Path] = None,
) -> torch.optim.Optimizer:
    """
    Create an optimizer for the model.

    Args:
        model: PyTorch model
        config: Optimizer configuration
        checkpoint_path: Optional path to checkpoint for loading optimizer state

    Returns:
        Configured optimizer
    """
    # Get model parameters
    parameters = model.parameters()

    # Create optimizer based on type
    if config.optimizer_type == "adamw":
        optimizer = AdamW(
            parameters,
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer_type == "adam":
        optimizer = Adam(
            parameters,
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer_type == "sgd":
        optimizer = SGD(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.scheduler_params.get("momentum", 0.9),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")

    # Load optimizer state from checkpoint if provided
    if checkpoint_path and checkpoint_path.exists():
        print(f"Loading optimizer state from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Loaded optimizer from update step {checkpoint.get('update_step', 'unknown')}")

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: OptimizerConfig,
    num_updates: Optional[int] = None,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create a learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        config: Optimizer configuration
        num_updates: Total number of training updates (for cosine annealing)

    Returns:
        Learning rate scheduler or None if not configured
    """
    if not config.scheduler_type:
        return None

    scheduler_type = config.scheduler_type.lower()
    params = config.scheduler_params

    if scheduler_type == "cosine":
        if num_updates is None:
            raise ValueError("num_updates is required for cosine annealing scheduler")
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_updates,
            eta_min=params.get("eta_min", 0),
        )
    elif scheduler_type == "step":
        scheduler = StepLR(
            optimizer,
            step_size=params.get("step_size", 1000),
            gamma=params.get("gamma", 0.1),
        )
    elif scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=params.get("mode", "min"),
            factor=params.get("factor", 0.5),
            patience=params.get("patience", 10),
            min_lr=params.get("min_lr", 1e-6),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


def apply_gradient_clipping(
    model: nn.Module,
    config: OptimizerConfig,
) -> float:
    """
    Apply gradient clipping to model parameters.

    Args:
        model: PyTorch model
        config: Optimizer configuration

    Returns:
        Total gradient norm before clipping
    """
    if config.gradient_clip_val <= 0:
        # Calculate norm without clipping
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm**0.5

    # Apply gradient clipping
    return torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val).item()


def create_default_optimizer_config() -> OptimizerConfig:
    """Create default optimizer configuration for overfitting experiment."""
    return OptimizerConfig(
        optimizer_type="adamw",
        learning_rate=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        gradient_clip_val=1.0,
        scheduler_type=None,  # No scheduler for overfitting
        scheduler_params=None,
    )


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float,
    no_decay_keywords: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with selective weight decay.

    Args:
        model: PyTorch model
        weight_decay: Weight decay value
        no_decay_keywords: List of keywords for parameters that shouldn't have weight decay
            (default: ["bias", "norm", "embedding"])

    Returns:
        List of parameter groups for optimizer
    """
    if no_decay_keywords is None:
        no_decay_keywords = ["bias", "norm", "embedding"]

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter should have weight decay
        if any(keyword in name.lower() for keyword in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

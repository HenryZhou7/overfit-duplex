"""
Factory for creating and configuring models.
"""

from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from overfit_trial.model import MachOverfitModel


class ModelConfig:
    """Configuration for model creation."""

    def __init__(
        self,
        num_quantizers: int = 32,
        mimi_audio_embed_dir: Optional[str] = None,
        checkpoint_path: Optional[Path] = None,
    ):
        self.num_quantizers = num_quantizers
        self.mimi_audio_embed_dir = mimi_audio_embed_dir
        self.checkpoint_path = checkpoint_path


def create_model(config: ModelConfig) -> MachOverfitModel:
    """
    Create and initialize the model.

    Args:
        config: Model configuration

    Returns:
        Initialized MachOverfitModel
    """
    model = MachOverfitModel(num_quantizers=config.num_quantizers, mimi_audio_embed_dir=config.mimi_audio_embed_dir)

    if config.checkpoint_path and config.checkpoint_path.exists():
        print(f"Loading model checkpoint from {config.checkpoint_path}")
        checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from update step {checkpoint.get('update_step', 'unknown')}")

    return model


def create_default_model_config() -> ModelConfig:
    """Create default model configuration."""
    return ModelConfig(
        num_quantizers=32,
        mimi_audio_embed_dir=None,
        checkpoint_path=None,
    )


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about the model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "total_parameters_m": total_params / 1e6,
        "trainable_parameters_m": trainable_params / 1e6,
    }


def print_model_summary(model: nn.Module) -> None:
    """Print a summary of the model."""
    info = get_model_info(model)

    print("=" * 50)
    print("Model Summary")
    print("=" * 50)
    print(f"Total parameters: {info['total_parameters']:,} ({info['total_parameters_m']:.2f}M)")
    print(f"Trainable parameters: {info['trainable_parameters']:,} ({info['trainable_parameters_m']:.2f}M)")
    if info["non_trainable_parameters"] > 0:
        print(f"Non-trainable parameters: {info['non_trainable_parameters']:,}")
    print("=" * 50)

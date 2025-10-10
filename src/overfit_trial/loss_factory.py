"""
Factory for creating and computing losses for duplex model training.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


class LossConfig:
    """Configuration for loss computation."""

    def __init__(
        self,
        semantic_loss_weight: float = 1.0,
        acoustic_loss_weight: float = 1.0,
        reduction: str = "mean",
    ):
        self.semantic_loss_weight = semantic_loss_weight
        self.acoustic_loss_weight = acoustic_loss_weight
        self.reduction = reduction


def compute_semantic_loss(
    semantic_logits: torch.Tensor,
    semantic_targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cross-entropy loss for semantic tokens.

    Args:
        semantic_logits: Predicted logits of shape (batch_size, seq_len-1, vocab_size)
        semantic_targets: Target tokens of shape (batch_size, seq_len-1)
        mask: Boolean mask of shape (batch_size, seq_len-1)

    Returns:
        Scalar loss tensor
    """
    batch_size = semantic_logits.shape[0]

    # Reshape for cross entropy
    semantic_loss = F.cross_entropy(
        semantic_logits.reshape(-1, semantic_logits.size(-1)), semantic_targets.reshape(-1), reduction="none"
    )

    # Reshape back and apply mask
    semantic_loss = semantic_loss.reshape(batch_size, -1)
    masked_loss = semantic_loss * mask

    # Average over valid positions
    if mask.sum() > 0:
        return masked_loss.sum() / mask.sum()
    else:
        return torch.tensor(0.0, device=semantic_logits.device)


def compute_acoustic_loss(
    audio_logits: torch.Tensor,
    acoustic_targets: torch.Tensor,
    mask: torch.Tensor,
    num_quantizers: int,
) -> torch.Tensor:
    """
    Compute cross-entropy loss for acoustic tokens.

    Args:
        audio_logits: Predicted logits of shape (batch_size, seq_len-1, num_quantizers-1, vocab_size)
        acoustic_targets: Target tokens of shape (batch_size, seq_len-1, num_quantizers-1)
        mask: Boolean mask of shape (batch_size, seq_len-1)
        num_quantizers: Total number of quantizers

    Returns:
        Scalar loss tensor
    """
    batch_size = audio_logits.shape[0]

    # Reshape for cross entropy
    audio_logits_flat = audio_logits.reshape(-1, audio_logits.size(-1))
    acoustic_targets_flat = acoustic_targets.reshape(-1)

    acoustic_loss = F.cross_entropy(audio_logits_flat, acoustic_targets_flat, reduction="none")

    # Reshape and apply mask
    acoustic_loss = acoustic_loss.reshape(batch_size, -1, num_quantizers - 1)
    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, num_quantizers - 1)
    masked_loss = acoustic_loss * mask_expanded

    # Average over valid positions
    if mask_expanded.sum() > 0:
        return masked_loss.sum() / mask_expanded.sum()
    else:
        return torch.tensor(0.0, device=audio_logits.device)


def calculate_losses(
    model_output: Tuple[torch.Tensor, torch.Tensor],
    target_codes: torch.Tensor,
    lengths: torch.Tensor,
    num_quantizers: int,
    config: LossConfig,
) -> Dict[str, torch.Tensor]:
    """
    Calculate semantic and acoustic losses with proper teacher forcing.

    Args:
        model_output: Tuple of (c0_logit, audio_logits)
            - c0_logit: (batch_size, seq_len, vocab_size) for semantic tokens
            - audio_logits: (batch_size, seq_len, num_quantizers-1, vocab_size) for acoustic tokens
        target_codes: (batch_size, num_quantizers, seq_len) target RVQ codes
        lengths: (batch_size,) actual sequence lengths
        num_quantizers: Total number of quantizers
        config: Loss configuration

    Returns:
        Dictionary with:
            - 'semantic_loss': Scalar loss for semantic tokens
            - 'acoustic_loss': Scalar loss for acoustic tokens
            - 'total_loss': Weighted sum of semantic and acoustic losses
    """
    c0_logit, audio_logits = model_output
    batch_size, seq_len = target_codes.shape[0], target_codes.shape[2]

    # Prepare targets
    # Semantic tokens are the first quantizer level (index 0)
    semantic_targets = target_codes[:, 0, :]  # (batch_size, seq_len)

    # Acoustic tokens are quantizer levels 1 through num_quantizers-1
    acoustic_targets = target_codes[:, 1:, :]  # (batch_size, num_quantizers-1, seq_len)
    acoustic_targets = acoustic_targets.transpose(1, 2)  # (batch_size, seq_len, num_quantizers-1)

    # Create length masks
    mask = torch.arange(seq_len, device=target_codes.device).expand(batch_size, seq_len)
    mask = mask < lengths.unsqueeze(1)  # (batch_size, seq_len)

    # Shift for next-token prediction (teacher forcing)
    semantic_logits_shifted = c0_logit[:, :-1, :]  # (batch_size, seq_len-1, vocab_size)
    semantic_targets_shifted = semantic_targets[:, 1:]  # (batch_size, seq_len-1)
    mask_shifted = mask[:, 1:]  # (batch_size, seq_len-1)

    audio_logits_shifted = audio_logits[:, :-1, :, :]  # (batch_size, seq_len-1, num_quantizers-1, vocab_size)
    acoustic_targets_shifted = acoustic_targets[:, 1:, :]  # (batch_size, seq_len-1, num_quantizers-1)

    # Compute individual losses
    semantic_loss = compute_semantic_loss(semantic_logits_shifted, semantic_targets_shifted, mask_shifted)

    acoustic_loss = compute_acoustic_loss(audio_logits_shifted, acoustic_targets_shifted, mask_shifted, num_quantizers)

    # Compute weighted total loss
    total_loss = config.semantic_loss_weight * semantic_loss + config.acoustic_loss_weight * acoustic_loss

    return {
        "semantic_loss": semantic_loss,
        "acoustic_loss": acoustic_loss,
        "total_loss": total_loss,
    }


def create_default_loss_config() -> LossConfig:
    """Create default loss configuration."""
    return LossConfig(
        semantic_loss_weight=1.0,
        acoustic_loss_weight=1.0,
        reduction="mean",
    )

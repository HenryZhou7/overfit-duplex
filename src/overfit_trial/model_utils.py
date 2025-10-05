"""
Utility functions used in https://github.com/SesameAILabs/csm/blob/main/models.py
"""

import torchtune
from torchtune.models import llama3_2
import torch
import torch.nn as nn
from pathlib import Path
import safetensors.torch


__all__ = [
    "prepare_transformer",
    "FLAVORS",
    "load_llama_weights",
    "MLP",
]


def llama3_2_1B() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_100M() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=2048,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


FLAVORS = {
    "llama-1B": llama3_2_1B,
    "llama-100M": llama3_2_100M,
}


def prepare_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


############################################################
# Henry Custom Functions
############################################################
MODEL_TO_TENSORPATH = {
    "llama-1B": Path.home()
    / "model_weights"
    / "Llama-3.2-1B-Instruct"
    / "model.safetensors",
}


def convert_hf_to_torchtune(hf_state_dict):
    """Convert HuggingFace Llama state dict keys to torchtune format."""
    torchtune_state_dict = {}

    key_mapping = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.scale",
    }

    for i in range(16):  # 16 layers for Llama-3.2-1B
        key_mapping.update(
            {
                f"model.layers.{i}.self_attn.q_proj.weight": f"layers.{i}.attn.q_proj.weight",
                f"model.layers.{i}.self_attn.k_proj.weight": f"layers.{i}.attn.k_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight": f"layers.{i}.attn.v_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight": f"layers.{i}.attn.output_proj.weight",
                f"model.layers.{i}.mlp.gate_proj.weight": f"layers.{i}.mlp.w1.weight",
                f"model.layers.{i}.mlp.down_proj.weight": f"layers.{i}.mlp.w2.weight",
                f"model.layers.{i}.mlp.up_proj.weight": f"layers.{i}.mlp.w3.weight",
                f"model.layers.{i}.input_layernorm.weight": f"layers.{i}.sa_norm.scale",
                f"model.layers.{i}.post_attention_layernorm.weight": f"layers.{i}.mlp_norm.scale",
            }
        )

    for hf_key, value in hf_state_dict.items():
        if hf_key in key_mapping:
            torchtune_key = key_mapping[hf_key]
            torchtune_state_dict[torchtune_key] = value

    return torchtune_state_dict


def load_llama_weights(model_id: str):
    """
    Load the llama model weights from the downloaded tensors.
    """
    model = FLAVORS[model_id]()
    hf_state_dict = safetensors.torch.load_file(MODEL_TO_TENSORPATH[model_id])
    torchtune_state_dict = convert_hf_to_torchtune(hf_state_dict)
    model.load_state_dict(torchtune_state_dict)
    return model


class MLP(nn.Module):
    """
    General purpose Multi-Layer Perceptron.

    Args:
        layers: List of integers specifying the number of neurons in each layer
                from input to output (e.g., [784, 256, 128, 10])
        non_linear: Activation function to apply between layers
        final_non_linear: Optional activation function to apply after the final layer
        dropout: Dropout probability (0.0 means no dropout)
        normalization: Whether to apply LayerNorm after each hidden layer
    """

    def __init__(
        self,
        layers: list[int],
        non_linear: nn.Module,
        final_non_linear: nn.Module | None = None,
        dropout: float = 0.0,
        normalization: bool = False,
    ):
        super().__init__()

        if len(layers) < 2:
            raise ValueError(
                "layers must contain at least 2 elements (input and output)"
            )

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if normalization else None
        self.non_linear = non_linear
        self.final_non_linear = final_non_linear
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        # Build the network
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

            # Add normalization for all layers except the last one
            if normalization and i < len(layers) - 2:
                self.norms.append(nn.LayerNorm(layers[i + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, layers[0])

        Returns:
            Output tensor of shape (batch_size, layers[-1])
        """
        # Process all hidden layers
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)

            # Apply normalization if enabled
            if self.norms is not None:
                x = self.norms[i](x)

            # Apply activation
            x = self.non_linear(x)

            # Apply dropout
            if self.dropout is not None:
                x = self.dropout(x)

        # Process final layer
        x = self.layers[-1](x)

        # Apply final activation if specified
        if self.final_non_linear is not None:
            x = self.final_non_linear(x)

        return x

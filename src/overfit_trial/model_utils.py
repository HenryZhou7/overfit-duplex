"""
Utility functions used in https://github.com/SesameAILabs/csm/blob/main/models.py
"""

import torchtune
from torchtune.models import llama3_2
import torch.nn as nn
import safetensors.torch


__all__ = [
    "prepare_transformer",
    "FLAVORS",
    "load_llama_weights",
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
        embed_dim=1024,
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
    "llama-1B": "$HOME/model_weights/Llama-3.2-1B-Instruct/model.safetensors",
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


def load_llama_weights(model, model_id: str):
    """
    Load the llama model weights from the downloaded tensors.
    """

    hf_state_dict = safetensors.torch.load_file(MODEL_TO_TENSORPATH[model_id])
    torchtune_state_dict = convert_hf_to_torchtune(hf_state_dict)
    model.load_state_dict(torchtune_state_dict)
    return model

"""
Implementation of the model.
"""

import torch
import torch.nn as nn
from transformers import MimiModel

from .model_utils import (
    prepare_transformer,
    FLAVORS,
    load_llama_weights,
)


class MachOverfitModel(nn.Module):
    def __init__(self, num_quantizers: int = 8, input_dim: int = 512):
        super().__init__()

        self.input_dim = input_dim
        self.num_quantizers = num_quantizers
        self._init_quantizer(num_quantizers)
        self._init_backbone()
        self._init_decoder()

    def _init_quantizer(self, num_quantizers: int):
        """
        Quantizer to convert the RVQ codes to dense features.
        """
        mimi_model = MimiModel.from_pretrained(
            "kyutai/mimi", num_quantizers=num_quantizers
        )
        self.quantizer = mimi_model.quantizer
        self.quantizer.eval()

    def _init_backbone(self, model_id: str):
        """
        The backbone LLM of the model.
        """
        if model_id.startswith("llama"):
            self.backbone, embed_dim = prepare_transformer(FLAVORS["llama-1B"]())
            self.backbone = load_llama_weights(self.backbone, model_id)
        else:
            raise ValueError(f"Model ID {model_id} not supported")

        self.input_proj = nn.Linear(self.input_dim, embed_dim)

    def _init_decoder(self):
        """
        Decoder that takes the transformer outputs and predicts the RVQ codes.
        """
        self.decoder = prepare_transformer(FLAVORS["llama-100M"]())

    def forward(self, codes_c1: torch.Tensor, codes_c2: torch.Tensor):
        """
        Forward pass of the model.

        The forward pass consists of two steps
            1. takes in the two channels of RVQ codes and outputs the latent features
            2. the latent features are passed through the decoder to predict the RVQ codes
        """

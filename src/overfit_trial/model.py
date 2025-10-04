"""
Implementation of the model.
"""
# ruff: noqa: F722

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import MimiModel
from jaxtyping import Float

from overfit_trial.model_utils import (
    prepare_transformer,
    FLAVORS,
    load_llama_weights,
)

MODEL_ID = "llama-1B"


def _create_causal_mask(seq_len: int, device: torch.device = torch.device("cpu")):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


class MachOverfitModel(nn.Module):
    def __init__(self, num_quantizers: int = 8, input_dim: int = 512):
        super().__init__()

        self.input_dim = input_dim
        self.num_quantizers = num_quantizers

        self._init_quantizer(num_quantizers)
        self._init_backbone(MODEL_ID)
        self._init_decoder()

        #
        self.causal_mask = _create_causal_mask(self.backbone.max_seq_len)

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
            backbone = load_llama_weights(model_id)
            self.backbone, embed_dim = prepare_transformer(backbone)
        else:
            raise ValueError(f"Model ID {model_id} not supported")

        # Need to convert the two channel inputs into the input to the transformer.
        self.input_proj = nn.Linear(self.input_dim * 2, embed_dim)

    def _init_decoder(self):
        """
        Decoder that takes the transformer outputs and predicts the RVQ codes.
        """
        self.decoder = prepare_transformer(FLAVORS["llama-100M"]())

    def forward(self, codes_c1: torch.Tensor, codes_c2: torch.Tensor):
        """
        Forward pass of the model.
            codes_c1 / codes_c2: (1, num_quantizers, num_frames)
                num_frames is sampled at 12.5Hz.

        The forward pass consists of two steps
            1. takes in the two channels of RVQ codes and outputs the latent features
            2. the latent features are passed through the decoder to predict the RVQ codes
        """
        bs = codes_c1.shape[0]
        T = codes_c1.shape[2]
        latent_features: Float[torch.Tensor, "bs T embed_dim"] = (
            self._embed_audio_codes(codes_c1, codes_c2)
        )

        mask = self.causal_mask.unsqueeze(0).expand(bs, -1, -1)[:, :T, :T]

        # transformer forward pass
        transformer_outputs: Float[torch.Tensor, "bs T embed_dim"] = self.backbone(
            latent_features, mask=mask
        )

        return transformer_outputs

    def _embed_audio_codes(self, codes_c1: torch.Tensor, codes_c2: torch.Tensor):
        """
        Embed the audio codes into latent features.
        """
        with torch.no_grad():
            c1_latent = self.quantizer.decode(codes_c1)
            c2_latent = self.quantizer.decode(codes_c2)

        # combine the two channels.
        latent_features: Float[torch.Tensor, "bs D T"] = torch.cat(
            [c1_latent, c2_latent], dim=1
        )
        latent_features: Float[torch.Tensor, "bs T D"] = latent_features.transpose(1, 2)

        # project to match the embedding dimension of the transformer
        latent_features: Float[torch.Tensor, "bs T embed_dim"] = self.input_proj(
            latent_features
        )
        return latent_features

"""
Implementation of the model.
"""
# ruff: noqa: F722

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class TimeDependentLinear(nn.Module):
    """
    This class implements a time-dependent linear layer, which is used as a
    RVQ-level-specific linear layer for the audio decoder.

    Audio decoding is unrolled exact (num_quantizers - 1) times corresponding to
    the acoustic quantizers. Each of the quantizer is using a different linear layer.
    """

    def __init__(self, T, D, C):
        super().__init__()
        self.W = nn.Parameter(torch.empty(T, D, C))  # [T, D, C]
        self.b = nn.Parameter(torch.empty(T, C))  # [T, C]
        self.reset_parameters()

    def reset_parameters(self):
        # Follow nn.Linear initialization
        for t in range(self.W.size(0)):
            nn.init.kaiming_uniform_(self.W[t], a=math.sqrt(5))
        fan_in = self.W.size(1)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        # x: [B, T, D]
        # y: [B, T, C]
        return torch.einsum("btd,tdc->btc", x, self.W) + self.b

    def ith_forward(self, i: int, x: torch.Tensor):
        """
        Forward pass for the i-th quantizer.

        Args:
            i: int, the index of the quantizer
            x: torch.Tensor, the input tensor of shape [..., D]

        Returns:
            torch.Tensor, the output tensor of shape [..., C]
        """
        T, D, C = self.W.shape
        if not (0 <= i < T):
            raise IndexError(f"i={i} out of range for T={T}")
        if x.shape[-1] != D:
            raise ValueError(f"Expected x[..., {D}], got {x.shape}")

        # self.W[i]: [D, C] -> F.linear expects weight [C, D]
        W_i = self.W[i].transpose(0, 1)  # [C, D]
        b_i = self.b[i]  # [C]
        return F.linear(x, W_i, b_i)


class MachOverfitModel(nn.Module):
    def __init__(self, num_quantizers: int = 8):
        super().__init__()

        # Quantizer for audio embeddings
        self._init_quantizer(num_quantizers)
        self.num_quantizers = num_quantizers
        self.code_num, self.code_dim = self.quantizer.acoustic_residual_vector_quantizer.layers[0].codebook.embed.shape
        self.input_dim = self.code_dim

        # Model
        self._init_backbone(MODEL_ID)
        self._init_decoder()

        # Model training and inference
        causal_mask = _create_causal_mask(self.backbone.max_seq_len)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        # Misc
        self.num_parameters = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters: {self.num_parameters:,}")

    def _init_quantizer(self, num_quantizers: int):
        """
        Quantizer to convert the RVQ codes to dense features.
        """
        mimi_model = MimiModel.from_pretrained("kyutai/mimi", num_quantizers=num_quantizers)
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

        # Head to project the transformer outputs to the semantic token logits.
        self.c0_head = nn.Linear(
            self.backbone.layers[-1].attn.embed_dim,
            self.quantizer.semantic_residual_vector_quantizer.layers[0].codebook.codebook_size,
        )

    def _init_decoder(self):
        """
        Decoder that takes the transformer outputs and predicts the RVQ codes.
        """
        self.decoder, embed_dim = prepare_transformer(FLAVORS["llama-100M"]())
        self.audio_feat_proj = TimeDependentLinear(self.num_quantizers - 1, self.code_dim, embed_dim)
        self.audio_head_proj = TimeDependentLinear(self.num_quantizers - 1, embed_dim, self.code_num)

    def forward(self, codes_c1: torch.Tensor, codes_c2: torch.Tensor):
        """
        Forward pass of the model.
            codes_c1 / codes_c2: (1, num_quantizers, num_frames)
                num_frames is sampled at 12.5Hz.
                c1 is the user audio and c2 is the assistant audio.
                The model takes in the two channels and predicts the c2 codes.

        The forward pass consists of two steps
            1. takes in the two channels of RVQ codes and outputs the latent features
            2. the latent features are passed through the decoder to predict the RVQ codes
        """
        bs = codes_c1.shape[0]
        T = codes_c1.shape[2]
        latent_features: Float[torch.Tensor, "bs T embed_dim"] = self._embed_two_channel_audio_codes(codes_c1, codes_c2)

        backbone_mask = self.causal_mask.unsqueeze(0).expand(bs, -1, -1)[:, :T, :T]

        # backbone inference to get the c0 semantic token.
        transformer_outputs: Float[torch.Tensor, "bs T embed_dim"] = self.backbone(latent_features, mask=backbone_mask)
        c0_logit = self.c0_head(transformer_outputs)

        # Audio generation: from semantic tokens to acoustic tokens.
        ac_quantizers = self.num_quantizers - 1
        leveled_audio_feats: Float[torch.Tensor, "total_frames num_quantizers embed_dim"] = (
            self._embed_audio_quantizer_tokens(codes_c2)
        )
        leveled_audio_feats = self.audio_feat_proj(leveled_audio_feats)
        total_frames = leveled_audio_feats.shape[0]
        leveled_mask = self.causal_mask.unsqueeze(0).expand(total_frames, -1, -1)[:, :ac_quantizers, :ac_quantizers]
        leveled_audio_feats = leveled_audio_feats[
            :, : self.num_quantizers - 1, :
        ]  # Need to decode the leveled audio tokens sequentially.
        decoder_hidden_states: Float[torch.Tensor, "total_frames num_quantizers embed_dim"] = self.decoder(
            leveled_audio_feats, mask=leveled_mask
        )
        audio_logits = self.audio_head_proj(decoder_hidden_states)

        return c0_logit, audio_logits

    def _embed_two_channel_audio_codes(self, codes_c1: torch.Tensor, codes_c2: torch.Tensor):
        """
        Embed the audio codes into latent features.
        """
        with torch.no_grad():
            c1_latent = self.quantizer.decode(codes_c1)
            c2_latent = self.quantizer.decode(codes_c2)

        # combine the two channels.
        latent_features: Float[torch.Tensor, "bs D T"] = torch.cat([c1_latent, c2_latent], dim=1)
        latent_features: Float[torch.Tensor, "bs T D"] = latent_features.transpose(1, 2)

        # project to match the embedding dimension of the transformer
        latent_features: Float[torch.Tensor, "bs T embed_dim"] = self.input_proj(latent_features)
        return latent_features

    def _embed_audio_quantizer_tokens(self, codes: torch.Tensor):
        """
        Embed the audio quantizer by each quantizer. The function preserves each
        RVQ token's embedding without summing them up.

        Args:
            codes: (bs, num_quantizers, num_frames)

        Output:
            latent: (bs, num_quantizers, num_frames, embed_dim)
        """
        num_ac_quantizers = self.num_quantizers - 1

        # Obtain only the acoustic tokens
        c2_tokens_ac: Float[torch.Tensor, "total_frames num_ac_quantizers"] = (
            codes[:, 1:, :].transpose(1, 2).reshape(-1, num_ac_quantizers)
        )

        # Get semantic tokens
        c2_tokens_semantic: Float[torch.Tensor, "total_frames 1"] = codes[:, 0, :].reshape(-1, 1)
        c2_semantic_feats: Float[torch.Tensor, "total_frames 1 embed_dim"] = (
            self.quantizer.semantic_residual_vector_quantizer.layers[0].codebook.decode(c2_tokens_semantic)
        )

        # Use the right codebook to obtain the dense audio features
        audio_feats = [c2_semantic_feats]
        for i in range(num_ac_quantizers):
            feat_i = (
                self.quantizer.acoustic_residual_vector_quantizer.layers[i]
                .codebook.decode(c2_tokens_ac[:, i])
                .unsqueeze(1)
            )
            audio_feats.append(feat_i)

        audio_feats: Float[torch.Tensor, "total_frames num_quantizers embed_dim"] = torch.cat(audio_feats, dim=1)

        return audio_feats

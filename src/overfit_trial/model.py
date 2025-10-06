"""
Implementation of the model.
"""
# ruff: noqa: F722

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from einops import rearrange

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
    def __init__(self, num_quantizers: int = 32, mimi_audio_embed_dir: str = None):
        super().__init__()

        # Quantizer for audio embeddings
        # self._init_mimi_quantizer(num_quantizers)
        self._load_mimi_audio_embeddings(num_quantizers, mimi_audio_embed_dir)

        # Model
        self._init_backbone(MODEL_ID)
        self._init_decoder()

        # Model training and inference
        causal_mask = _create_causal_mask(self.backbone.max_seq_len)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        # Misc
        self.num_parameters = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters: {self.num_parameters:,}")

    def _load_mimi_audio_embeddings(self, num_quantizers: int, mimi_audio_embed_dir: str):
        """
        Load up the exported mimi quantized audio embeddings in flat.

        The flat audio embeddings are of shape: (num_quantizers * code_num, code_dim)
        """
        if mimi_audio_embed_dir is None:
            project_root = Path(__file__).resolve().parents[2]
            mimi_audio_embed_file = (
                project_root / "asset" / "mimi_audio_embeddings" / (f"mimi_projected_embeddings_{num_quantizers}q.pt")
            )
        else:
            mimi_audio_embed_file = Path(mimi_audio_embed_dir) / f"mimi_projected_embeddings_{num_quantizers}q.pt"

        if not mimi_audio_embed_file.exists():
            raise FileNotFoundError(f"Mimi audio embeddings file {mimi_audio_embed_file} not found")

        audio_embed = torch.load(mimi_audio_embed_file)
        self.num_quantizers, self.code_num, self.code_dim = audio_embed.shape
        self.input_dim = self.code_dim

        audio_embed = audio_embed.reshape(self.num_quantizers * self.code_num, self.code_dim)

        self.register_buffer("audio_embed", audio_embed, persistent=False)

    def _init_mimi_quantizer(self, num_quantizers: int):
        """
        Quantizer to convert the RVQ codes to dense features.
        """
        mimi_model = MimiModel.from_pretrained("kyutai/mimi", num_quantizers=num_quantizers)
        self.quantizer = mimi_model.quantizer
        self.code_num = self.quantizer.acoustic_residual_vector_quantizer.layers[0].codebook.embed.shape[0]
        self.code_dim = self.quantizer.acoustic_residual_vector_quantizer.output_proj.weight.shape[0]
        self.num_quantizers = len(self.quantizer.acoustic_residual_vector_quantizer.layers) + len(
            self.quantizer.semantic_residual_vector_quantizer.layers
        )
        self.input_dim = self.code_dim
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
        self.c0_head = nn.Linear(embed_dim, self.code_num)

    def _init_decoder(self):
        """
        Decoder that takes the transformer outputs and predicts the RVQ codes.
        """
        self.decoder, embed_dim = prepare_transformer(FLAVORS["llama-100M"]())
        # Project all the code features to the input dimension of the decoder.
        self.audio_feat_proj = TimeDependentLinear(self.num_quantizers, self.code_dim, embed_dim)
        #
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
        latent_feat_c1 = rearrange(self._embed_audio(codes_c1, collapse_quantizer_levels=True), "bs d t -> bs t d")
        latent_feat_c2 = rearrange(self._embed_audio(codes_c2, collapse_quantizer_levels=True), "bs d t -> bs t d")
        latent_features = torch.cat([latent_feat_c1, latent_feat_c2], dim=2)

        # Project concatenated features to the backbone's embedding dim
        latent_features = self.input_proj(latent_features)
        backbone_mask = self._causal_mask(T, device=codes_c1.device).unsqueeze(0).expand(bs, -1, -1)

        transformer_outputs: Float[torch.Tensor, "bs T embed_dim"] = self.backbone(latent_features, mask=backbone_mask)
        c0_logit = self.c0_head(transformer_outputs)

        # Audio generation: from semantic tokens to acoustic tokens.
        ac_quantizers = self.num_quantizers - 1

        latent_quantizer_feats = self._embed_audio(codes_c2, collapse_quantizer_levels=False)  # [B, V, D, T]
        latent_quantizer_feats = rearrange(latent_quantizer_feats, "b v d t -> b t v d")
        latent_quantizer_feats = rearrange(latent_quantizer_feats, "b t v d -> (b t) v d")
        latent_quantizer_feats = self.audio_feat_proj(latent_quantizer_feats)

        total_frames = latent_quantizer_feats.shape[0]
        leveled_mask = (
            self._causal_mask(ac_quantizers, device=codes_c1.device).unsqueeze(0).expand(total_frames, -1, -1)
        )
        leveled_audio_feats = latent_quantizer_feats[:, :ac_quantizers, :]
        decoder_hidden_states = self.decoder(leveled_audio_feats, mask=leveled_mask)
        audio_logits = self.audio_head_proj(decoder_hidden_states)
        audio_logits = rearrange(audio_logits, "(b t) v c -> b t v c", b=bs)
        return c0_logit, audio_logits

    def _causal_mask(self, T: int, device: torch.device):
        return self.causal_mask[:T, :T].to(device)

    def _embed_audio(self, codes: torch.Tensor, collapse_quantizer_levels: bool = False):
        """
        Embed the audio codes into latent features, collapsing the quantizer levels.

        Args:
            codes: (B, V, T)

        Output:
            feats: (B, V, D, T) if collapse_quantizer_levels is False, otherwise (B, D, T)
        """
        if codes.dtype != torch.long:
            raise ValueError(f"Codes must be of type torch.long, got {codes.dtype}")

        offsets = (torch.arange(self.num_quantizers, device=codes.device) * self.code_num).view(
            1, self.num_quantizers, 1
        )
        global_ids = codes + offsets

        feats = F.embedding(global_ids, self.audio_embed)  # [B, V, T, D]
        feats = rearrange(feats, "b v t d -> b v d t")

        if collapse_quantizer_levels:
            feats = torch.sum(feats, dim=1)
        return feats

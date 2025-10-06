"""Feature extraction helpers built around the Kyutai MIMI model."""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
import torch
import torchaudio

from transformers import AutoFeatureExtractor, MimiModel


MIMI_MODEL_ID = "kyutai/mimi"


@lru_cache(maxsize=None)
def _get_feature_extractor(model_id: str = MIMI_MODEL_ID) -> AutoFeatureExtractor:
    """Load and cache the MIMI feature extractor."""

    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    if getattr(extractor, "sampling_rate", None) is None:
        raise ValueError("AutoFeatureExtractor for MIMI must define a sampling_rate")
    return extractor


@lru_cache(maxsize=None)
def _get_mimi_model(num_quantizers: int, model_id: str = MIMI_MODEL_ID) -> MimiModel:
    """Load and cache the pretrained MIMI model configured with ``num_quantizers``."""

    if num_quantizers < 1:
        raise ValueError("num_quantizers must be a positive integer")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MimiModel.from_pretrained(model_id, num_quantizers=num_quantizers)
    model.to(device)
    model.eval()
    return model


def _resample_audio(audio: np.ndarray, src_rate: int, target_rate: int) -> np.ndarray:
    """Return ``audio`` resampled to ``target_rate`` while keeping it mono."""

    if src_rate == target_rate:
        return audio.astype(np.float32, copy=False)

    waveform = torch.from_numpy(audio.astype(np.float32, copy=False)).unsqueeze(0)
    resampled = torchaudio.functional.resample(waveform, src_rate, target_rate)
    return resampled.squeeze(0).cpu().numpy()


def _reshape_codes(codes: torch.Tensor, num_quantizers: int) -> torch.Tensor:
    """Collapse the ``audio_codes`` tensor to shape ``(num_quantizers, num_frames)``."""

    if num_quantizers not in codes.shape:
        raise ValueError(
            f"Unexpected audio_codes shape {tuple(codes.shape)} for num_quantizers={num_quantizers}"
        )

    # Position the quantizer axis first, keep the rest (batch/time) flattened.
    quant_axes = [idx for idx, size in enumerate(codes.shape) if size == num_quantizers]
    if not quant_axes:
        raise ValueError("audio_codes tensor does not contain the quantizer dimension")

    codes = codes.movedim(quant_axes[0], 0)

    # Any remaining singleton dimensions (typically batch) should be removed before flattening.
    if codes.dim() > 2:
        remaining = int(np.prod([dim for dim in codes.shape[1:]]))
        codes = codes.contiguous().reshape(num_quantizers, remaining)
    elif (
        codes.dim() == 1
    ):  # pragma: no cover - defensive fallback for degenerate shapes
        codes = codes.unsqueeze(0)
    else:
        codes = codes.contiguous()

    return codes


def get_mimi_features(
    audio_arr: np.ndarray, src_rate: int, num_quantizers: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract semantic and acoustic tokens from ``audio_arr`` using the MIMI encoder."""

    if src_rate <= 0:
        raise ValueError("src_rate must be a positive integer")

    audio = np.asarray(audio_arr, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("audio_arr must be a 1-D mono waveform")
    if audio.size == 0:
        raise ValueError("audio_arr cannot be empty")

    feature_extractor = _get_feature_extractor()
    target_rate = int(feature_extractor.sampling_rate)
    audio = _resample_audio(audio, src_rate, target_rate)

    inputs = feature_extractor(
        raw_audio=audio,
        sampling_rate=target_rate,
        return_tensors="pt",
    )

    input_values = inputs.get("input_values")
    if input_values is None:
        raise ValueError("Feature extractor did not return input_values")

    model = _get_mimi_model(num_quantizers)
    device = next(model.parameters()).device
    input_values = input_values.to(device)
    attention_mask = inputs.get("attention_mask")
    encode_kwargs = {}
    if attention_mask is not None:
        encode_kwargs["attention_mask"] = attention_mask.to(device)

    with torch.inference_mode():
        encoder_outputs = model.encode(
            input_values,
            return_dict=True,
            **encode_kwargs,
        )

    codes = getattr(encoder_outputs, "audio_codes", None)
    if codes is None:
        raise ValueError("Encoder outputs missing audio_codes")

    codes = _reshape_codes(codes.detach().to(torch.int64).cpu(), num_quantizers)
    codes_np = codes.numpy()

    sem_token = codes_np[:1, :]
    ac_token = codes_np[1:, :]

    return sem_token, ac_token


__all__ = ["get_mimi_features"]

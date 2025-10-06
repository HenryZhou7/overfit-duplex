"""Data models for working with duplex conversation samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .data_postprocessing import apply_transcript_mask


class Sample(BaseModel):
    """Represents a single utterance sample from the dataset."""

    uid: str
    json_file: Path = Field(alias="json_path")
    wav_file: Path | None = None

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    @field_validator("json_file", mode="before")
    @classmethod
    def _coerce_json_path(cls, value: Any) -> Path:
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            return Path(value)
        raise TypeError("json_file must be a path-like string")

    @field_validator("wav_file", mode="before")
    @classmethod
    def _coerce_wav_path(cls, value: Any) -> Path | None:
        if value is None or value == "":
            return None
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            return Path(value)
        raise TypeError("wav_file must be a path-like string if provided")

    @model_validator(mode="after")
    def _ensure_wav_file(self) -> "Sample":
        if self.wav_file is None:
            self.wav_file = self.json_file.with_suffix(".wav")
        return self

    def get_single_channel_audio(
        self, rate: int, mask_transcript: bool = False, return_tensors: bool = False
    ) -> np.ndarray:
        """Load the waveform resampled to the requested rate as a single channel."""

        if rate <= 0:
            raise ValueError("rate must be a positive integer")
        if not self.wav_file.exists():
            raise FileNotFoundError(f"Audio file not found: {self.wav_file}")

        waveform, original_rate = sf.read(self.wav_file.as_posix(), dtype="float32", always_2d=False)

        if waveform.ndim == 2:
            if waveform.shape[1] != 1:
                raise ValueError(f"Expected mono audio, found {waveform.shape[1]} channels in {self.wav_file}")
            waveform = waveform[:, 0]
        elif waveform.ndim != 1:
            raise ValueError(f"Unexpected waveform shape {waveform.shape} in {self.wav_file}")

        if mask_transcript:
            if not self.json_file.exists():
                raise FileNotFoundError(f"Transcript file not found: {self.json_file}")
            with self.json_file.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            transcript = metadata.get("metadata:transcript")
            if transcript:
                waveform = apply_transcript_mask(
                    waveform,
                    transcript,
                    sampling_rate=original_rate,
                )

        if original_rate != rate:
            tensor = torch.from_numpy(waveform).unsqueeze(0)
            resampled = torchaudio.functional.resample(tensor, original_rate, rate)
            waveform = np.asarray(resampled.squeeze(0))

        if return_tensors:
            return torch.from_numpy(waveform).float()
        else:
            return waveform.astype(np.float32, copy=False)


class PairSample(BaseModel):
    """Represents a duplex conversation pair."""

    sample1: Sample
    sample2: Sample


__all__ = ["Sample", "PairSample"]

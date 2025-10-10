"""Pydantic data models for duplex audio conversations.

Relationship of the different classes

Conversation archive
  → DuplexConversation → load_codes()
    → DuplexCodes → .user/.assistant
      → MimiChannelCodes → .window(frame_slice)
        → WindowedChannelCodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SAMPLE_RATE_HZ: float = 7.5
DEFAULT_PAD_VALUE: int = -100
ChannelRole = Literal["user", "assistant"]


class FrameSlice(BaseModel):
    """Describe a slice window measured in quantized audio frames."""

    start: int = Field(default=0, ge=0)
    length: int | None = Field(default=None, gt=0)
    pad_to_length: bool = False
    pad_value: int = DEFAULT_PAD_VALUE

    model_config = ConfigDict(strict=True)

    @model_validator(mode="after")
    def _validate_pad_options(self) -> "FrameSlice":
        if self.length is None and self.pad_to_length:
            raise ValueError("pad_to_length requires length to be specified")
        return self

    def bounds(self, total_frames: int) -> Tuple[int, int, int]:
        """
        Resolve the slice against the available frame count.

        Returns ``(start, stop, actual_length)``.
        """

        if total_frames < 0:
            raise ValueError("total_frames must be non-negative")

        start = min(self.start, total_frames)
        if self.length is None:
            stop = total_frames
        else:
            stop = min(total_frames, start + self.length)
        actual = max(0, stop - start)
        return start, stop, actual

    def target_length(self, actual_length: int) -> int:
        """Return the desired output length after optional padding."""

        if self.length is not None and self.pad_to_length:
            return self.length
        return actual_length


class MimiChannelMetadata(BaseModel):
    """Metadata describing a single channel of quantized audio codes."""

    channel_id: str
    num_frames: int = Field(ge=0)
    semantic_quantizers: int = Field(default=1, ge=0)
    acoustic_quantizers: int = Field(ge=0)
    sample_rate_hz: float = Field(default=SAMPLE_RATE_HZ, gt=0)

    model_config = ConfigDict(frozen=True)

    @property
    def total_quantizers(self) -> int:
        return self.semantic_quantizers + self.acoustic_quantizers

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate_hz == 0:
            return 0.0
        return self.num_frames / self.sample_rate_hz


class MimiChannelCodes(BaseModel):
    """Semantic and acoustic tokens for a single speaker channel."""

    metadata: MimiChannelMetadata
    semantic_codes: np.ndarray
    acoustic_codes: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("semantic_codes", "acoustic_codes", mode="before")
    @classmethod
    def _ensure_numpy(cls, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim != 2:
            raise ValueError("Quantized codes must be a 2-D array")

        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError("Quantized codes must contain integer values")
        return array

    @model_validator(mode="after")
    def _check_shapes(self) -> "MimiChannelCodes":
        meta = self.metadata
        if self.semantic_codes.shape[1] != self.acoustic_codes.shape[1]:
            raise ValueError("Semantic and acoustic codes must share the frame dimension")
        total_frames = self.semantic_codes.shape[1]
        if meta.num_frames != total_frames:
            self.metadata = meta.model_copy(update={"num_frames": total_frames})
        if self.semantic_codes.shape[0] != meta.semantic_quantizers:
            self.metadata = self.metadata.model_copy(update={"semantic_quantizers": int(self.semantic_codes.shape[0])})
        if self.acoustic_codes.shape[0] != meta.acoustic_quantizers:
            self.metadata = self.metadata.model_copy(update={"acoustic_quantizers": int(self.acoustic_codes.shape[0])})
        return self

    @property
    def stacked(self) -> np.ndarray:
        """Return semantic+acoustic codes stacked along the quantizer axis."""

        return np.concatenate([self.semantic_codes, self.acoustic_codes], axis=0)

    @property
    def num_frames(self) -> int:
        return int(self.metadata.num_frames)

    @property
    def num_quantizers(self) -> int:
        return int(self.metadata.total_quantizers)

    def to_torch(self, dtype: torch.dtype = torch.long) -> torch.Tensor:
        """Convert the stacked codes to a tensor."""

        codes = self.stacked.astype(np.int64, copy=False)
        tensor = torch.from_numpy(codes)
        if dtype != torch.long:
            tensor = tensor.to(dtype)
        return tensor

    def window(self, frame_slice: FrameSlice | None) -> "WindowedChannelCodes":
        """Apply ``frame_slice`` and return a windowed view of the codes."""

        if frame_slice is None:
            mask = np.ones(self.num_frames, dtype=bool)
            return WindowedChannelCodes(
                metadata=self.metadata,
                codes=self.stacked,
                attention_mask=mask,
            )

        start, stop, actual_length = frame_slice.bounds(self.num_frames)
        sliced = self.stacked[:, start:stop]
        target_length = frame_slice.target_length(actual_length)
        if target_length < 0:
            raise ValueError("target_length cannot be negative")

        if target_length > sliced.shape[1]:
            pad_width = target_length - sliced.shape[1]
            if pad_width > 0:
                pad_spec = ((0, 0), (0, pad_width))
                sliced = np.pad(sliced, pad_spec, constant_values=frame_slice.pad_value)
            mask = np.zeros(target_length, dtype=bool)
            mask[:actual_length] = True
        else:
            sliced = sliced[:, :target_length]
            mask = np.ones(target_length, dtype=bool)
            mask[actual_length:] = False

        metadata = self.metadata.model_copy(update={"num_frames": sliced.shape[1]})
        return WindowedChannelCodes(metadata=metadata, codes=sliced, attention_mask=mask)


class WindowedChannelCodes(BaseModel):
    """Windowed channel codes paired with an attention mask."""

    metadata: MimiChannelMetadata
    codes: np.ndarray
    attention_mask: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_shapes(self) -> "WindowedChannelCodes":
        if self.codes.ndim != 2:
            raise ValueError("codes must have shape (quantizers, frames)")
        if self.attention_mask.ndim != 1:
            raise ValueError("attention_mask must be 1-D")
        if self.codes.shape[1] != self.attention_mask.shape[0]:
            raise ValueError("Mismatch between codes and attention_mask lengths")
        if self.metadata.num_frames != self.codes.shape[1]:
            self.metadata = self.metadata.model_copy(update={"num_frames": self.codes.shape[1]})
        return self

    @property
    def num_frames(self) -> int:
        return int(self.metadata.num_frames)

    def to_torch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert codes and mask to torch tensors."""

        codes = torch.from_numpy(self.codes.astype(np.int64, copy=False))
        mask = torch.from_numpy(self.attention_mask.astype(np.bool_, copy=False))
        return codes, mask


class MimiChannelArchive(BaseModel):
    """Pointer to the on-disk quantized features for a conversation side."""

    channel_id: str
    npz_path: Path = Field(alias="path")
    semantic_key: str = "sem_tokens"
    acoustic_key: str = "ac_tokens"
    sample_rate_hz: float = Field(default=SAMPLE_RATE_HZ, gt=0)

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    @field_validator("npz_path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Path | str) -> Path:
        return Path(value)

    @model_validator(mode="after")
    def _check_exists(self) -> "MimiChannelArchive":
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Quantized feature archive missing: {self.npz_path}")
        return self

    def peek_metadata(self) -> MimiChannelMetadata:
        """Return metadata without loading the full arrays."""

        with np.load(self.npz_path, allow_pickle=False, mmap_mode="r") as handle:
            if self.semantic_key not in handle or self.acoustic_key not in handle:
                raise KeyError(f"Expected keys '{self.semantic_key}' and '{self.acoustic_key}' in {self.npz_path.name}")
            sem_shape = handle[self.semantic_key].shape
            ac_shape = handle[self.acoustic_key].shape

        if sem_shape[1] != ac_shape[1]:
            raise ValueError("Semantic and acoustic arrays differ in frame counts")

        return MimiChannelMetadata(
            channel_id=self.channel_id,
            num_frames=int(sem_shape[1]),
            semantic_quantizers=int(sem_shape[0]),
            acoustic_quantizers=int(ac_shape[0]),
            sample_rate_hz=self.sample_rate_hz,
        )

    def load_codes(self) -> MimiChannelCodes:
        """Load semantic and acoustic token matrices."""

        with np.load(self.npz_path, allow_pickle=False) as handle:
            if self.semantic_key not in handle or self.acoustic_key not in handle:
                raise KeyError(f"Expected keys '{self.semantic_key}' and '{self.acoustic_key}' in {self.npz_path.name}")
            sem = np.asarray(handle[self.semantic_key], dtype=np.int64)
            ac = np.asarray(handle[self.acoustic_key], dtype=np.int64)

        metadata = MimiChannelMetadata(
            channel_id=self.channel_id,
            num_frames=int(sem.shape[1]),
            semantic_quantizers=int(sem.shape[0]),
            acoustic_quantizers=int(ac.shape[0]),
            sample_rate_hz=self.sample_rate_hz,
        )
        return MimiChannelCodes(metadata=metadata, semantic_codes=sem, acoustic_codes=ac)


class DuplexConversation(BaseModel):
    """Pair of speaker channels participating in a duplex exchange."""

    conversation_id: str
    user: MimiChannelArchive
    assistant: MimiChannelArchive

    model_config = ConfigDict(str_strip_whitespace=True)

    @model_validator(mode="after")
    def _validate_rates(self) -> "DuplexConversation":
        if self.user.sample_rate_hz != self.assistant.sample_rate_hz:
            raise ValueError("Both channels must share the same sample rate")
        return self

    @property
    def roles(self) -> Tuple[ChannelRole, ChannelRole]:
        return ("user", "assistant")

    def load_codes(self) -> "DuplexCodes":
        """Load both channels into memory."""

        user_codes = self.user.load_codes()
        assistant_codes = self.assistant.load_codes()
        return DuplexCodes(conversation_id=self.conversation_id, user=user_codes, assistant=assistant_codes)

    def peek_metadata(self) -> Tuple[MimiChannelMetadata, MimiChannelMetadata]:
        """Quick metadata inspection for each channel."""

        return self.user.peek_metadata(), self.assistant.peek_metadata()


class DuplexCodes(BaseModel):
    """Quantized codes for both sides of a conversation."""

    conversation_id: str
    user: MimiChannelCodes
    assistant: MimiChannelCodes

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _check_alignment(self) -> "DuplexCodes":
        if self.user.num_frames != self.assistant.num_frames:
            raise ValueError("Channel frame counts must match")
        if self.user.num_quantizers != self.assistant.num_quantizers:
            raise ValueError("Channel quantizer counts must match")
        return self

    @property
    def num_frames(self) -> int:
        return int(self.user.num_frames)

    @property
    def num_quantizers(self) -> int:
        return int(self.user.num_quantizers)

    def to_numpy(self) -> np.ndarray:
        """Return array shaped as (channels, quantizers, frames)."""

        return np.stack([self.user.stacked, self.assistant.stacked], axis=0)

    def to_torch(self) -> torch.Tensor:
        """Return tensor shaped as (channels, quantizers, frames)."""

        array = self.to_numpy().astype(np.int64, copy=False)
        return torch.from_numpy(array)

    def window(self, frame_slice: FrameSlice | None) -> "DuplexWindow":
        """Return a windowed view shared across both channels."""

        user_window = self.user.window(frame_slice)
        assistant_window = self.assistant.window(frame_slice)
        return DuplexWindow(
            conversation_id=self.conversation_id,
            user=user_window,
            assistant=assistant_window,
        )


class DuplexWindow(BaseModel):
    """Windowed duplex sample ready for collation."""

    conversation_id: str
    user: WindowedChannelCodes
    assistant: WindowedChannelCodes

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _check_masks(self) -> "DuplexWindow":
        if self.user.num_frames != self.assistant.num_frames:
            raise ValueError("Windowed channels must have identical frame lengths")
        return self

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return stacked codes and masks as numpy arrays."""

        codes = np.stack([self.user.codes, self.assistant.codes], axis=0)
        masks = np.stack([self.user.attention_mask, self.assistant.attention_mask], axis=0)
        return codes, masks

    def to_torch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return stacked codes and masks as tensors."""

        codes_np, masks_np = self.to_numpy()
        codes = torch.from_numpy(codes_np.astype(np.int64, copy=False))
        masks = torch.from_numpy(masks_np.astype(np.bool_, copy=False))
        return codes, masks

    @property
    def num_frames(self) -> int:
        return int(self.user.num_frames)


class CollateConfig(BaseModel):
    """Configuration helpers for the dataloader collate function."""

    pad_value: int = DEFAULT_PAD_VALUE
    stack_order: Tuple[ChannelRole, ChannelRole] = ("user", "assistant")
    return_masks: bool = True

    model_config = ConfigDict(validate_default=True)

    @model_validator(mode="after")
    def _validate_order(self) -> "CollateConfig":
        if set(self.stack_order) != {"user", "assistant"}:
            raise ValueError("stack_order must include both 'user' and 'assistant'")
        return self


__all__ = [
    "SAMPLE_RATE_HZ",
    "DEFAULT_PAD_VALUE",
    "ChannelRole",
    "FrameSlice",
    "MimiChannelMetadata",
    "MimiChannelCodes",
    "WindowedChannelCodes",
    "MimiChannelArchive",
    "DuplexConversation",
    "DuplexCodes",
    "DuplexWindow",
    "CollateConfig",
]

"""Utilities for post-processing dataset audio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class TranscriptTurn:
    """Representation of a single transcript turn used for masking."""

    start: float
    end: float

    @classmethod
    def from_mapping(cls, item: Mapping[str, object]) -> "TranscriptTurn | None":
        """Create a turn from a mapping, returning ``None`` on malformed values."""

        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            return None

        if not np.isfinite(start) or not np.isfinite(end):
            return None

        return cls(start=start, end=end)


def _iter_transcript_turns(
    transcript: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]] | None,
) -> Iterable[TranscriptTurn]:
    """Yield validated transcript turns from an arbitrary iterable."""

    if transcript is None:
        return ()

    for item in transcript:
        turn = TranscriptTurn.from_mapping(item)
        if turn is None:
            continue
        if turn.end <= turn.start:
            continue
        yield turn


def build_transcript_mask(
    transcript: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]] | None,
    *,
    num_samples: int,
    sampling_rate: int,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Construct a binary mask where transcript speech regions are ``1``.

    The routine mirrors the exploratory masking strategy used in
    ``notebooks/examine_dataset.ipynb`` by marking every transcript turn as
    a contiguous region between ``start`` and ``end`` (in seconds) and setting
    those samples to ``1`` while leaving the rest at ``0``. Values are clipped
    to the available sample range so out-of-bound transcripts do not raise.

    Parameters
    ----------
    transcript:
        Iterable transcript containing mappings with ``start`` and ``end`` keys.
    num_samples:
        Total number of samples in the corresponding audio array.
    sampling_rate:
        Sampling rate for the waveform, in Hz.
    dtype:
        Numpy dtype of the mask. Defaults to ``float32`` as used in the notebook.

    Returns
    -------
    np.ndarray
        A ``num_samples``-long mask with ``1`` marking transcript speech.
    """

    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")

    mask = np.zeros(num_samples, dtype=dtype)
    if num_samples == 0:
        return mask

    for turn in _iter_transcript_turns(transcript):
        start_idx = int(turn.start * sampling_rate)
        end_idx = int(turn.end * sampling_rate)

        if end_idx <= 0 or start_idx >= num_samples:
            continue

        start_idx = max(0, start_idx)
        end_idx = min(num_samples, end_idx)

        if end_idx > start_idx:
            mask[start_idx:end_idx] = 1

    return mask


def apply_transcript_mask(
    waveform: np.ndarray,
    transcript: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]] | None,
    *,
    sampling_rate: int,
    invert: bool = False,
) -> np.ndarray:
    """Apply a transcript-derived mask to ``waveform``.

    Parameters
    ----------
    waveform:
        One-dimensional waveform array.
    transcript:
        Iterable transcript containing mappings with ``start`` and ``end`` keys.
    sampling_rate:
        Sampling rate for the waveform, in Hz.
    invert:
        When ``True`` keep only the non-transcript regions instead of speech.

    Returns
    -------
    np.ndarray
        Waveform with transcript regions retained (or removed when ``invert``).
    """

    if waveform.ndim != 1:
        raise ValueError("waveform must be a 1-D array")

    mask = build_transcript_mask(
        transcript,
        num_samples=waveform.shape[0],
        sampling_rate=sampling_rate,
        dtype=waveform.dtype,
    )

    if invert:
        mask = 1 - mask

    return waveform * mask


__all__ = ["build_transcript_mask", "apply_transcript_mask"]

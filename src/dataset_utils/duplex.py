"""Utility helpers for turning Seamless Interaction samples into duplex data."""

from __future__ import annotations

from pathlib import Path
from seamless_interaction.fs import SeamlessInteractionFS
from dataset_utils.data_models import Sample


def get_data_sample(fs: SeamlessInteractionFS, uid: str) -> Sample:
    """Return a :class:`Sample` describing the assets for ``uid``."""

    file_paths = fs.get_path_list_for_file_id_local(uid)
    sample_json = next(Path(path) for path in file_paths if path.endswith(".json"))

    return Sample(
        uid=uid,
        json_file=sample_json,
        wav_file=sample_json.with_suffix(".wav"),
    )


__all__ = ["get_data_sample"]

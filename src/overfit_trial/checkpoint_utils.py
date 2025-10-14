"""
Utilities for robust checkpoint loading.

Provides a relaxed loader that handles legacy checkpoints which
pickled TrainingConfig as "__main__.TrainingConfig".
"""

from __future__ import annotations

import sys
import os
import torch


def load_checkpoint(path: str | bytes | os.PathLike, map_location: str | torch.device = "cpu"):
    """
    Load a checkpoint while handling legacy TrainingConfig pickling.

    If the checkpoint was saved when TrainingConfig lived in a script executed
    as __main__, unpickling in a different process (e.g., a notebook) will fail
    with AttributeError. This helper patches __main__.TrainingConfig to point to
    the importable class and retries.
    """
    try:
        return torch.load(path, map_location=map_location)
    except AttributeError as e:
        msg = str(e)
        if "TrainingConfig" in msg and "__main__" in msg:
            # Patch the current __main__ module to expose TrainingConfig
            try:
                from overfit_trial.training_config import TrainingConfig  # absolute import as per project convention

                setattr(sys.modules["__main__"], "TrainingConfig", TrainingConfig)
                return torch.load(path, map_location=map_location)
            except Exception:
                # Fall through and re-raise the original error if patching fails
                pass
        raise

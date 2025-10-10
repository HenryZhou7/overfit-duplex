"""PyTorch dataset for loading duplex conversation pairs from CSV."""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from overfit_trial.data_models import (
    DuplexConversation,
    FrameSlice,
    MimiChannelArchive,
)


class DuplexPairDataset(Dataset):
    """Dataset for loading pairs of duplex conversations from a CSV file.

    The CSV should have two columns: channel1_id, channel2_id
    Each row represents a conversation pair.
    """

    def __init__(
        self,
        csv_file: Path | str,
        window_size: int,
        data_dir: Path | str,
        num_quantizers: int = 32,
        pad_to_window_size: bool = True,
        random_window: bool = False,
    ):
        """Initialize the dataset.

        Args:
            csv_file: Path to CSV file with channel pairs
            window_size: Size of the window in frames
            data_dir: Directory containing the NPZ files
            num_quantizers: Number of quantizers (default: 32)
            pad_to_window_size: Whether to pad short sequences
            random_window: Whether to randomly sample windows from the conversation
                if True, randomly sample the window size from the conversation
                if False, always use the first window_size frames
        """
        self.csv_file = Path(csv_file)
        self.window_size = window_size
        self.data_dir = Path(data_dir)
        self.num_quantizers = num_quantizers
        self.pad_to_window_size = pad_to_window_size
        self.random_window = random_window

        # Load conversation pairs from CSV
        self.conversation_pairs = self._load_csv()

    def _load_csv(self) -> List[Tuple[str, str]]:
        """Load channel pairs from CSV file."""
        pairs = []
        with open(self.csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                channel1_id = row["channel1_id"].strip()
                channel2_id = row["channel2_id"].strip()
                pairs.append((channel1_id, channel2_id))
        return pairs

    def __len__(self) -> int:
        """Return the total number of conversation pairs."""
        return len(self.conversation_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int]:
        """Get a windowed sample.

        Returns:
            dict with keys:
                - 'user_codes': Tensor of shape (num_quantizers, window_size)
                - 'assistant_codes': Tensor of shape (num_quantizers, window_size)
                - 'user_length': Actual length of user audio (int)
                - 'assistant_length': Actual length of assistant audio (int)
        """
        channel1_id, channel2_id = self.conversation_pairs[idx]

        # Create channel archives
        npz_path1 = self.data_dir / f"{channel1_id}.{self.num_quantizers}q.npz"
        npz_path2 = self.data_dir / f"{channel2_id}.{self.num_quantizers}q.npz"

        archive1 = MimiChannelArchive(
            channel_id=channel1_id,
            npz_path=npz_path1,
        )
        archive2 = MimiChannelArchive(
            channel_id=channel2_id,
            npz_path=npz_path2,
        )

        # Create conversation and load codes
        conversation = DuplexConversation(
            conversation_id=f"{channel1_id}_{channel2_id}_{idx}",
            user=archive1,
            assistant=archive2,
        )
        duplex_codes = conversation.load_codes()

        # Calculate the actual length before windowing
        total_frames = duplex_codes.num_frames

        if total_frames < self.window_size:
            raise ValueError(f"Conversation {channel1_id}_{channel2_id}_{idx} has less than {self.window_size} frames")

        if self.random_window and total_frames > self.window_size:
            rng = np.random.default_rng()
            start_idx = rng.integers(0, total_frames - self.window_size + 1)
        else:
            start_idx = 0

        # Calculate actual lengths for this window
        actual_length = min(self.window_size, total_frames - start_idx)

        # Create frame slice for windowing
        frame_slice = FrameSlice(
            start=start_idx,
            length=self.window_size,
            pad_to_length=self.pad_to_window_size,
        )

        # Get windowed data
        windowed = duplex_codes.window(frame_slice)

        # Convert to torch tensors (only get codes, not masks)
        codes, _ = windowed.to_torch()

        return {
            "user_codes": codes[0],  # Shape: (num_quantizers, window_size)
            "assistant_codes": codes[1],  # Shape: (num_quantizers, window_size)
            "user_length": actual_length,  # Actual length of user audio in this window
            "assistant_length": actual_length,  # Actual length of assistant audio in this window
        }


def collate_duplex_batch(
    batch: List[Dict[str, torch.Tensor | int]],
) -> Dict[str, torch.Tensor | int]:
    """Collate function for batching duplex samples.

    Args:
        batch: List of samples from DuplexPairDataset

    Returns:
        Tuple of (user_codes, assistant_codes, user_lengths, assistant_lengths)
        - user_codes: shape (batch_size, num_quantizers, window_size)
        - assistant_codes: shape (batch_size, num_quantizers, window_size)
        - user_lengths: shape (batch_size,)
        - assistant_lengths: shape (batch_size,)
    """
    user_codes = torch.stack([sample["user_codes"] for sample in batch])
    assistant_codes = torch.stack([sample["assistant_codes"] for sample in batch])
    user_lengths = torch.tensor([sample["user_length"] for sample in batch])
    assistant_lengths = torch.tensor([sample["assistant_length"] for sample in batch])

    model_inputs = {
        "user_codes": user_codes,
        "assistant_codes": assistant_codes,
        "user_lengths": user_lengths,
        "assistant_lengths": assistant_lengths,
    }
    return model_inputs

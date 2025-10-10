"""
Factory for creating datasets and data loaders for duplex training.
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from overfit_trial.pair_dataset import DuplexPairDataset, collate_duplex_batch


class DataConfig:
    """Configuration for data loading."""

    def __init__(
        self,
        train_csv: Path,
        eval_csv: Path,
        data_dir: Path,
        window_size: int = 1024,
        batch_size: int = 4,
        num_workers: int = 0,
        num_quantizers: int = 32,
        random_window_train: bool = False,
        pin_memory: bool = True,
    ):
        self.train_csv = train_csv
        self.eval_csv = eval_csv
        self.data_dir = data_dir
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_quantizers = num_quantizers
        self.random_window_train = random_window_train
        self.pin_memory = pin_memory


def create_datasets(config: DataConfig) -> Tuple[DuplexPairDataset, DuplexPairDataset]:
    """
    Create training and evaluation datasets.

    Args:
        config: Data configuration

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    train_dataset = DuplexPairDataset(
        csv_file=config.train_csv,
        window_size=config.window_size,
        data_dir=config.data_dir,
        num_quantizers=config.num_quantizers,
        random_window=config.random_window_train,
        pad_to_window_size=True,
    )

    eval_dataset = DuplexPairDataset(
        csv_file=config.eval_csv,
        window_size=config.window_size,
        data_dir=config.data_dir,
        num_quantizers=config.num_quantizers,
        random_window=False,  # Always use deterministic windows for eval
        pad_to_window_size=True,
    )

    return train_dataset, eval_dataset


def create_data_loaders(
    train_dataset: DuplexPairDataset, eval_dataset: DuplexPairDataset, config: DataConfig, device: torch.device
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders from datasets.

    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: Data configuration
        device: Device to use for pin_memory

    Returns:
        Tuple of (train_loader, eval_loader)
    """
    use_pin_memory = config.pin_memory and device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_duplex_batch,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_duplex_batch,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, eval_loader


def create_default_data_config() -> DataConfig:
    """Create default data configuration for overfitting experiment."""
    project_root = Path(__file__).resolve().parents[2]

    return DataConfig(
        train_csv=project_root / "asset" / "csv" / "single_sample_x16.csv",
        eval_csv=project_root / "asset" / "csv" / "single_sample_x1.csv",
        data_dir=project_root / "asset" / "single_pair_dataset",
        window_size=512,
        batch_size=2,
        num_workers=0,
        num_quantizers=32,
        random_window_train=False,  # For overfitting, use deterministic windows
        pin_memory=True,
    )

"""
Centralized training configuration for the overfit trial.

Moving TrainingConfig into a real module avoids pickling it as
"__main__.TrainingConfig" when the training script is executed, which
prevents unpickling errors in notebooks and other processes.
"""

from pathlib import Path

from overfit_trial.data_factory import (
    DataConfig,
    create_default_data_config,
)
from overfit_trial.model_factory import (
    ModelConfig,
    create_default_model_config,
)
from overfit_trial.loss_factory import (
    LossConfig,
    create_default_loss_config,
)
from overfit_trial.optimizer_factory import (
    OptimizerConfig,
    create_default_optimizer_config,
)


class TrainingConfig:
    """Main training configuration."""

    def __init__(
        self,
        num_updates: int = 800,
        log_interval: int = 10,
        eval_interval: int = 100,
        checkpoint_interval: int = 200,
        checkpoint_dir: Path = Path("checkpoints/overfit_trial"),
        tensorboard_dir: Path = Path("runs/overfit_trial"),
        # Sub-configurations
        data_config: DataConfig | None = None,
        model_config: ModelConfig | None = None,
        loss_config: LossConfig | None = None,
        optimizer_config: OptimizerConfig | None = None,
    ):
        self.num_updates = num_updates
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_dir = tensorboard_dir

        # Initialize sub-configurations with defaults if not provided
        self.data_config = data_config or create_default_data_config()
        self.model_config = model_config or create_default_model_config()
        self.loss_config = loss_config or create_default_loss_config()
        self.optimizer_config = optimizer_config or create_default_optimizer_config()

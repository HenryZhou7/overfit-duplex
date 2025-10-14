"""
Minimal training script to overfit a pre-trained Llama model on duplex conversation data.

Command:
    # under overfit-duplex
    uv run python -m overfit_trial.train
"""

from typing import Dict

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm

# Import factory modules
from overfit_trial.data_factory import (
    create_datasets,
    create_data_loaders,
)
from overfit_trial.model_factory import (
    create_model,
    print_model_summary,
)
from overfit_trial.loss_factory import (
    calculate_losses,
)
from overfit_trial.optimizer_factory import (
    create_optimizer,
    apply_gradient_clipping,
)
from overfit_trial.training_config import TrainingConfig


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Perform a single training step."""
    model.train()

    # Move batch to device
    user_codes = batch["user_codes"].to(device)
    assistant_codes = batch["assistant_codes"].to(device)
    assistant_lengths = batch["assistant_lengths"].to(device)

    # Forward pass
    model_output = model(user_codes, assistant_codes)

    # Calculate losses using factory function
    losses = calculate_losses(
        model_output=model_output,
        target_codes=assistant_codes,  # Predict assistant codes
        lengths=assistant_lengths,
        num_quantizers=config.data_config.num_quantizers,
        config=config.loss_config,
    )

    # Backward pass
    optimizer.zero_grad()
    losses["total_loss"].backward()

    # Gradient clipping using factory function
    grad_norm = apply_gradient_clipping(model, config.optimizer_config)

    optimizer.step()

    return {
        "loss/total": losses["total_loss"].item(),
        "loss/semantic": losses["semantic_loss"].item(),
        "loss/acoustic": losses["acoustic_loss"].item(),
        "grad_norm": grad_norm,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module, eval_loader: torch.utils.data.DataLoader, config: TrainingConfig, device: torch.device
) -> Dict[str, float]:
    """Evaluate the model on the evaluation dataset."""
    model.eval()

    total_semantic_loss = 0.0
    total_acoustic_loss = 0.0
    total_loss = 0.0
    num_batches = 0

    for batch in eval_loader:
        # Move batch to device
        user_codes = batch["user_codes"].to(device)
        assistant_codes = batch["assistant_codes"].to(device)
        assistant_lengths = batch["assistant_lengths"].to(device)

        # Forward pass
        model_output = model(user_codes, assistant_codes)

        # Calculate losses using factory function
        losses = calculate_losses(
            model_output=model_output,
            target_codes=assistant_codes,
            lengths=assistant_lengths,
            num_quantizers=config.data_config.num_quantizers,
            config=config.loss_config,
        )

        total_semantic_loss += losses["semantic_loss"].item()
        total_acoustic_loss += losses["acoustic_loss"].item()
        total_loss += losses["total_loss"].item()
        num_batches += 1

    avg_semantic_loss = total_semantic_loss / num_batches if num_batches > 0 else 0
    avg_acoustic_loss = total_acoustic_loss / num_batches if num_batches > 0 else 0
    avg_total_loss = total_loss / num_batches if num_batches > 0 else 0

    return {
        "eval/semantic_loss": avg_semantic_loss,
        "eval/acoustic_loss": avg_acoustic_loss,
        "eval/total_loss": avg_total_loss,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    update_step: int,
    config: TrainingConfig,
    metrics: Dict[str, float],
) -> None:
    """Save a training checkpoint."""
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = config.checkpoint_dir / f"checkpoint_update_{update_step}.pt"

    torch.save(
        {
            "update_step": update_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        checkpoint_path,
    )

    print(f"Saved checkpoint to {checkpoint_path}")


def main():
    """Main training loop."""
    # Setup
    config = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model using factory
    print("Creating model...")
    model = create_model(config.model_config)
    model.to(device)
    print_model_summary(model)

    # Create datasets and dataloaders using factory
    print("Creating datasets...")
    train_dataset, eval_dataset = create_datasets(config.data_config)
    train_loader, eval_loader = create_data_loaders(train_dataset, eval_dataset, config.data_config, device)

    # Create optimizer using factory
    print("Creating optimizer...")
    optimizer = create_optimizer(model, config.optimizer_config)

    # Setup TensorBoard
    config.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(config.tensorboard_dir))

    # Training loop
    print(f"Starting training for {config.num_updates} updates...")
    update_step = 0
    train_iter = iter(train_loader)

    with tqdm(total=config.num_updates, desc="Training") as pbar:
        while update_step < config.num_updates:
            # Get next batch (cycle through dataset if needed)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Training step
            metrics = train_step(model, batch, optimizer, config, device)

            # Logging
            if update_step % config.log_interval == 0:
                for key, value in metrics.items():
                    writer.add_scalar(key, value, update_step)

                pbar.set_postfix(
                    {
                        "loss": f"{metrics['loss/total']:.4f}",
                        "semantic": f"{metrics['loss/semantic']:.4f}",
                        "acoustic": f"{metrics['loss/acoustic']:.4f}",
                    }
                )

            # Evaluation
            if update_step % config.eval_interval == 0 and update_step > 0:
                eval_metrics = evaluate(model, eval_loader, config, device)
                for key, value in eval_metrics.items():
                    writer.add_scalar(key, value, update_step)

                print(
                    f"\nUpdate {update_step} - Eval: "
                    f"semantic={eval_metrics['eval/semantic_loss']:.4f}, "
                    f"acoustic={eval_metrics['eval/acoustic_loss']:.4f}, "
                    f"total={eval_metrics['eval/total_loss']:.4f}"
                )

            # Checkpointing
            if update_step % config.checkpoint_interval == 0 and update_step > 0:
                save_checkpoint(model, optimizer, update_step, config, metrics)

            update_step += 1
            pbar.update(1)

    # Final evaluation
    print("\nRunning final evaluation...")
    final_eval_metrics = evaluate(model, eval_loader, config, device)
    print("Final evaluation metrics:")
    for key, value in final_eval_metrics.items():
        print(f"  {key}: {value:.4f}")
        writer.add_scalar(key, value, update_step)

    # Save final checkpoint
    save_checkpoint(model, optimizer, update_step, config, metrics)

    writer.close()
    print("Training complete!")


if __name__ == "__main__":
    main()

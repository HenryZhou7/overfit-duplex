"""
Minimal forward-pass script to drive the MachOverfitModel for VSCode debugging.

Usage examples:
  - Run directly:    python src/overfit_trial/test_scripts/duplex_model_forward_pass.py
  - With options:    python src/overfit_trial/test_scripts/duplex_model_forward_pass.py --frames 32 --device cpu

Notes:
  - This script creates random, shape-correct RVQ token tensors for two channels
    (c1: user, c2: assistant) and feeds them to the model.
  - Set a breakpoint inside overfit_trial/model.py:MachOverfitModel.forward
    around the line that computes `transformer_outputs` to inspect it.
"""

import argparse
import os
import random
import sys
from typing import Tuple

import torch

# Ensure running from repo root or add src to sys.path when invoked externally
REPO_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

from overfit_trial.model import MachOverfitModel  # noqa: E402


def _make_dummy_tokens(
    model: MachOverfitModel, batch_size: int, frames: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create random but shape- and range-correct RVQ token tensors for two channels.

    Returns:
      (c1_tokens, c2_tokens): both shaped (B, num_quantizers, T), dtype long
    """
    # Total quantizers reported by model (semantic + acoustic)
    num_quantizers = model.num_quantizers

    # Codebook sizes for semantic and acoustic quantizers
    sem_code_size = model.quantizer.semantic_residual_vector_quantizer.layers[0].codebook.codebook_size
    ac_code_size = model.code_num

    # Construct per-quantizer codebook sizes: first is semantic, rest acoustic
    per_q_sizes = [sem_code_size] + [ac_code_size] * (num_quantizers - 1)

    def sample_tokens() -> torch.Tensor:
        # Create zeros and fill each quantizer slice with random ints in valid range
        tokens = torch.zeros(batch_size, num_quantizers, frames, dtype=torch.long, device=device)
        for q, size in enumerate(per_q_sizes):
            if size <= 1:
                continue
            tokens[:, q, :] = torch.randint(0, size, (batch_size, frames), device=device)
        return tokens

    return sample_tokens(), sample_tokens()


def main():
    parser = argparse.ArgumentParser(description="Run a single forward pass for debugging.")
    parser.add_argument("--frames", type=int, default=32, help="Number of time frames.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    # Instantiate model
    model = MachOverfitModel()
    model.to(device)
    model.eval()

    # Prepare dummy tokens
    c1_tokens, c2_tokens = _make_dummy_tokens(model, args.batch_size, args.frames, device)

    # Forward pass (set breakpoint inside model.forward to inspect transformer_outputs)
    with torch.inference_mode():
        c0_logits, audio_logits = model(c1_tokens, c2_tokens)

    # Quick shapes summary for sanity
    print("Input shapes:")
    print(f"  c1_tokens: {tuple(c1_tokens.shape)} (B, Q, T)")
    print(f"  c2_tokens: {tuple(c2_tokens.shape)} (B, Q, T)")

    print("\nOutput shapes:")
    # c0 semantic logits: (B, T, semantic_vocab)
    print(f"  c0_logits: {tuple(c0_logits.shape)} (B, T, semantic_vocab)")
    # audio logits per quantizer level: (total_frames, ac_quantizers, acoustic_vocab)
    print(f"  audio_logits: {tuple(audio_logits.shape)} (T*B, Q_ac, acoustic_vocab)")

    print("\nDone. Set a breakpoint in MachOverfitModel.forward to inspect transformer_outputs.")


if __name__ == "__main__":
    main()

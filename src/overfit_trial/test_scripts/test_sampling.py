#!/usr/bin/env python3
"""Test sampling functions."""

import torch

from overfit_trial.model import sample_topk


def test_sampling():
    """Test sampling functions."""
    print("Testing sampling functions...")

    # Create dummy logits
    batch_size = 2
    vocab_size = 100
    logits = torch.randn(batch_size, vocab_size)

    # Test top-k sampling
    for temp in [0.5, 1.0, 2.0]:
        for k in [1, 10, 50]:
            samples = sample_topk(logits, topk=k, temperature=temp)
            assert samples.shape == (batch_size, 1)
            assert torch.all(samples >= 0) and torch.all(samples < vocab_size)
            print(f"✓ Top-k sampling (k={k}, temp={temp})")

    print("✓ All sampling tests passed!")


if __name__ == "__main__":
    test_sampling()

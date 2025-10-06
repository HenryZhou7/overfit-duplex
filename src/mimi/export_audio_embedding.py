"""Export MIMI audio codebook embeddings as tensors.

This script loads the Kyutai MIMI model with a configurable number of
quantizers, collects the per-quantizer codebook embeddings, applies the
quantizer output projection (256 -> 512) to obtain the projected embeddings,
and saves the results to the target location.

Usage examples:
- Save a single .pt file with both tensors:
    uv run python -m mimi.export_audio_embedding -q 32 -o asset/mimi_embeddings_q32.pt

- Save separate files inside a directory:
    uv run python -m mimi.export_audio_embedding -q 32 -o asset/mimi_q32/
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Annotated

import torch
import typer
from transformers import MimiModel
from transformers import logging as hf_logging

from mimi.feature_extraction import MIMI_MODEL_ID


def _collect_codebook_embeddings(model: MimiModel) -> torch.Tensor:
    """Return stacked codebook embeddings with shape (Q, V, D).

    - Q: number of quantizers (1 semantic + acoustic quantizers)
    - V: codebook size (e.g., 2048)
    - D: embedding dim of the codebook (e.g., 256)
    """
    q_sem = model.quantizer.semantic_residual_vector_quantizer
    q_ac = model.quantizer.acoustic_residual_vector_quantizer

    embeddings = [q_sem.layers[0].codebook.embed]
    embeddings += [layer.codebook.embed for layer in q_ac.layers]
    stacked = torch.stack(embeddings, dim=0)  # (Q, V, D)
    return stacked


def _project_embeddings(model: MimiModel, codebook_embeds: torch.Tensor) -> torch.Tensor:
    """Apply semantic output projection to codebook embeddings.

    Input shape: (Q, V, D); Output shape: (Q, V, P)
    """
    out_proj = model.quantizer.semantic_residual_vector_quantizer.output_proj  # Conv1d(D->P, k=1)
    proj = out_proj(codebook_embeds.transpose(1, 2))  # (Q, P, V)
    proj = proj.transpose(1, 2).contiguous()  # (Q, V, P)
    return proj


def _build_payload(
    model: MimiModel, codebook_embeds: torch.Tensor, projected_embeds: torch.Tensor
) -> Dict[str, torch.Tensor | int | str]:
    """Assemble a metadata dict for saving."""
    Q, V, D = codebook_embeds.shape
    _, _, P = projected_embeds.shape
    payload: Dict[str, torch.Tensor | int | str] = {
        "model_id": MIMI_MODEL_ID,
        "num_quantizers": int(Q),
        "codebook_size": int(V),
        "raw_dim": int(D),
        "proj_dim": int(P),
        "codebook_embeddings": codebook_embeds.detach().to(torch.float32).cpu(),
        "projected_embeddings": projected_embeds.detach().to(torch.float32).cpu(),
    }
    return payload


def export_audio_embeddings(num_quantizers: int, out_path: Path) -> None:
    """Export raw and projected audio codebook embeddings to ``out_path``.

    If ``out_path`` has a file suffix (e.g., ".pt"), saves a single file with
    a dictionary containing both `codebook_embeddings` and `projected_embeddings`.
    Otherwise, treats ``out_path`` as a directory and writes two separate files:
      - codebook_embeddings.pt
      - projected_embeddings.pt
    """

    if num_quantizers < 1:
        raise ValueError("num_quantizers must be a positive integer")

    hf_logging.set_verbosity_error()

    # Load model on CPU; GPU is unnecessary for this small computation.
    model = MimiModel.from_pretrained(MIMI_MODEL_ID, num_quantizers=num_quantizers)
    model.eval()

    with torch.inference_mode():
        codebook = _collect_codebook_embeddings(model)  # (Q, V, D)
        if codebook.shape[0] != num_quantizers:
            raise RuntimeError(f"Collected {codebook.shape[0]} quantizers but expected {num_quantizers}")
        projected = _project_embeddings(model, codebook)  # (Q, V, P)

    out_path = Path(out_path)
    if out_path.suffix:  # looks like a file path -> save single consolidated payload
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _build_payload(model, codebook, projected)
        torch.save(payload, out_path)
        return

    # Otherwise, treat as a directory and save separate files
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(codebook.detach().to(torch.float32).cpu(), out_path / f"mimi_codebook_embeddings_{num_quantizers}q.pt")
    torch.save(projected.detach().to(torch.float32).cpu(), out_path / f"mimi_projected_embeddings_{num_quantizers}q.pt")


def main(
    num_quantizers: Annotated[
        int, typer.Option("-q", "--num-quantizers", help="Total quantizers (semantic + acoustic)")
    ],
    out: Annotated[Path, typer.Option("-o", "--out", help="Target .pt file or directory for outputs")],
) -> None:
    """Export embeddings with the given configuration to the target location."""
    export_audio_embeddings(num_quantizers=num_quantizers, out_path=out)


if __name__ == "__main__":
    typer.run(main)

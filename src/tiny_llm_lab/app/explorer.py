"""Framework-neutral next-token explorer service."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tiny_llm_lab.app.formatting import format_token
from tiny_llm_lab.inference import next_token_distribution
from tiny_llm_lab.tokenizer import Tokenizer


@dataclass(frozen=True)
class ExplorerSession:
    model: nn.Module
    tokenizer: Tokenizer
    device: torch.device


@dataclass(frozen=True)
class TokenRow:
    index: int
    token_id: int
    text: str


@dataclass(frozen=True)
class PredictionRow:
    token_id: int
    text: str
    probability: float


@dataclass(frozen=True)
class ExplorerResult:
    tokens: tuple[TokenRow, ...]
    predictions: tuple[PredictionRow, ...]


def inspect_next_token(
    session: ExplorerSession,
    prompt: str,
    *,
    temperature: float,
    display_count: int,
) -> ExplorerResult:
    """Tokenize a prompt and return its top next-token predictions."""
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    vocabulary_size = session.tokenizer.vocabulary_size
    if not 1 <= display_count <= vocabulary_size:
        raise ValueError(f"Display count must be between 1 and {vocabulary_size}")

    token_ids = session.tokenizer.encode(prompt)
    if len(token_ids) > session.model.config.context_length:
        raise ValueError(
            f"Prompt has {len(token_ids)} tokens but the model context limit is "
            f"{session.model.config.context_length}"
        )
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=session.device)
    distribution = next_token_distribution(
        session.model,
        input_ids,
        temperature=temperature,
        top_k=None,
    )
    probabilities, prediction_ids = torch.topk(distribution.probabilities[0], display_count)
    return ExplorerResult(
        tokens=tuple(
            TokenRow(index=index, token_id=token_id, text=format_token(session.tokenizer, token_id))
            for index, token_id in enumerate(token_ids)
        ),
        predictions=tuple(
            PredictionRow(
                token_id=int(token_id),
                text=format_token(session.tokenizer, int(token_id)),
                probability=float(probability),
            )
            for probability, token_id in zip(probabilities.cpu(), prediction_ids.cpu(), strict=True)
        ),
    )

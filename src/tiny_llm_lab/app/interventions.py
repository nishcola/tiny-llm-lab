"""Framework-neutral comparison of baseline and intervention inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tiny_llm_lab.app.explorer import ExplorerSession
from tiny_llm_lab.app.formatting import format_token
from tiny_llm_lab.inference import next_token_distribution
from tiny_llm_lab.interventions import InterventionSet


@dataclass(frozen=True)
class ChangedToken:
    token_id: int
    text: str
    baseline_probability: float
    modified_probability: float
    delta_probability: float


@dataclass(frozen=True)
class InterventionComparison:
    changed_tokens: tuple[ChangedToken, ...]


def compare_intervention(
    session: ExplorerSession,
    prompt: str,
    *,
    intervention: InterventionSet,
    temperature: float,
    display_count: int,
) -> InterventionComparison:
    """Compare full next-token distributions for one prompt and temporary intervention."""
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    if not intervention.enabled:
        raise ValueError("An enabled intervention is required for comparison")
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
    baseline = next_token_distribution(session.model, input_ids, temperature=temperature)
    modified = next_token_distribution(
        session.model,
        input_ids,
        temperature=temperature,
        interventions=intervention,
    )
    deltas = modified.probabilities[0] - baseline.probabilities[0]
    changed_indices = torch.argsort(deltas.abs(), descending=True)[:display_count]
    return InterventionComparison(
        changed_tokens=tuple(
            ChangedToken(
                token_id=int(token_id),
                text=format_token(session.tokenizer, int(token_id)),
                baseline_probability=float(baseline.probabilities[0, token_id]),
                modified_probability=float(modified.probabilities[0, token_id]),
                delta_probability=float(deltas[token_id]),
            )
            for token_id in changed_indices.cpu()
        )
    )

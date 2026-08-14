"""Framework-neutral next-token explorer service."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from tiny_llm_lab.app.formatting import format_token
from tiny_llm_lab.inference import next_token_distribution
from tiny_llm_lab.model import InstrumentationRequest
from tiny_llm_lab.tokenizer import Tokenizer
from tiny_llm_lab.training.checkpoint import (
    TimelineCheckpoint,
    TimelineRun,
    load_timeline_checkpoint,
)


ATTENTION_DISPLAY_LIMIT = 32


@dataclass(frozen=True)
class ExplorerSession:
    model: nn.Module
    tokenizer: Tokenizer
    device: torch.device


class TimelineCheckpointCache:
    """Small on-demand LRU cache for timeline models used by the explorer."""

    def __init__(self, run: TimelineRun, device: torch.device, *, max_entries: int = 2) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.run = run
        self.device = device
        self.max_entries = max_entries
        self._sessions: OrderedDict[int, ExplorerSession] = OrderedDict()

    def load(self, checkpoint: TimelineCheckpoint) -> ExplorerSession:
        if checkpoint.step in self._sessions:
            self._sessions.move_to_end(checkpoint.step)
            return self._sessions[checkpoint.step]
        loaded = load_timeline_checkpoint(self.run, checkpoint, map_location=self.device)
        session = ExplorerSession(loaded.model, loaded.tokenizer, self.device)
        self._sessions[checkpoint.step] = session
        if len(self._sessions) > self.max_entries:
            _, evicted = self._sessions.popitem(last=False)
            del evicted
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return session

    @property
    def cached_steps(self) -> tuple[int, ...]:
        return tuple(self._sessions)


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


@dataclass(frozen=True)
class AttentionView:
    """CPU-ready attention values and labels for one layer/head heatmap."""

    token_labels: tuple[str, ...]
    values: tuple[tuple[float | None, ...], ...]


@dataclass(frozen=True)
class AttentionInspectionResult:
    """One prompt's selected causal-attention matrix and truncation status."""

    view: AttentionView
    was_truncated: bool


@dataclass(frozen=True)
class PromptInspectionResult:
    """Next-token and attention inspection data derived from one prompt."""

    next_token: ExplorerResult
    attention: AttentionInspectionResult


def attention_input_token_ids(
    token_ids: Sequence[int], *, context_length: int
) -> tuple[tuple[int, ...], bool]:
    """Return the prompt prefix that fits a readable causal-attention matrix."""
    if context_length <= 0:
        raise ValueError("Model context length must be positive")
    limit = min(ATTENTION_DISPLAY_LIMIT, context_length)
    visible_ids = tuple(token_ids[:limit])
    return visible_ids, len(token_ids) > len(visible_ids)


def prepare_attention_view(
    tokenizer: Tokenizer,
    token_ids: Sequence[int],
    attention_weights: tuple[Tensor, ...],
    *,
    layer_index: int,
    head_index: int,
) -> AttentionView:
    """Select one captured attention head and make it safe for UI rendering."""
    if not 0 <= layer_index < len(attention_weights):
        raise ValueError(f"Layer index must be between 0 and {len(attention_weights) - 1}")
    layer_weights = attention_weights[layer_index]
    if layer_weights.ndim != 4 or layer_weights.shape[0] != 1:
        raise ValueError("Attention weights must have shape (1, heads, sequence, sequence)")
    if layer_weights.shape[-2] != layer_weights.shape[-1]:
        raise ValueError("Attention weights must be square over sequence positions")
    if layer_weights.shape[-1] != len(token_ids):
        raise ValueError("Attention token count must match the captured sequence length")
    if not 0 <= head_index < layer_weights.shape[1]:
        raise ValueError(f"Head index must be between 0 and {layer_weights.shape[1] - 1}")

    selected_matrix = layer_weights[0, head_index].detach().to(device="cpu", dtype=torch.float32)
    values = selected_matrix.tolist()
    return AttentionView(
        token_labels=tuple(
            f"{index} · {format_token(tokenizer, token_id)}"
            for index, token_id in enumerate(token_ids)
        ),
        values=tuple(
            tuple(value if key_index <= query_index else None for key_index, value in enumerate(row))
            for query_index, row in enumerate(values)
        ),
    )


def inspect_attention(
    session: ExplorerSession,
    prompt: str,
    *,
    layer_index: int,
    head_index: int,
) -> AttentionInspectionResult:
    """Capture and format one selected attention head through the instrumentation API."""
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    token_ids = session.tokenizer.encode(prompt)
    visible_ids, was_truncated = attention_input_token_ids(
        token_ids, context_length=session.model.config.context_length
    )
    input_ids = torch.tensor([visible_ids], dtype=torch.long, device=session.device)
    distribution = next_token_distribution(
        session.model,
        input_ids,
        instrumentation=InstrumentationRequest(attention_weights=True),
    )
    instrumentation = distribution.instrumentation
    if instrumentation is None or instrumentation.attention_weights is None:
        raise ValueError("Model did not return requested attention weights")
    return AttentionInspectionResult(
        view=prepare_attention_view(
            session.tokenizer,
            visible_ids,
            instrumentation.attention_weights,
            layer_index=layer_index,
            head_index=head_index,
        ),
        was_truncated=was_truncated,
    )


def inspect_prompt(
    session: ExplorerSession,
    prompt: str,
    *,
    temperature: float,
    display_count: int,
    layer_index: int,
    head_index: int,
) -> PromptInspectionResult:
    """Inspect predictions and attention while reusing a short prompt's forward pass."""
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
    visible_ids, was_truncated = attention_input_token_ids(
        token_ids, context_length=session.model.config.context_length
    )
    full_input_ids = torch.tensor([token_ids], dtype=torch.long, device=session.device)
    request = InstrumentationRequest(attention_weights=True)
    if was_truncated:
        prediction_distribution = next_token_distribution(
            session.model,
            full_input_ids,
            temperature=temperature,
            top_k=None,
        )
        attention_input_ids = torch.tensor([visible_ids], dtype=torch.long, device=session.device)
        attention_distribution = next_token_distribution(
            session.model,
            attention_input_ids,
            instrumentation=request,
        )
    else:
        prediction_distribution = next_token_distribution(
            session.model,
            full_input_ids,
            temperature=temperature,
            top_k=None,
            instrumentation=request,
        )
        attention_distribution = prediction_distribution
    probabilities, prediction_ids = torch.topk(prediction_distribution.probabilities[0], display_count)
    instrumentation = attention_distribution.instrumentation
    if instrumentation is None or instrumentation.attention_weights is None:
        raise ValueError("Model did not return requested attention weights")
    return PromptInspectionResult(
        next_token=ExplorerResult(
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
                for probability, token_id in zip(
                    probabilities.cpu(), prediction_ids.cpu(), strict=True
                )
            ),
        ),
        attention=AttentionInspectionResult(
            view=prepare_attention_view(
                session.tokenizer,
                visible_ids,
                instrumentation.attention_weights,
                layer_index=layer_index,
                head_index=head_index,
            ),
            was_truncated=was_truncated,
        ),
    )


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

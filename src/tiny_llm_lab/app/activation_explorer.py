"""Framework-neutral inspection of post-GELU MLP hidden units."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from pathlib import Path

import torch

from tiny_llm_lab.app.explorer import ExplorerSession
from tiny_llm_lab.app.formatting import format_token
from tiny_llm_lab.inference import next_token_distribution
from tiny_llm_lab.model import InstrumentationRequest


MLP_REPRESENTATION = "Post-GELU MLP hidden unit before the output projection"
DEFAULT_SCAN_MAX_BYTES = 2 * 1024 * 1024
DEFAULT_SCAN_MAX_TOKENS = 100_000
DEFAULT_SCAN_BATCH_SIZE = 8
DEFAULT_SCAN_RESULT_COUNT = 10
SNIPPET_RADIUS_TOKENS = 16


@dataclass(frozen=True)
class ActivationToken:
    position: int
    token_id: int
    token: str
    value: float
    magnitude: float


@dataclass(frozen=True)
class PromptActivationResult:
    representation: str
    layer_index: int
    unit_index: int
    tokens: tuple[ActivationToken, ...]


@dataclass(frozen=True)
class ActivationMatch:
    token_position: int
    token_id: int
    token: str
    value: float
    snippet: str


@dataclass(frozen=True)
class ActivationScanResult:
    source: Path
    scanned_tokens: int
    matches: tuple[ActivationMatch, ...]


def inspect_mlp_activation(
    session: ExplorerSession, prompt: str, *, layer_index: int, unit_index: int
) -> PromptActivationResult:
    """Return one selected post-GELU MLP unit's value at each prompt token."""
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    _validate_selection(session, layer_index, unit_index)
    token_ids = session.tokenizer.encode(prompt)
    if len(token_ids) > session.model.config.context_length:
        raise ValueError(
            f"Prompt has {len(token_ids)} tokens but the model context limit is "
            f"{session.model.config.context_length}"
        )
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=session.device)
    values = _selected_unit_values(session, input_ids, layer_index, unit_index)[0]
    return PromptActivationResult(
        representation=MLP_REPRESENTATION,
        layer_index=layer_index,
        unit_index=unit_index,
        tokens=tuple(
            ActivationToken(
                position=position,
                token_id=token_id,
                token=format_token(session.tokenizer, token_id),
                value=float(value),
                magnitude=abs(float(value)),
            )
            for position, (token_id, value) in enumerate(zip(token_ids, values.tolist(), strict=True))
        ),
    )


def scan_mlp_activation(
    session: ExplorerSession,
    corpus_path: str | Path,
    *,
    train_fraction: float,
    layer_index: int,
    unit_index: int,
    max_bytes: int = DEFAULT_SCAN_MAX_BYTES,
    max_tokens: int = DEFAULT_SCAN_MAX_TOKENS,
    batch_size: int = DEFAULT_SCAN_BATCH_SIZE,
    result_count: int = DEFAULT_SCAN_RESULT_COUNT,
) -> ActivationScanResult:
    """Scan a bounded training-corpus prefix without retaining corpus-wide activations."""
    _validate_selection(session, layer_index, unit_index)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if min(max_bytes, max_tokens, batch_size, result_count) <= 0:
        raise ValueError("scan bounds and batch size must be positive")
    source = Path(corpus_path)
    try:
        with source.open("rb") as corpus_file:
            text = _decode_bounded_utf8(corpus_file.read(max_bytes))
    except UnicodeDecodeError as error:
        raise ValueError("Training corpus must be valid UTF-8") from error
    except OSError as error:
        raise ValueError(f"Could not read training corpus: {error}") from error
    encoded = session.tokenizer.encode(text)
    train_end = int(len(encoded) * train_fraction)
    token_ids = tuple(encoded[:train_end][:max_tokens])
    if not token_ids:
        raise ValueError("Training corpus scan contains no tokens")

    candidates: list[tuple[float, int, int]] = []
    context_length = session.model.config.context_length
    full_end = len(token_ids) - (len(token_ids) % context_length)
    for range_start in range(0, full_end, context_length * batch_size):
        starts = tuple(range(range_start, min(full_end, range_start + context_length * batch_size), context_length))
        batch = torch.tensor(
            [token_ids[start : start + context_length] for start in starts],
            dtype=torch.long,
            device=session.device,
        )
        _merge_batch_candidates(
            candidates,
            _selected_unit_values(session, batch, layer_index, unit_index),
            starts,
            result_count,
        )
    if full_end < len(token_ids):
        start = full_end
        batch = torch.tensor([token_ids[start:]], dtype=torch.long, device=session.device)
        _merge_batch_candidates(
            candidates,
            _selected_unit_values(session, batch, layer_index, unit_index),
            (start,),
            result_count,
        )

    ordered = sorted(candidates, key=lambda candidate: (-candidate[0], candidate[2]))
    return ActivationScanResult(
        source=source,
        scanned_tokens=len(token_ids),
        matches=tuple(
            ActivationMatch(
                token_position=position,
                token_id=token_ids[position],
                token=format_token(session.tokenizer, token_ids[position]),
                value=value,
                snippet=_snippet(session, token_ids, position),
            )
            for value, _, position in ordered
        ),
    )


def _validate_selection(session: ExplorerSession, layer_index: int, unit_index: int) -> None:
    if not 0 <= layer_index < session.model.config.num_layers:
        raise ValueError(f"Layer must be between 0 and {session.model.config.num_layers - 1}")
    if not 0 <= unit_index < session.model.config.mlp_dim:
        raise ValueError(f"Unit must be between 0 and {session.model.config.mlp_dim - 1}")


def _selected_unit_values(
    session: ExplorerSession, input_ids: torch.Tensor, layer_index: int, unit_index: int
) -> torch.Tensor:
    output = next_token_distribution(
        session.model,
        input_ids,
        instrumentation=InstrumentationRequest(mlp_activation_layer=layer_index),
    )
    instrumentation = output.instrumentation
    if (
        instrumentation is None
        or instrumentation.selected_mlp_activation is None
        or instrumentation.selected_mlp_activation_layer != layer_index
    ):
        raise ValueError("Model did not return the requested MLP activation")
    return instrumentation.selected_mlp_activation[:, :, unit_index].detach().to("cpu", torch.float32)


def _merge_batch_candidates(
    candidates: list[tuple[float, int, int]],
    values: torch.Tensor,
    starts: tuple[int, ...],
    result_count: int,
) -> None:
    local_count = min(result_count, values.numel())
    top_values, flat_indices = torch.topk(values.reshape(-1), local_count)
    sequence_length = values.shape[1]
    for value, flat_index in zip(top_values.tolist(), flat_indices.tolist(), strict=True):
        batch_index, token_index = divmod(flat_index, sequence_length)
        position = starts[batch_index] + token_index
        candidate = (float(value), -position, position)
        if len(candidates) < result_count:
            heapq.heappush(candidates, candidate)
        elif candidate > candidates[0]:
            heapq.heapreplace(candidates, candidate)


def _snippet(session: ExplorerSession, token_ids: tuple[int, ...], position: int) -> str:
    excerpt = token_ids[max(0, position - SNIPPET_RADIUS_TOKENS) : position + SNIPPET_RADIUS_TOKENS + 1]
    try:
        return session.tokenizer.decode(excerpt)
    except ValueError:
        return " ".join(format_token(session.tokenizer, token_id) for token_id in excerpt)


def _decode_bounded_utf8(raw_bytes: bytes) -> str:
    """Decode a prefix while dropping only an incomplete final UTF-8 character."""
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        if error.end == len(raw_bytes) and error.reason == "unexpected end of data":
            return raw_bytes[: error.start].decode("utf-8")
        raise

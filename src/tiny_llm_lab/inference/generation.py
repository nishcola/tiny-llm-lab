"""Framework-neutral next-token inspection and autoregressive generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from tiny_llm_lab.model import InstrumentationRequest, ModelInstrumentation, ModelOutput


class InferenceModel(Protocol):
    config: object
    training: bool

    def eval(self) -> "InferenceModel": ...

    def train(self, mode: bool = True) -> "InferenceModel": ...

    def __call__(self, input_ids: Tensor, *, instrumentation: InstrumentationRequest | None = None) -> ModelOutput: ...


@dataclass(frozen=True)
class NextTokenOutput:
    """Raw logits and sampling probabilities for the final input position."""

    logits: Tensor
    probabilities: Tensor
    instrumentation: ModelInstrumentation | None = None


@dataclass(frozen=True)
class GenerationStep:
    """Inspection data for one sampled continuation token."""

    logits: Tensor
    probabilities: Tensor
    token_ids: Tensor
    instrumentation: ModelInstrumentation | None = None


@dataclass(frozen=True)
class GenerationOutput:
    """Prompt token IDs followed by generated continuation token IDs."""

    token_ids: Tensor
    steps: tuple[GenerationStep, ...] | None = None


def _validate_distribution_arguments(temperature: float, top_k: int | None) -> None:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")


def _sampling_probabilities(logits: Tensor, temperature: float, top_k: int | None) -> Tensor:
    scaled_logits = logits / temperature
    if top_k is not None:
        retained_count = min(top_k, scaled_logits.shape[-1])
        retained_logits, retained_indices = torch.topk(scaled_logits, retained_count, dim=-1)
        filtered_logits = torch.full_like(scaled_logits, float("-inf"))
        filtered_logits.scatter_(-1, retained_indices, retained_logits)
        scaled_logits = filtered_logits
    return torch.softmax(scaled_logits, dim=-1)


def _next_token_distribution(
    model: InferenceModel,
    input_ids: Tensor,
    *,
    temperature: float,
    top_k: int | None,
    instrumentation: InstrumentationRequest | None,
) -> NextTokenOutput:
    output = model(input_ids, instrumentation=instrumentation)
    logits = output.logits[:, -1, :]
    return NextTokenOutput(
        logits=logits,
        probabilities=_sampling_probabilities(logits, temperature, top_k),
        instrumentation=output.instrumentation,
    )


def next_token_distribution(
    model: InferenceModel,
    input_ids: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    instrumentation: InstrumentationRequest | None = None,
) -> NextTokenOutput:
    """Inspect final-position logits and the distribution used for sampling."""
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have shape (batch, sequence) with at least one token")
    _validate_distribution_arguments(temperature, top_k)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            return _next_token_distribution(
                model,
                input_ids,
                temperature=temperature,
                top_k=top_k,
                instrumentation=instrumentation,
            )
    finally:
        model.train(was_training)


def generate(
    model: InferenceModel,
    input_ids: Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    do_sample: bool = True,
    generator: torch.Generator | None = None,
    return_trace: bool = False,
    instrumentation: InstrumentationRequest | None = None,
) -> GenerationOutput:
    """Autoregressively generate tokens, optionally retaining per-step inspection data."""
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have shape (batch, sequence) with at least one token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    _validate_distribution_arguments(temperature, top_k)
    if instrumentation is not None and not return_trace:
        raise ValueError("return_trace=True is required when generation instrumentation is requested")

    generated = input_ids.clone()
    steps: list[GenerationStep] = []
    context_length = model.config.context_length
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                context = generated[:, -context_length:]
                distribution = _next_token_distribution(
                    model,
                    context,
                    temperature=temperature,
                    top_k=top_k,
                    instrumentation=instrumentation,
                )
                if do_sample:
                    next_token_ids = torch.multinomial(
                        distribution.probabilities,
                        num_samples=1,
                        generator=generator,
                    )
                else:
                    next_token_ids = distribution.probabilities.argmax(dim=-1, keepdim=True)
                generated = torch.cat((generated, next_token_ids), dim=1)
                if return_trace:
                    steps.append(
                        GenerationStep(
                            logits=distribution.logits,
                            probabilities=distribution.probabilities,
                            token_ids=next_token_ids,
                            instrumentation=distribution.instrumentation,
                        )
                    )
    finally:
        model.train(was_training)

    return GenerationOutput(token_ids=generated, steps=tuple(steps) if return_trace else None)

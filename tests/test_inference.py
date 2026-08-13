from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from tiny_llm_lab.config import ModelConfig
from tiny_llm_lab.inference import generate, next_token_distribution
from tiny_llm_lab.model import (
    DecoderOnlyTransformer,
    InstrumentationRequest,
    ModelOutput,
)


def small_config() -> ModelConfig:
    return ModelConfig(
        vocabulary_size=7,
        context_length=5,
        embedding_dim=12,
        num_layers=2,
        num_heads=3,
        mlp_dim=24,
        dropout=0.0,
    )


class FixedLogitModel(nn.Module):
    """A real module with a predictable distribution for inference tests."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(context_length=5)
        self.register_buffer("fixed_logits", torch.tensor([0.0, 1.0, 2.0, 3.0]))

    def forward(self, input_ids: Tensor, *, instrumentation: object | None = None) -> ModelOutput:
        batch_size, sequence_length = input_ids.shape
        logits = self.fixed_logits.expand(batch_size, sequence_length, -1)
        return ModelOutput(logits=logits)


def test_next_token_distribution_applies_temperature_and_top_k() -> None:
    model = FixedLogitModel()
    input_ids = torch.tensor([[0, 1]])

    baseline = next_token_distribution(model, input_ids, temperature=1.0)
    cold = next_token_distribution(model, input_ids, temperature=0.5)
    restricted = next_token_distribution(model, input_ids, top_k=2)

    torch.testing.assert_close(baseline.logits, torch.tensor([[0.0, 1.0, 2.0, 3.0]]))
    assert cold.probabilities[0, 3] > baseline.probabilities[0, 3]
    torch.testing.assert_close(restricted.probabilities[:, :2], torch.zeros(1, 2))
    torch.testing.assert_close(restricted.probabilities.sum(dim=-1), torch.ones(1))


def test_top_k_sampling_never_selects_excluded_tokens() -> None:
    result = generate(
        FixedLogitModel(),
        torch.tensor([[0]]),
        max_new_tokens=4,
        top_k=1,
        do_sample=True,
        generator=torch.Generator().manual_seed(9),
    )

    assert torch.equal(result.token_ids, torch.tensor([[0, 3, 3, 3, 3]]))


def test_greedy_generation_is_deterministic_and_preserves_prompt() -> None:
    model = DecoderOnlyTransformer(small_config()).train()
    prompt = torch.tensor([[1, 2]])

    first = generate(model, prompt, max_new_tokens=4, do_sample=False)
    second = generate(model, prompt, max_new_tokens=4, do_sample=False)

    assert torch.equal(first.token_ids, second.token_ids)
    assert torch.equal(first.token_ids[:, :2], prompt)
    assert first.token_ids.shape == (1, 6)
    assert model.training


def test_generation_trace_contains_step_distributions_and_requested_captures() -> None:
    model = DecoderOnlyTransformer(small_config()).eval()
    request = InstrumentationRequest(attention_weights=True, hidden_states=True)

    result = generate(
        model,
        torch.tensor([[1, 2]]),
        max_new_tokens=2,
        do_sample=False,
        return_trace=True,
        instrumentation=request,
    )

    assert result.steps is not None
    assert len(result.steps) == 2
    assert result.steps[0].logits.shape == (1, 7)
    assert result.steps[0].probabilities.shape == (1, 7)
    assert result.steps[0].instrumentation is not None


def test_model_instrumentation_has_requested_semantic_tensor_shapes() -> None:
    model = DecoderOnlyTransformer(small_config()).eval()
    request = InstrumentationRequest(
        attention_weights=True,
        hidden_states=True,
        attention_outputs=True,
        mlp_activations=True,
    )

    output = model(torch.randint(0, 7, (2, 4)), instrumentation=request)

    assert output.instrumentation is not None
    captured = output.instrumentation
    assert captured.attention_weights is not None
    assert len(captured.attention_weights) == 2
    assert captured.attention_weights[0].shape == (2, 3, 4, 4)
    assert captured.hidden_states is not None
    assert len(captured.hidden_states) == 3
    assert captured.hidden_states[0].shape == (2, 4, 12)
    assert captured.attention_outputs is not None
    assert captured.attention_outputs[0].shape == (2, 4, 12)
    assert captured.mlp_activations is not None
    assert captured.mlp_activations[0].shape == (2, 4, 24)
    assert not captured.hidden_states[0].requires_grad


def test_normal_forward_does_not_return_instrumentation() -> None:
    output = DecoderOnlyTransformer(small_config())(torch.randint(0, 7, (2, 4)))

    assert output.instrumentation is None


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"max_new_tokens": -1}, "max_new_tokens"),
        ({"max_new_tokens": 1, "temperature": 0.0}, "temperature"),
        ({"max_new_tokens": 1, "top_k": 0}, "top_k"),
        ({"max_new_tokens": 1, "instrumentation": InstrumentationRequest()}, "return_trace"),
    ],
)
def test_generate_rejects_invalid_arguments(arguments: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        generate(DecoderOnlyTransformer(small_config()), torch.tensor([[1]]), **arguments)


def test_generation_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="at least one token"):
        generate(DecoderOnlyTransformer(small_config()), torch.empty(1, 0, dtype=torch.long), max_new_tokens=1)

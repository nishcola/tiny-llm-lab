import pytest
import torch

from tiny_llm_lab.config import ModelConfig
from tiny_llm_lab.model import CausalSelfAttention, DecoderOnlyTransformer, ModelOutput


def small_config() -> ModelConfig:
    return ModelConfig(
        vocabulary_size=11,
        context_length=8,
        embedding_dim=16,
        num_layers=2,
        num_heads=4,
        mlp_dim=32,
        dropout=0.0,
    )


def test_attention_shapes_and_causal_mask() -> None:
    attention = CausalSelfAttention(small_config()).eval()
    inputs = torch.randn(2, 5, 16)

    outputs, weights = attention(inputs, return_attention=True)

    assert outputs.shape == (2, 5, 16)
    assert weights is not None
    assert weights.shape == (2, 4, 5, 5)
    future_positions = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    assert torch.count_nonzero(weights[:, :, future_positions]) == 0
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 4, 5))


def test_model_forward_shapes_and_optional_attentions() -> None:
    model = DecoderOnlyTransformer(small_config()).eval()
    input_ids = torch.randint(0, 11, (2, 6))

    output = model(input_ids, return_attentions=True)

    assert isinstance(output, ModelOutput)
    assert output.logits.shape == (2, 6, 11)
    assert output.loss is None
    assert output.attentions is not None
    assert len(output.attentions) == 2
    assert output.attentions[0].shape == (2, 4, 6, 6)


def test_model_computes_finite_loss() -> None:
    model = DecoderOnlyTransformer(small_config())
    input_ids = torch.randint(0, 11, (2, 6))
    targets = torch.randint(0, 11, (2, 6))

    output = model(input_ids, targets)

    assert output.loss is not None
    assert torch.isfinite(output.loss)


def test_model_rejects_sequence_longer_than_context() -> None:
    model = DecoderOnlyTransformer(small_config())

    with pytest.raises(ValueError, match="context length"):
        model(torch.randint(0, 11, (1, 9)))


def test_model_requires_vocabulary_size() -> None:
    config = ModelConfig(
        context_length=8,
        embedding_dim=16,
        num_layers=2,
        num_heads=4,
        mlp_dim=32,
    )

    with pytest.raises(ValueError, match="vocabulary_size"):
        DecoderOnlyTransformer(config)


import torch

from tiny_llm_lab.app.explorer import ExplorerSession
from tiny_llm_lab.app.interventions import compare_intervention
from tiny_llm_lab.config import ModelConfig
from tiny_llm_lab.inference import generate, next_token_distribution
from tiny_llm_lab.interventions import DisableAttentionHead, InterventionSet, ScaleMLPActivation
from tiny_llm_lab.model import DecoderOnlyTransformer
from tiny_llm_lab.tokenizer import CharacterTokenizer


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


def test_no_intervention_set_matches_baseline_logits_exactly() -> None:
    model = DecoderOnlyTransformer(small_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])

    baseline = model(input_ids).logits
    unchanged = model(input_ids, interventions=InterventionSet()).logits

    torch.testing.assert_close(unchanged, baseline, rtol=0.0, atol=0.0)


def test_empty_intervention_set_matches_baseline_inference_exactly() -> None:
    model = DecoderOnlyTransformer(small_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])

    baseline = next_token_distribution(model, input_ids)
    unchanged = next_token_distribution(model, input_ids, interventions=InterventionSet())

    torch.testing.assert_close(unchanged.logits, baseline.logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(unchanged.probabilities, baseline.probabilities, rtol=0.0, atol=0.0)


def test_interventions_do_not_mutate_checkpoint_weights() -> None:
    model = DecoderOnlyTransformer(small_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    checkpoint_before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    model(
        input_ids,
        interventions=InterventionSet(
            DisableAttentionHead(layer_index=0, head_index=1),
            ScaleMLPActivation(layer_index=1, unit_index=3, scale=0.0),
        ),
    )

    checkpoint_after = model.state_dict()
    for name, value in checkpoint_before.items():
        torch.testing.assert_close(checkpoint_after[name], value, rtol=0.0, atol=0.0)


def test_disabling_attention_head_removes_only_that_head_output() -> None:
    config = ModelConfig(
        vocabulary_size=4,
        context_length=2,
        embedding_dim=4,
        num_layers=1,
        num_heads=2,
        mlp_dim=8,
        dropout=0.0,
    )
    attention = DecoderOnlyTransformer(config).blocks[0].attention.eval()
    with torch.no_grad():
        attention.query_key_value.weight.zero_()
        attention.query_key_value.bias.zero_()
        attention.query_key_value.weight[8:, :] = torch.eye(4)
        attention.output_projection.weight.copy_(torch.eye(4))
        attention.output_projection.bias.zero_()
    inputs = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

    baseline, baseline_weights = attention(inputs, return_attention=True)
    modified, modified_weights = attention(
        inputs,
        return_attention=True,
        interventions=InterventionSet(DisableAttentionHead(layer_index=0, head_index=1)),
        layer_index=0,
    )

    assert baseline_weights is not None
    assert modified_weights is not None
    torch.testing.assert_close(modified_weights, baseline_weights, rtol=0.0, atol=0.0)
    torch.testing.assert_close(baseline, inputs)
    torch.testing.assert_close(modified, torch.tensor([[[1.0, 2.0, 0.0, 0.0]]]))


def test_mlp_activation_scaling_changes_only_selected_unit_at_every_position() -> None:
    activations = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    interventions = InterventionSet(ScaleMLPActivation(layer_index=1, unit_index=1, scale=0.5))

    modified = interventions.apply_mlp_activations(layer_index=1, activations=activations)
    untouched_layer = interventions.apply_mlp_activations(layer_index=0, activations=activations)

    torch.testing.assert_close(
        modified,
        torch.tensor([[[1.0, 1.0, 3.0], [4.0, 2.5, 6.0]]]),
    )
    torch.testing.assert_close(untouched_layer, activations)
    torch.testing.assert_close(activations, torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]))


def test_comparison_ranks_changed_tokens_and_is_deterministic() -> None:
    model = DecoderOnlyTransformer(small_config()).eval()
    tokenizer = CharacterTokenizer.from_text("abcdefghijk")
    session = ExplorerSession(model, tokenizer, torch.device("cpu"))
    intervention = InterventionSet(ScaleMLPActivation(layer_index=0, unit_index=0, scale=0.0))

    first = compare_intervention(
        session, "ab", intervention=intervention, temperature=1.0, display_count=4
    )
    second = compare_intervention(
        session, "ab", intervention=intervention, temperature=1.0, display_count=4
    )

    assert len(first.changed_tokens) == 4
    assert [row.token_id for row in first.changed_tokens] == [row.token_id for row in second.changed_tokens]
    assert [row.delta_probability for row in first.changed_tokens] == [row.delta_probability for row in second.changed_tokens]
    assert [abs(row.delta_probability) for row in first.changed_tokens] == sorted(
        (abs(row.delta_probability) for row in first.changed_tokens), reverse=True
    )


def test_greedy_generation_with_intervention_is_deterministic() -> None:
    model = DecoderOnlyTransformer(small_config()).eval()
    intervention = InterventionSet(DisableAttentionHead(layer_index=0, head_index=0))
    prompt = torch.tensor([[1, 2]])

    first = generate(model, prompt, max_new_tokens=3, do_sample=False, interventions=intervention)
    second = generate(model, prompt, max_new_tokens=3, do_sample=False, interventions=intervention)

    torch.testing.assert_close(first.token_ids, second.token_ids, rtol=0.0, atol=0.0)

import pytest
import torch
from torch import Tensor, nn

from tiny_llm_lab.app.explorer import (
    ATTENTION_DISPLAY_LIMIT,
    ExplorerSession,
    attention_input_token_ids,
    inspect_attention,
    inspect_prompt,
    prepare_attention_view,
)
from tiny_llm_lab.model import InstrumentationRequest, ModelInstrumentation, ModelOutput
from tiny_llm_lab.app.streamlit_page import render_attention_heatmap
from tiny_llm_lab.tokenizer import CharacterTokenizer


def test_prepare_attention_view_selects_layer_head_and_masks_future_tokens() -> None:
    tokenizer = CharacterTokenizer.from_text("ab ")
    captures = (
        torch.zeros(1, 2, 3, 3),
        torch.tensor(
            [
                [
                    [[1.0, 0.0, 0.0], [0.25, 0.75, 0.0], [0.1, 0.2, 0.7]],
                    [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.2, 0.3, 0.5]],
                ]
            ]
        ),
    )

    view = prepare_attention_view(tokenizer, [0, 1, 2], captures, layer_index=1, head_index=1)

    assert view.token_labels == ("0 · ␠", "1 · a", "2 · b")
    assert view.values[0] == (1.0, None, None)
    assert view.values[1] == (0.5, 0.5, None)
    assert view.values[2] == pytest.approx((0.2, 0.3, 0.5))


@pytest.mark.parametrize(
    ("layer_index", "head_index", "message"),
    [(-1, 0, "Layer"), (2, 0, "Layer"), (0, -1, "Head"), (0, 2, "Head")],
)
def test_prepare_attention_view_rejects_invalid_layer_or_head(
    layer_index: int, head_index: int, message: str
) -> None:
    tokenizer = CharacterTokenizer.from_text("ab")
    captures = (torch.ones(1, 2, 2, 2),)

    with pytest.raises(ValueError, match=message):
        prepare_attention_view(tokenizer, [0, 1], captures, layer_index=layer_index, head_index=head_index)


def test_prepare_attention_view_rejects_token_and_matrix_length_mismatch() -> None:
    tokenizer = CharacterTokenizer.from_text("ab")

    with pytest.raises(ValueError, match="token count"):
        prepare_attention_view(
            tokenizer,
            [0, 1],
            (torch.ones(1, 1, 3, 3),),
            layer_index=0,
            head_index=0,
        )


def test_attention_input_token_ids_caps_at_display_or_context_limit() -> None:
    token_ids = list(range(40))

    visible, was_truncated = attention_input_token_ids(token_ids, context_length=64)
    assert visible == tuple(range(ATTENTION_DISPLAY_LIMIT))
    assert was_truncated

    visible, was_truncated = attention_input_token_ids(token_ids, context_length=7)
    assert visible == tuple(range(7))
    assert was_truncated


class AttentionCaptureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"context_length": 64, "num_layers": 1, "num_heads": 2})()
        self.last_input_ids: Tensor | None = None
        self.last_instrumentation: InstrumentationRequest | None = None
        self.call_count = 0

    def forward(
        self, input_ids: Tensor, *, instrumentation: InstrumentationRequest | None = None
    ) -> ModelOutput:
        self.call_count += 1
        self.last_input_ids = input_ids
        self.last_instrumentation = instrumentation
        sequence_length = input_ids.shape[1]
        weights = torch.tril(torch.ones(1, 2, sequence_length, sequence_length))
        return ModelOutput(
            logits=torch.zeros(1, sequence_length, 2),
            instrumentation=ModelInstrumentation(attention_weights=(weights,)),
        )


def test_inspect_attention_requests_instrumentation_and_limits_model_input() -> None:
    model = AttentionCaptureModel()
    session = ExplorerSession(model, CharacterTokenizer.from_text("ab"), torch.device("cpu"))

    result = inspect_attention(session, "ab" * 20, layer_index=0, head_index=1)

    assert model.last_instrumentation == InstrumentationRequest(attention_weights=True)
    assert model.last_input_ids is not None
    assert model.last_input_ids.shape == (1, ATTENTION_DISPLAY_LIMIT)
    assert result.was_truncated
    assert result.view.token_labels[0] == "0 · a"


def test_inspect_prompt_reuses_one_instrumented_forward_for_short_prompts() -> None:
    model = AttentionCaptureModel()
    session = ExplorerSession(model, CharacterTokenizer.from_text("ab"), torch.device("cpu"))

    result = inspect_prompt(
        session,
        "ab",
        temperature=1.0,
        display_count=1,
        layer_index=0,
        head_index=1,
    )

    assert model.call_count == 1
    assert model.last_instrumentation == InstrumentationRequest(attention_weights=True)
    assert result.next_token.predictions[0].token_id == 0
    assert not result.attention.was_truncated


class FakeChartPage:
    def __init__(self) -> None:
        self.figures: list[object] = []

    def plotly_chart(self, figure: object, **kwargs: object) -> None:
        self.figures.append(figure)


def test_attention_heatmap_hides_future_cells_and_enables_exact_hover() -> None:
    view = prepare_attention_view(
        CharacterTokenizer.from_text("ab"),
        [0, 1],
        (torch.tensor([[[[1.0, 0.0], [0.4, 0.6]]]]),),
        layer_index=0,
        head_index=0,
    )
    page = FakeChartPage()

    render_attention_heatmap(page, view)

    assert len(page.figures) == 1
    figure = page.figures[0]
    assert figure.data[0].hoverongaps is False
    assert figure.data[0].z[0][1] is None
    assert "Attention" in figure.data[0].hovertemplate
    assert figure.layout.xaxis.side == "top"

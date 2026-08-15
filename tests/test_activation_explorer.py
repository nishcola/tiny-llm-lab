from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from tiny_llm_lab.app.activation_explorer import inspect_mlp_activation, scan_mlp_activation
from tiny_llm_lab.app.explorer import ExplorerSession
from tiny_llm_lab.app.streamlit_page import render_activation_explorer
from tiny_llm_lab.config import DataConfig, ExperimentConfig, ModelConfig
from tiny_llm_lab.model import InstrumentationRequest, ModelInstrumentation, ModelOutput
from tiny_llm_lab.tokenizer import CharacterTokenizer


class TokenValueActivationModel(nn.Module):
    """Exposes each token ID as every selected MLP-unit activation."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(context_length=3, num_layers=1, mlp_dim=4)
        self.register_buffer("logits", torch.zeros(1, 1, 3))
        self.batch_sizes: list[int] = []

    def forward(
        self, input_ids: Tensor, *, instrumentation: InstrumentationRequest | None = None
    ) -> ModelOutput:
        self.batch_sizes.append(input_ids.shape[0])
        selected = None
        selected_layer = None
        if instrumentation is not None and instrumentation.mlp_activation_layer is not None:
            selected = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, self.config.mlp_dim)
            selected_layer = instrumentation.mlp_activation_layer
        return ModelOutput(
            logits=self.logits.expand(input_ids.shape[0], input_ids.shape[1], -1),
            instrumentation=ModelInstrumentation(
                selected_mlp_activation=selected,
                selected_mlp_activation_layer=selected_layer,
            ) if selected is not None else None,
        )


def test_prompt_activation_inspection_reports_signed_values_and_magnitudes() -> None:
    tokenizer = CharacterTokenizer.from_text("abc")
    session = ExplorerSession(TokenValueActivationModel(), tokenizer, torch.device("cpu"))

    result = inspect_mlp_activation(session, "cab", layer_index=0, unit_index=2)

    assert result.representation == "Post-GELU MLP hidden unit before the output projection"
    assert [(point.token_id, point.value, point.magnitude) for point in result.tokens] == [
        (2, 2.0, 2.0),
        (0, 0.0, 0.0),
        (1, 1.0, 1.0),
    ]


@pytest.mark.parametrize(("layer_index", "unit_index", "message"), [(1, 0, "Layer"), (0, 4, "Unit")])
def test_prompt_activation_inspection_rejects_invalid_selections(
    layer_index: int, unit_index: int, message: str
) -> None:
    session = ExplorerSession(TokenValueActivationModel(), CharacterTokenizer.from_text("abc"), torch.device("cpu"))

    with pytest.raises(ValueError, match=message):
        inspect_mlp_activation(session, "ab", layer_index=layer_index, unit_index=unit_index)


def test_corpus_scan_keeps_only_the_strongest_bounded_matches(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abc" * 6, encoding="utf-8")
    model = TokenValueActivationModel()
    session = ExplorerSession(model, CharacterTokenizer.from_text("abc"), torch.device("cpu"))

    result = scan_mlp_activation(
        session,
        corpus,
        train_fraction=0.9,
        layer_index=0,
        unit_index=0,
        max_bytes=1024,
        max_tokens=8,
        batch_size=2,
        result_count=2,
    )

    assert result.scanned_tokens == 8
    assert [(match.token_position, match.value) for match in result.matches] == [(2, 2.0), (5, 2.0)]
    assert result.matches[0].snippet == "abcabcab"
    assert max(model.batch_sizes) <= 2


def test_corpus_scan_breaks_equal_activation_ties_by_earlier_token_position(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcabcabc", encoding="utf-8")
    session = ExplorerSession(TokenValueActivationModel(), CharacterTokenizer.from_text("abc"), torch.device("cpu"))

    result = scan_mlp_activation(
        session, corpus, train_fraction=0.9, layer_index=0, unit_index=0,
        max_bytes=1024, max_tokens=16, batch_size=1, result_count=1,
    )

    assert result.matches[0].token_position == 2


def test_corpus_scan_tolerates_a_valid_utf8_character_cut_by_the_byte_bound(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcé", encoding="utf-8")
    session = ExplorerSession(TokenValueActivationModel(), CharacterTokenizer.from_text("abcé"), torch.device("cpu"))

    result = scan_mlp_activation(
        session, corpus, train_fraction=0.9, layer_index=0, unit_index=0,
        max_bytes=4, max_tokens=10, batch_size=1, result_count=1,
    )

    assert result.scanned_tokens == 2


def test_corpus_scan_reports_a_missing_training_corpus() -> None:
    session = ExplorerSession(TokenValueActivationModel(), CharacterTokenizer.from_text("abc"), torch.device("cpu"))

    with pytest.raises(ValueError, match="Could not read training corpus"):
        scan_mlp_activation(
            session, "missing-corpus.txt", train_fraction=0.9, layer_index=0, unit_index=0
        )


class ActivationPage:
    def __init__(self) -> None:
        self.captions: list[str] = []
        self.selectboxes: list[str] = []
        self.figures: list[object] = []

    def subheader(self, value: str) -> None: pass
    def caption(self, value: str) -> None: self.captions.append(value)
    def selectbox(self, label: str, options: range, **kwargs: object) -> int:
        self.selectboxes.append(label)
        return next(iter(options))
    def plotly_chart(self, figure: object, **kwargs: object) -> None: self.figures.append(figure)
    def info(self, value: str) -> None: pass
    def error(self, value: str) -> None: raise AssertionError(value)


def test_activation_view_renders_exploratory_label_controls_and_token_plot() -> None:
    page = ActivationPage()
    session = ExplorerSession(TokenValueActivationModel(), CharacterTokenizer.from_text("abc"), torch.device("cpu"))

    render_activation_explorer(page, session, "cab")

    assert page.selectboxes == ["MLP layer", "MLP hidden unit"]
    assert any("exploratory" in caption.lower() for caption in page.captions)
    assert len(page.figures) == 1


class ScanActivationPage(ActivationPage):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[dict[str, object]]] = []

    def button(self, label: str) -> bool: return label == "Scan training corpus"
    def dataframe(self, value: list[dict[str, object]], **kwargs: object) -> None: self.tables.append(value)


def test_activation_view_scans_the_checkpoint_training_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcabcabc", encoding="utf-8")
    page = ScanActivationPage()
    config = ExperimentConfig(
        model=ModelConfig(vocabulary_size=3, context_length=3, embedding_dim=3, num_layers=1, num_heads=1, mlp_dim=4),
        data=DataConfig(path=corpus, train_fraction=0.9),
    )
    session = ExplorerSession(
        TokenValueActivationModel(), CharacterTokenizer.from_text("abc"), torch.device("cpu"), config
    )

    render_activation_explorer(page, session, "cab")

    assert page.tables[0][0]["Token position"] == 2
    assert page.tables[0][0]["Activation"] == "2.000000"

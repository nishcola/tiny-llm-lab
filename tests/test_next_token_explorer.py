from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from tiny_llm_lab.app.explorer import ExplorerSession, inspect_next_token
from tiny_llm_lab.app.formatting import format_token
from tiny_llm_lab.app.streamlit_page import render_explorer
from tiny_llm_lab.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from tiny_llm_lab.data import DatasetMetadata
from tiny_llm_lab.model import DecoderOnlyTransformer, ModelOutput
from tiny_llm_lab.tokenizer import BytePairTokenizer, CharacterTokenizer
from tiny_llm_lab.training.checkpoint import load_inference_checkpoint, save_checkpoint


class FixedLogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(context_length=3, vocabulary_size=4)
        self.register_buffer("fixed_logits", torch.tensor([0.0, 1.0, 2.0, 3.0]))

    def forward(self, input_ids: Tensor, *, instrumentation: object | None = None) -> ModelOutput:
        batch_size, sequence_length = input_ids.shape
        return ModelOutput(logits=self.fixed_logits.expand(batch_size, sequence_length, -1))


def test_explorer_returns_token_rows_and_probability_ranked_predictions() -> None:
    session = ExplorerSession(
        model=FixedLogitModel(),
        tokenizer=CharacterTokenizer.from_text("abcd"),
        device=torch.device("cpu"),
    )

    result = inspect_next_token(session, "ab", temperature=1.0, display_count=2)

    assert [(row.index, row.token_id, row.text) for row in result.tokens] == [(0, 0, "a"), (1, 1, "b")]
    assert [row.token_id for row in result.predictions] == [3, 2]
    assert result.predictions[0].probability > result.predictions[1].probability


def test_explorer_temperature_changes_probabilities_but_display_count_only_limits_rows() -> None:
    session = ExplorerSession(
        model=FixedLogitModel(),
        tokenizer=CharacterTokenizer.from_text("abcd"),
        device=torch.device("cpu"),
    )

    cold = inspect_next_token(session, "a", temperature=0.5, display_count=1)
    baseline = inspect_next_token(session, "a", temperature=1.0, display_count=4)

    assert len(cold.predictions) == 1
    assert len(baseline.predictions) == 4
    assert cold.predictions[0].token_id == baseline.predictions[0].token_id == 3
    assert cold.predictions[0].probability > baseline.predictions[0].probability


@pytest.mark.parametrize(("prompt", "message"), [("", "empty"), ("abcd", "context")])
def test_explorer_rejects_empty_and_over_context_prompts(prompt: str, message: str) -> None:
    session = ExplorerSession(
        model=FixedLogitModel(),
        tokenizer=CharacterTokenizer.from_text("abcd"),
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match=message):
        inspect_next_token(session, prompt, temperature=1.0, display_count=2)


def test_token_formatting_makes_whitespace_controls_unicode_and_byte_pieces_readable() -> None:
    characters = CharacterTokenizer.from_text(" \n\t\x07é")
    byte_pair = BytePairTokenizer.train("hello", vocabulary_size=256)

    assert format_token(characters, characters.encode(" ")[0]) == "␠"
    assert format_token(characters, characters.encode("\n")[0]) == "↵"
    assert format_token(characters, characters.encode("\t")[0]) == "⇥"
    assert format_token(characters, characters.encode("\x07")[0]) == "\\x07"
    assert format_token(characters, characters.encode("é")[0]) == "é"
    assert format_token(byte_pair, 195) == "bytes: 0xc3"


def test_inference_checkpoint_loader_reconstructs_model_and_tokenizer(tmp_path: Path) -> None:
    config = ExperimentConfig(
        model=ModelConfig(vocabulary_size=4, context_length=4, embedding_dim=12, num_layers=1, num_heads=3, mlp_dim=24),
        data=DataConfig(path=tmp_path / "corpus.txt"),
        training=TrainingConfig(device="cpu", max_steps=1, output_dir=tmp_path / "checkpoints"),
    )
    tokenizer = CharacterTokenizer.from_text("abcd")
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        model=DecoderOnlyTransformer(config.model),
        optimizer=torch.optim.AdamW(DecoderOnlyTransformer(config.model).parameters(), lr=1e-3),
        tokenizer=tokenizer,
        config=config,
        step=3,
        validation_loss=1.0,
        dataset_metadata=DatasetMetadata(source="fixture", byte_count=4, sha256="digest"),
    )

    loaded = load_inference_checkpoint(checkpoint, map_location="cpu")

    assert loaded.model.config == config.model
    assert loaded.model.training is False
    assert loaded.tokenizer == tokenizer
    assert loaded.step == 3


class FakeStreamlit:
    def __init__(self) -> None:
        self.tables: list[list[dict[str, object]]] = []
        self.table_options: list[dict[str, object]] = []
        self.errors: list[str] = []

    def title(self, value: str) -> None: pass
    def caption(self, value: str) -> None: pass
    def text_area(self, label: str, *, value: str) -> str: return value
    def number_input(self, label: str, **kwargs: object) -> object: return kwargs["value"]
    def subheader(self, value: str) -> None: pass
    def dataframe(self, value: list[dict[str, object]], **kwargs: object) -> None:
        self.tables.append(value)
        self.table_options.append(kwargs)
    def error(self, value: str) -> None: self.errors.append(value)


def test_streamlit_page_renders_prompt_controls_token_and_prediction_tables() -> None:
    page = FakeStreamlit()
    session = ExplorerSession(FixedLogitModel(), CharacterTokenizer.from_text("abcd"), torch.device("cpu"))

    render_explorer(page, session, default_prompt="ab")

    assert len(page.tables) == 2
    assert page.tables[0][0] == {"Token ID": 0, "Token text": "a"}
    assert page.tables[1][0]["Token"] == "d"
    assert page.tables[1][0]["Probability"].endswith("%")
    assert all(options["width"] == "stretch" for options in page.table_options)
    assert not page.errors

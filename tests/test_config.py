from pathlib import Path

import pytest

from tiny_llm_lab.config import ExperimentConfig, ModelConfig, TokenizerConfig, load_config


def test_load_config_resolves_expected_values(tmp_path: Path) -> None:
    config_path = tmp_path / "test.toml"
    config_path.write_text(
        """
[model]
context_length = 8
embedding_dim = 16
num_layers = 2
num_heads = 4
mlp_dim = 32
dropout = 0.0

[data]
path = "corpus.txt"
train_fraction = 0.8

[training]
max_steps = 3
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert isinstance(config, ExperimentConfig)
    assert config.model.embedding_dim == 16
    assert config.data.path == Path("corpus.txt")
    assert config.training.max_steps == 3


def test_model_config_rejects_incompatible_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(
            context_length=8,
            embedding_dim=10,
            num_layers=2,
            num_heads=4,
            mlp_dim=32,
        )


def test_model_config_defaults_to_learned_positions_and_validates_position_encoding() -> None:
    config = ModelConfig(
        context_length=8,
        embedding_dim=16,
        num_layers=2,
        num_heads=4,
        mlp_dim=32,
    )

    assert config.position_encoding == "learned"
    with pytest.raises(ValueError, match="position_encoding"):
        ModelConfig(
            context_length=8,
            embedding_dim=16,
            num_layers=2,
            num_heads=4,
            mlp_dim=32,
            position_encoding="rotary",
        )


def test_tokenizer_config_defaults_to_bpe_and_accepts_character_mode() -> None:
    assert TokenizerConfig().kind == "byte_pair"
    assert TokenizerConfig(kind="character").kind == "character"
    with pytest.raises(ValueError, match="kind"):
        TokenizerConfig(kind="wordpiece")

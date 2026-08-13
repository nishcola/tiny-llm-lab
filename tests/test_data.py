from pathlib import Path

import pytest
import torch

from tiny_llm_lab.data import TextDataset
from tiny_llm_lab.tokenizer import CharacterTokenizer


def test_contiguous_train_validation_split(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("abcdefghij", encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text("abcdefghij")

    dataset = TextDataset.from_file(corpus_path, tokenizer, train_fraction=0.6)

    assert tokenizer.decode(dataset.train_tokens.tolist()) == "abcdef"
    assert tokenizer.decode(dataset.validation_tokens.tolist()) == "ghij"


def test_batch_shapes_and_shifted_targets(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text(corpus_path.read_text(encoding="utf-8"))
    dataset = TextDataset.from_file(corpus_path, tokenizer, train_fraction=0.8)

    inputs, targets = dataset.get_batch(
        "train",
        batch_size=4,
        context_length=5,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(7),
    )

    assert inputs.shape == (4, 5)
    assert targets.shape == (4, 5)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_batch_rejects_context_too_long(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("abcdefghij", encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text("abcdefghij")
    dataset = TextDataset.from_file(corpus_path, tokenizer, train_fraction=0.5)

    with pytest.raises(ValueError, match="context length"):
        dataset.get_batch("validation", 1, 5, torch.device("cpu"))


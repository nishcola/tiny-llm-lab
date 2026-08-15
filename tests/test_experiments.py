from pathlib import Path

import pytest

from tiny_llm_lab.experiments import (
    CONTROLLED_EXPERIMENT_SEEDS,
    controlled_experiment_conditions,
    prepare_corpus,
    summarize_results,
)


def test_controlled_experiment_conditions_define_the_seven_runs() -> None:
    conditions = controlled_experiment_conditions()

    assert CONTROLLED_EXPERIMENT_SEEDS == (1337, 2027)
    assert [(item.study, item.name) for item in conditions] == [
        ("tokenization", "character"),
        ("tokenization", "byte-bpe-320"),
        ("position", "learned"),
        ("position", "sinusoidal"),
        ("attention-heads", "2-heads"),
        ("attention-heads", "4-heads"),
        ("attention-heads", "8-heads"),
    ]
    assert {item.config.model.num_heads for item in conditions if item.study == "attention-heads"} == {2, 4, 8}
    assert {item.config.model.embedding_dim for item in conditions} == {192}


def test_prepare_corpus_splits_raw_text_before_fitting_the_tokenizer(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcd" * 100, encoding="utf-8")

    prepared = prepare_corpus(corpus, tokenizer_kind="character", train_fraction=0.75)

    assert prepared.raw_train_text == "abcd" * 75
    assert prepared.raw_validation_text == "abcd" * 25
    assert prepared.tokenizer.vocabulary_size == 4
    assert len(prepared.dataset.train_tokens) == 300
    assert len(prepared.dataset.validation_tokens) == 100


def test_prepare_corpus_rejects_characters_missing_from_the_train_partition(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a" * 9 + "z", encoding="utf-8")

    with pytest.raises(ValueError, match="validation characters"):
        prepare_corpus(corpus, tokenizer_kind="character", train_fraction=0.9)


def test_summarize_results_aggregates_two_seeds_without_claiming_significance() -> None:
    summary = summarize_results(
        [
            {"study": "position", "condition": "learned", "seed": 1337, "validation_loss": 1.5, "bits_per_byte": 0.5, "parameter_count": 10, "training_seconds": 4.0},
            {"study": "position", "condition": "learned", "seed": 2027, "validation_loss": 1.7, "bits_per_byte": 0.7, "parameter_count": 10, "training_seconds": 5.0},
        ]
    )

    assert summary == [
        {
            "study": "position",
            "condition": "learned",
            "runs": 2,
            "validation_loss_mean": 1.6,
            "validation_loss_range": [1.5, 1.7],
            "bits_per_byte_mean": 0.6,
            "bits_per_byte_range": [0.5, 0.7],
            "parameter_count": 10,
            "training_seconds_mean": 4.5,
        }
    ]

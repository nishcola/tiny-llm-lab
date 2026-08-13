import pytest

from tiny_llm_lab.tokenizer import CharacterTokenizer


def test_round_trip_preserves_text_and_sorted_vocabulary() -> None:
    tokenizer = CharacterTokenizer.from_text("cab\n")

    assert tokenizer.vocabulary == ("\n", "a", "b", "c")
    assert tokenizer.decode(tokenizer.encode("cab\n")) == "cab\n"


def test_serialized_tokenizer_restores_same_encoding() -> None:
    tokenizer = CharacterTokenizer.from_text("hello")

    restored = CharacterTokenizer.from_state_dict(tokenizer.state_dict())

    assert restored.vocabulary == tokenizer.vocabulary
    assert restored.encode("hello") == tokenizer.encode("hello")


def test_unknown_characters_and_ids_are_rejected() -> None:
    tokenizer = CharacterTokenizer.from_text("abc")

    with pytest.raises(ValueError, match="Unknown character"):
        tokenizer.encode("d")
    with pytest.raises(ValueError, match="Unknown token id"):
        tokenizer.decode([3])


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        CharacterTokenizer.from_text("")


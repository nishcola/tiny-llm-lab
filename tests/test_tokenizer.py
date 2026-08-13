import pytest

from tiny_llm_lab.tokenizer import BytePairTokenizer, CharacterTokenizer


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


def test_byte_pair_tokenizer_round_trips_unicode_and_unusual_input() -> None:
    text = "Caf\u00e9 \U0001f44b\nrare: \u0378"

    tokenizer = BytePairTokenizer.train(text, vocabulary_size=260)

    assert tokenizer.decode(tokenizer.encode(text)) == text
    with pytest.raises(ValueError, match="Unknown token id"):
        tokenizer.decode([tokenizer.vocabulary_size])


def test_byte_pair_training_and_encoding_are_deterministic() -> None:
    text = "banana bandana banana bandana"

    first = BytePairTokenizer.train(text, vocabulary_size=260)
    second = BytePairTokenizer.train(text, vocabulary_size=260)

    assert first.state_dict() == second.state_dict()
    assert first.encode("banana bandana") == second.encode("banana bandana")


def test_byte_pair_tokenizer_respects_requested_vocabulary_size() -> None:
    tokenizer = BytePairTokenizer.train("abcabcabcabcabcabc", vocabulary_size=259)

    assert tokenizer.vocabulary_size == 259
    with pytest.raises(ValueError, match="at least 256"):
        BytePairTokenizer.train("abc", vocabulary_size=255)


def test_byte_pair_tokenizer_json_round_trip_preserves_encoding(tmp_path) -> None:
    tokenizer = BytePairTokenizer.train("hello hello hello", vocabulary_size=258)
    output_path = tmp_path / "tokenizer.json"

    tokenizer.save(output_path)
    restored = BytePairTokenizer.load(output_path)

    assert restored.state_dict() == tokenizer.state_dict()
    assert restored.encode("hello hello") == tokenizer.encode("hello hello")

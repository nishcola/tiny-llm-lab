"""A deterministic character-level tokenizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class CharacterTokenizer:
    vocabulary: tuple[str, ...]
    _character_to_id: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.vocabulary:
            raise ValueError("Tokenizer vocabulary cannot be empty")
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ValueError("Tokenizer vocabulary contains duplicate characters")
        if any(len(character) != 1 for character in self.vocabulary):
            raise ValueError("Tokenizer vocabulary entries must be single characters")
        object.__setattr__(
            self,
            "_character_to_id",
            {character: token_id for token_id, character in enumerate(self.vocabulary)},
        )

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        if not text:
            raise ValueError("Cannot build a tokenizer from an empty corpus")
        return cls(tuple(sorted(set(text))))

    @property
    def vocabulary_size(self) -> int:
        return len(self.vocabulary)

    def encode(self, text: str) -> list[int]:
        try:
            return [self._character_to_id[character] for character in text]
        except KeyError as error:
            raise ValueError(f"Unknown character: {error.args[0]!r}") from error

    def decode(self, token_ids: Iterable[int]) -> str:
        characters: list[str] = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= self.vocabulary_size:
                raise ValueError(f"Unknown token id: {token_id}")
            characters.append(self.vocabulary[token_id])
        return "".join(characters)

    def state_dict(self) -> dict[str, Sequence[str]]:
        return {"vocabulary": list(self.vocabulary)}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Sequence[str]]) -> "CharacterTokenizer":
        return cls(tuple(state["vocabulary"]))


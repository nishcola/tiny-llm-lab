"""Small, inspectable tokenizers used by the language-model pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Protocol, Sequence


class Tokenizer(Protocol):
    """The tokenization operations required by data and training code."""

    @property
    def vocabulary_size(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Iterable[int]) -> str: ...

    def state_dict(self) -> Mapping[str, object]: ...


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


@dataclass(frozen=True)
class BytePairTokenizer:
    """A byte-level BPE tokenizer with explicitly stored merge rules.

    Base tokens represent all possible UTF-8 bytes, so every Python string is
    encodable. Each merge records the two token IDs it joins; their order is
    the merge rank used during encoding.
    """

    vocabulary: tuple[bytes, ...]
    merges: tuple[tuple[int, int], ...]
    target_vocabulary_size: int

    BASE_VOCABULARY_SIZE = 256
    STATE_VERSION = 1

    def __post_init__(self) -> None:
        if self.target_vocabulary_size < self.BASE_VOCABULARY_SIZE:
            raise ValueError("vocabulary_size must be at least 256 for byte-level BPE")
        if len(self.vocabulary) < self.BASE_VOCABULARY_SIZE:
            raise ValueError("Byte-level BPE vocabulary must include all 256 byte tokens")
        if len(self.vocabulary) > self.target_vocabulary_size:
            raise ValueError("Vocabulary cannot exceed its configured target size")
        expected_base = tuple(bytes([token_id]) for token_id in range(self.BASE_VOCABULARY_SIZE))
        if self.vocabulary[: self.BASE_VOCABULARY_SIZE] != expected_base:
            raise ValueError("Byte-level BPE vocabulary must begin with byte tokens in order")
        if len(self.merges) != len(self.vocabulary) - self.BASE_VOCABULARY_SIZE:
            raise ValueError("Each non-byte vocabulary token must have one merge rule")
        for merge_index, (left_id, right_id) in enumerate(self.merges):
            merged_id = self.BASE_VOCABULARY_SIZE + merge_index
            if not 0 <= left_id < merged_id or not 0 <= right_id < merged_id:
                raise ValueError("Merge rules must reference earlier token IDs")
            if self.vocabulary[merged_id] != self.vocabulary[left_id] + self.vocabulary[right_id]:
                raise ValueError("Vocabulary token does not match its merge rule")

    @classmethod
    def train(cls, text: str, vocabulary_size: int) -> "BytePairTokenizer":
        if not text:
            raise ValueError("Cannot train a tokenizer from an empty corpus")
        if vocabulary_size < cls.BASE_VOCABULARY_SIZE:
            raise ValueError("vocabulary_size must be at least 256 for byte-level BPE")

        token_sequences = [list(piece.encode("utf-8")) for piece in cls._pretokenize(text)]
        vocabulary = [bytes([token_id]) for token_id in range(cls.BASE_VOCABULARY_SIZE)]
        merges: list[tuple[int, int]] = []
        while len(vocabulary) < vocabulary_size:
            pair_counts: dict[tuple[int, int], int] = {}
            for token_ids in token_sequences:
                for pair in zip(token_ids, token_ids[1:]):
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
            if not pair_counts:
                break
            selected_pair = min(pair_counts, key=lambda pair: (-pair_counts[pair], pair))
            new_token_id = len(vocabulary)
            vocabulary.append(vocabulary[selected_pair[0]] + vocabulary[selected_pair[1]])
            merges.append(selected_pair)
            token_sequences = [
                cls._replace_pair(token_ids, selected_pair, new_token_id)
                for token_ids in token_sequences
            ]
        return cls(tuple(vocabulary), tuple(merges), vocabulary_size)

    @staticmethod
    def _pretokenize(text: str) -> list[str]:
        return re.findall(r"\s+|\S+", text)

    @staticmethod
    def _replace_pair(
        token_ids: list[int], pair: tuple[int, int], replacement: int
    ) -> list[int]:
        result: list[int] = []
        index = 0
        while index < len(token_ids):
            if index + 1 < len(token_ids) and (token_ids[index], token_ids[index + 1]) == pair:
                result.append(replacement)
                index += 2
            else:
                result.append(token_ids[index])
                index += 1
        return result

    @property
    def vocabulary_size(self) -> int:
        return len(self.vocabulary)

    def encode(self, text: str) -> list[int]:
        encoded: list[int] = []
        for piece in self._pretokenize(text):
            token_ids = list(piece.encode("utf-8"))
            for merge_index, pair in enumerate(self.merges):
                token_ids = self._replace_pair(
                    token_ids, pair, self.BASE_VOCABULARY_SIZE + merge_index
                )
            encoded.extend(token_ids)
        return encoded

    def decode(self, token_ids: Iterable[int]) -> str:
        pieces: list[bytes] = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= self.vocabulary_size:
                raise ValueError(f"Unknown token id: {token_id}")
            pieces.append(self.vocabulary[token_id])
        try:
            return b"".join(pieces).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Token IDs do not form valid UTF-8 text") from error

    def state_dict(self) -> dict[str, object]:
        return {
            "version": self.STATE_VERSION,
            "target_vocabulary_size": self.target_vocabulary_size,
            "vocabulary": [piece.hex() for piece in self.vocabulary],
            "merges": [list(pair) for pair in self.merges],
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "BytePairTokenizer":
        if state.get("version") != cls.STATE_VERSION:
            raise ValueError(f"Unsupported BytePairTokenizer version: {state.get('version')!r}")
        vocabulary_values = state.get("vocabulary")
        merge_values = state.get("merges")
        target_size = state.get("target_vocabulary_size")
        if not isinstance(vocabulary_values, list) or not all(
            isinstance(piece, str) for piece in vocabulary_values
        ):
            raise ValueError("BytePairTokenizer vocabulary must be a list of hexadecimal strings")
        if not isinstance(merge_values, list) or not isinstance(target_size, int):
            raise ValueError("BytePairTokenizer state is missing merges or target vocabulary size")
        try:
            vocabulary = tuple(bytes.fromhex(piece) for piece in vocabulary_values)
            merges = tuple(
                (int(pair[0]), int(pair[1]))
                for pair in merge_values
                if isinstance(pair, list) and len(pair) == 2
            )
        except (TypeError, ValueError) as error:
            raise ValueError("BytePairTokenizer state contains invalid merge data") from error
        if len(merges) != len(merge_values):
            raise ValueError("BytePairTokenizer merges must be pairs of token IDs")
        return cls(vocabulary, merges, target_size)

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.state_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "BytePairTokenizer":
        with Path(path).open(encoding="utf-8") as tokenizer_file:
            return cls.from_state_dict(json.load(tokenizer_file))

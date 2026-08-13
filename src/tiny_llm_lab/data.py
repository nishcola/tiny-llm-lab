"""Text corpus loading, metadata, splitting, and random batching."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from torch import Tensor

from tiny_llm_lab.tokenizer import CharacterTokenizer


@dataclass(frozen=True)
class DatasetMetadata:
    source: str
    byte_count: int
    sha256: str

    def to_dict(self) -> dict[str, str | int]:
        return {
            "source": self.source,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class TextDataset:
    train_tokens: Tensor
    validation_tokens: Tensor
    metadata: DatasetMetadata

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        tokenizer: CharacterTokenizer,
        train_fraction: float = 0.9,
        source: str | None = None,
    ) -> "TextDataset":
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        corpus_path = Path(path)
        raw_bytes = corpus_path.read_bytes()
        text = raw_bytes.decode("utf-8")
        encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        split_index = int(len(encoded) * train_fraction)
        if split_index < 1 or split_index >= len(encoded):
            raise ValueError("Corpus is too short for the requested train/validation split")
        return cls(
            train_tokens=encoded[:split_index],
            validation_tokens=encoded[split_index:],
            metadata=DatasetMetadata(
                source=source or str(corpus_path),
                byte_count=len(raw_bytes),
                sha256=hashlib.sha256(raw_bytes).hexdigest(),
            ),
        )

    def get_batch(
        self,
        split: str,
        batch_size: int,
        context_length: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        if split == "train":
            tokens = self.train_tokens
        elif split in {"validation", "val"}:
            tokens = self.validation_tokens
        else:
            raise ValueError("split must be 'train' or 'validation'")
        if batch_size <= 0 or context_length <= 0:
            raise ValueError("batch size and context length must be positive")
        possible_starts = len(tokens) - context_length
        if possible_starts <= 0:
            raise ValueError("context length must be smaller than the selected split")
        starts = torch.randint(possible_starts, (batch_size,), generator=generator)
        inputs = torch.stack([tokens[start : start + context_length] for start in starts.tolist()])
        targets = torch.stack([tokens[start + 1 : start + context_length + 1] for start in starts.tolist()])
        return inputs.to(device), targets.to(device)


"""Versioned, resumable PyTorch checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Mapping

import torch
from torch import nn

from tiny_llm_lab.config import ExperimentConfig
from tiny_llm_lab.data import DatasetMetadata
from tiny_llm_lab.model import DecoderOnlyTransformer
from tiny_llm_lab.tokenizer import BytePairTokenizer, CharacterTokenizer, Tokenizer


CHECKPOINT_VERSION = 2


@dataclass(frozen=True)
class LoadedCheckpoint:
    config: ExperimentConfig
    tokenizer: Tokenizer
    step: int
    validation_loss: float
    dataset_metadata: DatasetMetadata


@dataclass(frozen=True)
class InferenceCheckpoint:
    """A checkpoint reconstructed for read-only model inspection."""

    model: DecoderOnlyTransformer
    tokenizer: Tokenizer
    config: ExperimentConfig
    step: int
    validation_loss: float
    dataset_metadata: DatasetMetadata


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    tokenizer: Tokenizer,
    config: ExperimentConfig,
    step: int,
    validation_loss: float,
    dataset_metadata: DatasetMetadata,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "config": config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "tokenizer": _tokenizer_payload(tokenizer),
        "step": step,
        "validation_loss": validation_loss,
        "dataset_metadata": dataset_metadata.to_dict(),
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("version") not in {1, CHECKPOINT_VERSION}:
        raise ValueError(f"Unsupported checkpoint version: {payload.get('version')!r}")
    config = ExperimentConfig.from_dict(payload["config"])
    model_config = getattr(model, "config", None)
    if model_config is not None and model_config != config.model:
        raise ValueError("Checkpoint model configuration does not match the supplied model")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    random.setstate(payload["rng_state"]["python"])
    torch.set_rng_state(payload["rng_state"]["torch"].cpu())
    cuda_state = payload["rng_state"].get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])
    metadata_values = payload["dataset_metadata"]
    return LoadedCheckpoint(
        config=config,
        tokenizer=_load_tokenizer(payload["tokenizer"]),
        step=int(payload["step"]),
        validation_loss=float(payload["validation_loss"]),
        dataset_metadata=DatasetMetadata(**metadata_values),
    )


def load_inference_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> InferenceCheckpoint:
    """Recreate an evaluation-mode model and tokenizer from a checkpoint."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("version") not in {1, CHECKPOINT_VERSION}:
        raise ValueError(f"Unsupported checkpoint version: {payload.get('version')!r}")
    config = ExperimentConfig.from_dict(payload["config"])
    model = DecoderOnlyTransformer(config.model).to(map_location)
    model.load_state_dict(payload["model_state"])
    model.eval()
    metadata_values = payload["dataset_metadata"]
    return InferenceCheckpoint(
        model=model,
        tokenizer=_load_tokenizer(payload["tokenizer"]),
        config=config,
        step=int(payload["step"]),
        validation_loss=float(payload["validation_loss"]),
        dataset_metadata=DatasetMetadata(**metadata_values),
    )


def _tokenizer_payload(tokenizer: Tokenizer) -> dict[str, object]:
    if isinstance(tokenizer, CharacterTokenizer):
        tokenizer_type = "character"
    elif isinstance(tokenizer, BytePairTokenizer):
        tokenizer_type = "byte_pair"
    else:
        raise TypeError(f"Unsupported tokenizer type: {type(tokenizer).__name__}")
    return {"type": tokenizer_type, "state": dict(tokenizer.state_dict())}


def _load_tokenizer(payload: object) -> Tokenizer:
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint tokenizer payload must be a mapping")
    if "type" not in payload:
        return CharacterTokenizer.from_state_dict(payload)
    tokenizer_type = payload.get("type")
    state = payload.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint tokenizer state must be a mapping")
    if tokenizer_type == "character":
        return CharacterTokenizer.from_state_dict(state)
    if tokenizer_type == "byte_pair":
        return BytePairTokenizer.from_state_dict(state)
    raise ValueError(f"Unsupported checkpoint tokenizer type: {tokenizer_type!r}")

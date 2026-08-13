"""Versioned, resumable PyTorch checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import torch
from torch import nn

from tiny_llm_lab.config import ExperimentConfig
from tiny_llm_lab.data import DatasetMetadata
from tiny_llm_lab.tokenizer import CharacterTokenizer


CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class LoadedCheckpoint:
    config: ExperimentConfig
    tokenizer: CharacterTokenizer
    step: int
    validation_loss: float
    dataset_metadata: DatasetMetadata


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    tokenizer: CharacterTokenizer,
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
        "tokenizer": tokenizer.state_dict(),
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
    if payload.get("version") != CHECKPOINT_VERSION:
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
        torch.cuda.set_rng_state_all(cuda_state)
    metadata_values = payload["dataset_metadata"]
    return LoadedCheckpoint(
        config=config,
        tokenizer=CharacterTokenizer.from_state_dict(payload["tokenizer"]),
        step=int(payload["step"]),
        validation_loss=float(payload["validation_loss"]),
        dataset_metadata=DatasetMetadata(**metadata_values),
    )


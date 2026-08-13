"""Validated experiment configuration loaded from TOML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
import tomllib


@dataclass(frozen=True)
class ModelConfig:
    context_length: int
    embedding_dim: int
    num_layers: int
    num_heads: int
    mlp_dim: int
    dropout: float = 0.0
    vocabulary_size: int | None = None

    def __post_init__(self) -> None:
        for name in ("context_length", "embedding_dim", "num_layers", "num_heads", "mlp_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.vocabulary_size is not None and self.vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive when provided")


@dataclass(frozen=True)
class DataConfig:
    path: Path
    train_fraction: float = 0.9
    source_url: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "auto"
    seed: int = 1337
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_steps: int = 1000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    checkpoint_interval: int = 500
    output_dir: Path = Path("checkpoints")

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        for name in (
            "batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "eval_interval",
            "eval_batches",
            "checkpoint_interval",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer settings must be non-negative, with positive learning rate and gradient norm")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["data"]["path"] = str(self.data.path)
        result["training"]["output_dir"] = str(self.training.output_dir)
        return result

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ExperimentConfig":
        model_values = dict(values["model"])
        data_values = dict(values["data"])
        training_values = dict(values.get("training", {}))
        data_values["path"] = Path(data_values["path"])
        if "output_dir" in training_values:
            training_values["output_dir"] = Path(training_values["output_dir"])
        return cls(
            model=ModelConfig(**model_values),
            data=DataConfig(**data_values),
            training=TrainingConfig(**training_values),
        )


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("rb") as config_file:
        return ExperimentConfig.from_dict(tomllib.load(config_file))


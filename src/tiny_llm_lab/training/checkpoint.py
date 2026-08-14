"""Versioned, resumable PyTorch checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
from typing import Any, Mapping

import torch
from torch import nn

from tiny_llm_lab.config import ExperimentConfig
from tiny_llm_lab.data import DatasetMetadata
from tiny_llm_lab.model import DecoderOnlyTransformer
from tiny_llm_lab.tokenizer import BytePairTokenizer, CharacterTokenizer, Tokenizer


CHECKPOINT_VERSION = 2
TIMELINE_VERSION = 1


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
    validation_loss: float | None
    dataset_metadata: DatasetMetadata


@dataclass(frozen=True)
class TimelineCheckpoint:
    step: int
    validation_loss: float | None
    path: Path
    byte_count: int
    sha256: str
    model_fingerprint: str
    tokenizer_fingerprint: str
    available: bool = True
    error: str | None = None


@dataclass(frozen=True)
class TimelineMetric:
    step: int
    training_loss: float | None
    validation_loss: float | None


@dataclass(frozen=True)
class TimelineRun:
    path: Path
    run_id: str
    checkpoints: tuple[TimelineCheckpoint, ...]
    metrics: tuple[TimelineMetric, ...]
    error: str | None = None


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


def create_timeline_run(
    output_dir: str | Path,
    *,
    config: ExperimentConfig,
    tokenizer: Tokenizer,
    dataset_metadata: DatasetMetadata,
    run_name: str | None = None,
) -> TimelineRun:
    """Create a self-describing run directory without loading any model weights."""
    root = Path(output_dir) / "runs"
    selected_name = run_name or config.training.run_name or _timestamped_run_name()
    if Path(selected_name).name != selected_name:
        raise ValueError("run_name must be a single directory name")
    run_path = root / selected_name
    if run_path.exists() and (run_name is not None or config.training.run_name is not None):
        raise FileExistsError(f"Training run already exists: {run_path}")
    suffix = 2
    while run_path.exists():
        run_path = root / f"{selected_name}-{suffix}"
        suffix += 1
    selected_name = run_path.name
    (run_path / "resume").mkdir(parents=True)
    (run_path / "timeline").mkdir()
    tokenizer_payload = _tokenizer_payload(tokenizer)
    manifest = {
        "version": TIMELINE_VERSION,
        "run_id": selected_name,
        "config": config.to_dict(),
        "dataset_metadata": dataset_metadata.to_dict(),
        "tokenizer": tokenizer_payload,
        "model_fingerprint": _fingerprint(config.to_dict()["model"]),
        "tokenizer_fingerprint": _fingerprint(tokenizer_payload),
        "checkpoint_policy": {
            "interval": config.training.checkpoint_interval,
            "max_timeline_checkpoints": config.training.max_timeline_checkpoints,
        },
    }
    _write_json_atomic(run_path / "run.json", manifest)
    _write_json_atomic(run_path / "timeline" / "index.json", {"version": TIMELINE_VERSION, "checkpoints": []})
    (run_path / "metrics.jsonl").touch()
    return TimelineRun(path=run_path, run_id=selected_name, checkpoints=(), metrics=())


def discover_timeline_run(path: str | Path) -> TimelineRun:
    """Read run metadata and checkpoint availability without deserializing weights."""
    run_path = Path(path)
    try:
        manifest = _read_json(run_path / "run.json")
        if manifest.get("version") != TIMELINE_VERSION:
            raise ValueError(f"Unsupported timeline run version: {manifest.get('version')!r}")
        run_id = _required_string(manifest, "run_id")
        index = _read_json(run_path / "timeline" / "index.json")
        if index.get("version") != TIMELINE_VERSION or not isinstance(index.get("checkpoints"), list):
            raise ValueError("Timeline index is malformed")
        checkpoints = tuple(
            _checkpoint_from_index(run_path, record, manifest) for record in index["checkpoints"]
        )
        metrics = _read_metrics(run_path / "metrics.jsonl")
        return TimelineRun(run_path, run_id, tuple(sorted(checkpoints, key=lambda item: item.step)), metrics)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return TimelineRun(run_path, run_path.name, (), (), str(error))


def append_timeline_metric(
    run: TimelineRun,
    *,
    step: int,
    training_loss: float | None,
    validation_loss: float | None,
) -> None:
    if step <= 0:
        raise ValueError("Metric step must be positive")
    record = {
        "step": step,
        "training_loss": _finite_or_none(training_loss),
        "validation_loss": _finite_or_none(validation_loss),
    }
    with (run.path / "metrics.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def save_timeline_checkpoint(
    run: TimelineRun,
    *,
    model: nn.Module,
    tokenizer: Tokenizer,
    config: ExperimentConfig,
    step: int,
    validation_loss: float | None,
    dataset_metadata: DatasetMetadata,
) -> TimelineCheckpoint:
    """Save an inference-only checkpoint and update the run index after verification."""
    if step <= 0:
        raise ValueError("Checkpoint step must be positive")
    manifest = _read_json(run.path / "run.json")
    config_dict = config.to_dict()
    tokenizer_payload = _tokenizer_payload(tokenizer)
    model_fingerprint = _fingerprint(config_dict["model"])
    tokenizer_fingerprint = _fingerprint(tokenizer_payload)
    if manifest.get("model_fingerprint") != model_fingerprint:
        raise ValueError("Timeline model configuration is incompatible with the run")
    if manifest.get("tokenizer_fingerprint") != tokenizer_fingerprint:
        raise ValueError("Timeline tokenizer is incompatible with the run")
    relative_path = Path("timeline") / f"step_{step:06d}.pt"
    checkpoint_path = run.path / relative_path
    payload: dict[str, Any] = {
        "version": TIMELINE_VERSION,
        "kind": "timeline_inference",
        "run_id": run.run_id,
        "config": config_dict,
        "model_state": model.state_dict(),
        "tokenizer": tokenizer_payload,
        "step": step,
        "validation_loss": _finite_or_none(validation_loss),
        "dataset_metadata": dataset_metadata.to_dict(),
        "model_fingerprint": model_fingerprint,
        "tokenizer_fingerprint": tokenizer_fingerprint,
    }
    temporary_path = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(checkpoint_path)
    record = TimelineCheckpoint(
        step=step,
        validation_loss=_finite_or_none(validation_loss),
        path=relative_path,
        byte_count=checkpoint_path.stat().st_size,
        sha256=_file_sha256(checkpoint_path),
        model_fingerprint=model_fingerprint,
        tokenizer_fingerprint=tokenizer_fingerprint,
    )
    _update_timeline_index(run.path, record, config.training.max_timeline_checkpoints)
    return record


def load_timeline_checkpoint(
    run: TimelineRun,
    checkpoint: TimelineCheckpoint,
    *,
    map_location: str | torch.device = "cpu",
) -> InferenceCheckpoint:
    """Load one verified timeline checkpoint for read-only inspection."""
    if not checkpoint.available:
        raise ValueError(checkpoint.error or "Checkpoint is unavailable")
    checkpoint_path = _resolve_run_path(run.path, checkpoint.path)
    if checkpoint_path.stat().st_size != checkpoint.byte_count:
        raise ValueError("Checkpoint size does not match the timeline index")
    if _file_sha256(checkpoint_path) != checkpoint.sha256:
        raise ValueError("Checkpoint checksum does not match the timeline index")
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError(f"Could not read timeline checkpoint: {error}") from error
    if not isinstance(payload, Mapping) or payload.get("version") != TIMELINE_VERSION:
        raise ValueError("Unsupported timeline checkpoint version")
    if payload.get("kind") != "timeline_inference" or payload.get("run_id") != run.run_id:
        raise ValueError("Checkpoint does not belong to this timeline run")
    if payload.get("model_fingerprint") != checkpoint.model_fingerprint:
        raise ValueError("Checkpoint model fingerprint does not match the index")
    if payload.get("tokenizer_fingerprint") != checkpoint.tokenizer_fingerprint:
        raise ValueError("Checkpoint tokenizer fingerprint does not match the index")
    try:
        config = ExperimentConfig.from_dict(payload["config"])
        if _fingerprint(config.to_dict()["model"]) != checkpoint.model_fingerprint:
            raise ValueError("Checkpoint model configuration does not match the timeline metadata")
        tokenizer_payload = payload["tokenizer"]
        if _fingerprint(tokenizer_payload) != checkpoint.tokenizer_fingerprint:
            raise ValueError("Checkpoint tokenizer does not match the timeline metadata")
        model = DecoderOnlyTransformer(config.model).to(map_location)
        model.load_state_dict(payload["model_state"])
        model.eval()
        metadata = DatasetMetadata(**payload["dataset_metadata"])
        return InferenceCheckpoint(
            model=model,
            tokenizer=_load_tokenizer(tokenizer_payload),
            config=config,
            step=int(payload["step"]),
            validation_loss=_finite_or_none(payload.get("validation_loss")),
            dataset_metadata=metadata,
        )
    except (KeyError, TypeError, RuntimeError, ValueError) as error:
        raise ValueError(f"Malformed timeline checkpoint: {error}") from error


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


def _timestamped_run_name() -> str:
    return datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ")


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _write_json_atomic(path: Path, value: object) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _required_string(values: Mapping[str, Any], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing or invalid {name}")
    return value


def _resolve_run_path(run_path: Path, relative_path: Path) -> Path:
    resolved = (run_path / relative_path).resolve()
    if run_path.resolve() not in resolved.parents:
        raise ValueError("Checkpoint path escapes the run directory")
    return resolved


def _checkpoint_from_index(
    run_path: Path, values: object, manifest: Mapping[str, Any]
) -> TimelineCheckpoint:
    if not isinstance(values, Mapping):
        raise ValueError("Timeline checkpoint record must be an object")
    relative_path = Path(_required_string(values, "path"))
    checkpoint_path = _resolve_run_path(run_path, relative_path)
    record = TimelineCheckpoint(
        step=int(values["step"]),
        validation_loss=_finite_or_none(values.get("validation_loss")),
        path=relative_path,
        byte_count=int(values["byte_count"]),
        sha256=_required_string(values, "sha256"),
        model_fingerprint=_required_string(values, "model_fingerprint"),
        tokenizer_fingerprint=_required_string(values, "tokenizer_fingerprint"),
    )
    error = None
    if record.step <= 0 or record.byte_count <= 0:
        error = "Checkpoint index contains an invalid step or byte count"
    elif record.model_fingerprint != manifest.get("model_fingerprint"):
        error = "Checkpoint model configuration is incompatible with the run"
    elif record.tokenizer_fingerprint != manifest.get("tokenizer_fingerprint"):
        error = "Checkpoint tokenizer is incompatible with the run"
    elif not checkpoint_path.is_file():
        error = "Checkpoint file is missing"
    elif checkpoint_path.stat().st_size != record.byte_count:
        error = "Checkpoint size does not match the timeline index"
    elif _file_sha256(checkpoint_path) != record.sha256:
        error = "Checkpoint checksum does not match the timeline index"
    return TimelineCheckpoint(**{**record.__dict__, "available": error is None, "error": error})


def _read_metrics(path: Path) -> tuple[TimelineMetric, ...]:
    if not path.exists():
        return ()
    metrics: list[TimelineMetric] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError("Metric must be an object")
            metrics.append(
                TimelineMetric(
                    step=int(value["step"]),
                    training_loss=_finite_or_none(value.get("training_loss")),
                    validation_loss=_finite_or_none(value.get("validation_loss")),
                )
            )
        except (TypeError, ValueError, json.JSONDecodeError, KeyError):
            continue
    return tuple(sorted(metrics, key=lambda metric: metric.step))


def _update_timeline_index(run_path: Path, new_record: TimelineCheckpoint, maximum: int) -> None:
    index_path = run_path / "timeline" / "index.json"
    index = _read_json(index_path)
    existing = [_checkpoint_from_index(run_path, record, _read_json(run_path / "run.json")) for record in index["checkpoints"]]
    retained = _retain_evenly((*existing, new_record), maximum)
    retained_paths = {record.path for record in retained}
    for record in existing:
        if record.path not in retained_paths:
            _resolve_run_path(run_path, record.path).unlink(missing_ok=True)
    _write_json_atomic(
        index_path,
        {
            "version": TIMELINE_VERSION,
            "checkpoints": [
                {
                    "step": record.step,
                    "validation_loss": record.validation_loss,
                    "path": record.path.as_posix(),
                    "byte_count": record.byte_count,
                    "sha256": record.sha256,
                    "model_fingerprint": record.model_fingerprint,
                    "tokenizer_fingerprint": record.tokenizer_fingerprint,
                }
                for record in retained
            ],
        },
    )


def _retain_evenly(
    checkpoints: tuple[TimelineCheckpoint, ...], maximum: int
) -> tuple[TimelineCheckpoint, ...]:
    ordered = tuple(sorted(checkpoints, key=lambda record: record.step))
    if maximum < 2:
        raise ValueError("max_timeline_checkpoints must be at least 2")
    if len(ordered) <= maximum:
        return ordered
    indexes = {round(index * (len(ordered) - 1) / (maximum - 1)) for index in range(maximum)}
    return tuple(ordered[index] for index in sorted(indexes))

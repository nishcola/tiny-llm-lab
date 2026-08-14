"""Framework-neutral analysis helpers for learned input token embeddings."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable

import torch
from torch import Tensor, nn


CACHE_VERSION = 1
PCA_METHOD = "pca"
MAX_PCA_FIT_ROWS = 20_000
MAX_PLOT_POINTS = 10_000
SEARCH_MATCH_LIMIT = 100
_CACHE_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class EmbeddingNeighbor:
    """One token ranked by cosine similarity to a selected token."""

    token_id: int
    cosine_similarity: float


@dataclass(frozen=True)
class EmbeddingProjection:
    """Serializable labels and two-dimensional PCA coordinates."""

    checkpoint_digest: str
    token_labels: tuple[str, ...]
    coordinates: tuple[tuple[float, float], ...]

    def coordinate_tensor(self) -> Tensor:
        return torch.tensor(self.coordinates, dtype=torch.float64)


def extract_token_embeddings(model: nn.Module) -> Tensor:
    """Copy a model's learned input embeddings to detached CPU memory."""
    try:
        embeddings = model.token_embeddings.weight
    except AttributeError as error:
        raise ValueError("Model does not expose token_embeddings.weight") from error
    if embeddings.ndim != 2:
        raise ValueError("Token embeddings must have shape (vocabulary, embedding_dim)")
    return embeddings.detach().to(device="cpu").clone()


def pca_2d(embeddings: Tensor) -> Tensor:
    """Return deterministic, centered two-dimensional PCA coordinates on CPU."""
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must have shape (vocabulary, embedding_dim)")
    vocabulary_size, embedding_dim = embeddings.shape
    if vocabulary_size == 0 or embedding_dim == 0:
        raise ValueError("Embeddings must have non-empty vocabulary and embedding dimensions")

    all_embeddings = embeddings.detach().to(device="cpu", dtype=torch.float64)
    fit_embeddings = all_embeddings[_evenly_spaced_indices(vocabulary_size, MAX_PCA_FIT_ROWS)]
    mean = fit_embeddings.mean(dim=0, keepdim=True)
    _, _, right_vectors = torch.linalg.svd(fit_embeddings - mean, full_matrices=False)
    component_count = min(2, right_vectors.shape[0])
    components = right_vectors[:component_count].clone()
    for index in range(component_count):
        pivot = int(components[index].abs().argmax())
        if components[index, pivot] < 0:
            components[index].neg_()
    coordinates = (all_embeddings - mean) @ components.T
    if component_count == 2:
        return coordinates
    return torch.cat((coordinates, torch.zeros(vocabulary_size, 2 - component_count, dtype=torch.float64)), dim=1)


def nearest_neighbors(embeddings: Tensor, token_id: int, *, limit: int = 10) -> tuple[EmbeddingNeighbor, ...]:
    """Rank other tokens by cosine similarity in the original embedding space."""
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must have shape (vocabulary, embedding_dim)")
    vocabulary_size = embeddings.shape[0]
    if not 0 <= token_id < vocabulary_size:
        raise ValueError("token_id is outside the embedding vocabulary")
    if limit <= 0:
        raise ValueError("limit must be positive")

    values = embeddings.detach().to(device="cpu", dtype=torch.float64)
    query = values[token_id]
    query_norm = torch.linalg.vector_norm(query)
    norms = torch.linalg.vector_norm(values, dim=1)
    if query_norm == 0:
        similarities = torch.zeros(vocabulary_size, dtype=torch.float64, device=values.device)
    else:
        denominators = norms * query_norm
        similarities = torch.where(
            denominators > 0,
            (values @ query) / denominators,
            torch.zeros_like(denominators),
        )
    similarities[token_id] = -torch.inf
    ordered = torch.argsort(similarities, descending=True, stable=True)
    selected = ordered[: min(limit, max(0, vocabulary_size - 1))].to(device="cpu")
    return tuple(
        EmbeddingNeighbor(int(candidate), float(similarities[candidate])) for candidate in selected
    )


def search_token_labels(
    token_labels: Iterable[str], query: str, *, limit: int = SEARCH_MATCH_LIMIT
) -> tuple[int, ...]:
    """Find exact label matches first, then substring matches, in token-ID order."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    normalized = query.casefold().strip()
    if not normalized:
        return ()
    exact: list[int] = []
    partial: list[int] = []
    for token_id, label in enumerate(token_labels):
        comparable = label.casefold()
        if comparable == normalized:
            exact.append(token_id)
        elif normalized in comparable:
            partial.append(token_id)
    return tuple((exact + partial)[:limit])


def plot_token_ids(
    vocabulary_size: int,
    *,
    include: Iterable[int] = (),
    limit: int = MAX_PLOT_POINTS,
) -> tuple[int, ...]:
    """Select a bounded deterministic point set while retaining highlighted tokens."""
    if vocabulary_size < 0:
        raise ValueError("vocabulary_size must not be negative")
    if limit <= 0:
        raise ValueError("limit must be positive")
    required = {token_id for token_id in include if 0 <= token_id < vocabulary_size}
    if vocabulary_size <= limit:
        return tuple(range(vocabulary_size))
    if len(required) > limit:
        raise ValueError("limit must accommodate all highlighted token IDs")
    available_slots = limit - len(required)
    sampled = [
        token_id
        for token_id in _evenly_spaced_indices(vocabulary_size, limit).tolist()
        if token_id not in required
    ][:available_slots]
    return tuple(sorted(set(sampled).union(required)))


class EmbeddingProjectionCache:
    """Versioned JSON cache for PCA coordinates belonging to one checkpoint digest."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def load(self, checkpoint_digest: str, token_labels: tuple[str, ...]) -> EmbeddingProjection | None:
        path = self._path_for(checkpoint_digest)
        try:
            with path.open(encoding="utf-8") as source:
                payload = json.load(source)
        except (OSError, json.JSONDecodeError):
            return None
        return _projection_from_payload(payload, checkpoint_digest, token_labels)

    def save(self, projection: EmbeddingProjection) -> Path:
        path = self._path_for(projection.checkpoint_digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        payload = {
            "version": CACHE_VERSION,
            "method": PCA_METHOD,
            "checkpoint_digest": projection.checkpoint_digest,
            "token_labels": list(projection.token_labels),
            "coordinates": [list(point) for point in projection.coordinates],
        }
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
        return path

    def load_or_compute(
        self, checkpoint_digest: str, token_labels: tuple[str, ...], embeddings: Tensor
    ) -> EmbeddingProjection:
        cached = self.load(checkpoint_digest, token_labels)
        if cached is not None:
            return cached
        coordinates = pca_2d(embeddings)
        projection = EmbeddingProjection(
            checkpoint_digest=checkpoint_digest,
            token_labels=token_labels,
            coordinates=tuple((float(point[0]), float(point[1])) for point in coordinates.tolist()),
        )
        try:
            self.save(projection)
        except OSError:
            pass
        return projection

    def _path_for(self, checkpoint_digest: str) -> Path:
        if not _CACHE_KEY_PATTERN.fullmatch(checkpoint_digest):
            raise ValueError("checkpoint_digest must contain only letters, digits, underscores, or hyphens")
        return self.directory / f"{checkpoint_digest}.json"


class CheckpointDigestCache:
    """Avoid re-hashing an unchanged standalone checkpoint during app reruns."""

    def __init__(self) -> None:
        self._digests: dict[Path, tuple[int, int, str]] = {}

    def digest(self, path: str | Path) -> str:
        checkpoint_path = Path(path).resolve()
        stat = checkpoint_path.stat()
        current_identity = (stat.st_size, stat.st_mtime_ns)
        cached = self._digests.get(checkpoint_path)
        if cached is not None and cached[:2] == current_identity:
            return cached[2]
        hasher = hashlib.sha256()
        with checkpoint_path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        self._digests[checkpoint_path] = (*current_identity, digest)
        return digest


def _projection_from_payload(
    payload: object, checkpoint_digest: str, token_labels: tuple[str, ...]
) -> EmbeddingProjection | None:
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("version") != CACHE_VERSION
        or payload.get("method") != PCA_METHOD
        or payload.get("checkpoint_digest") != checkpoint_digest
        or payload.get("token_labels") != list(token_labels)
    ):
        return None
    values = payload.get("coordinates")
    if not isinstance(values, list) or len(values) != len(token_labels):
        return None
    coordinates: list[tuple[float, float]] = []
    for point in values:
        if not isinstance(point, list) or len(point) != 2:
            return None
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        coordinates.append((x, y))
    return EmbeddingProjection(checkpoint_digest, token_labels, tuple(coordinates))


def _evenly_spaced_indices(size: int, limit: int) -> Tensor:
    if size <= limit:
        return torch.arange(size, dtype=torch.long)
    return torch.arange(limit, dtype=torch.long) * (size - 1) // (limit - 1)

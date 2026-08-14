import json

import pytest
import torch
from torch import nn

from tiny_llm_lab.embedding_analysis import (
    EmbeddingProjectionCache,
    extract_token_embeddings,
    nearest_neighbors,
    pca_2d,
    plot_token_ids,
)


class EmbeddingFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(4, 2)
        with torch.no_grad():
            self.token_embeddings.weight.copy_(
                torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
            )


def test_extract_token_embeddings_returns_a_detached_cpu_copy_with_vocabulary_shape() -> None:
    model = EmbeddingFixture()

    extracted = extract_token_embeddings(model)
    with torch.no_grad():
        model.token_embeddings.weight.zero_()

    assert extracted.device.type == "cpu"
    assert extracted.shape == (4, 2)
    torch.testing.assert_close(extracted, torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]))


def test_nearest_neighbors_excludes_query_handles_zero_norms_and_breaks_ties_by_token_id() -> None:
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.0, 0.0], [0.9, -0.1]]
    )

    neighbors = nearest_neighbors(embeddings, token_id=0, limit=4)

    assert [neighbor.token_id for neighbor in neighbors] == [1, 4, 2, 3]
    assert neighbors[0].cosine_similarity == pytest.approx(neighbors[1].cosine_similarity)
    assert neighbors[-1].cosine_similarity == 0.0


def test_pca_2d_is_repeatable_centered_and_uses_a_canonical_component_sign() -> None:
    embeddings = torch.tensor([[3.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-3.0, 0.0]])

    first = pca_2d(embeddings)
    second = pca_2d(embeddings)

    torch.testing.assert_close(first, second, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(first.mean(dim=0), torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(first[:, 0], torch.tensor([3.0, 1.0, -1.0, -3.0], dtype=torch.float64))
    torch.testing.assert_close(first[:, 1], torch.zeros(4, dtype=torch.float64), atol=1e-12, rtol=0)


def test_projection_cache_round_trips_coordinates_and_reuses_a_matching_entry(tmp_path) -> None:
    cache = EmbeddingProjectionCache(tmp_path)
    labels = ("a", "b", "c")
    embeddings = torch.tensor([[3.0, 0.0], [0.0, 1.0], [-3.0, 0.0]])

    first = cache.load_or_compute("checkpoint-digest", labels, embeddings)
    second = cache.load_or_compute("checkpoint-digest", labels, torch.zeros_like(embeddings))

    assert first == second
    payload = json.loads((tmp_path / "checkpoint-digest.json").read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["method"] == "pca"
    assert payload["checkpoint_digest"] == "checkpoint-digest"
    assert payload["token_labels"] == ["a", "b", "c"]


def test_projection_cache_rejects_serialized_coordinates_that_do_not_match_token_labels(tmp_path) -> None:
    cache = EmbeddingProjectionCache(tmp_path)
    (tmp_path / "checkpoint-digest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "method": "pca",
                "checkpoint_digest": "checkpoint-digest",
                "token_labels": ["a", "b"],
                "coordinates": [[0.0, 1.0]],
            }
        ),
        encoding="utf-8",
    )

    assert cache.load("checkpoint-digest", ("a", "b")) is None


def test_plot_sampling_keeps_highlighted_tokens_without_exceeding_the_point_limit() -> None:
    token_ids = plot_token_ids(10_050, limit=10_000, include=(10_048, 10_049))

    assert len(token_ids) == 10_000
    assert {10_048, 10_049}.issubset(token_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_nearest_neighbors_moves_cuda_embeddings_to_cpu_before_analysis() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], device="cuda")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()

    neighbors = nearest_neighbors(embeddings, token_id=0, limit=2)
    torch.cuda.synchronize()

    assert [neighbor.token_id for neighbor in neighbors] == [1, 2]
    assert torch.cuda.memory_allocated() == baseline

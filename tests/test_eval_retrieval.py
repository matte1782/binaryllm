"""Contract tests for Phase 1 T6 retrieval metrics.

Validates top-k neighbor search, overlap@k, nDCG@k, and recall@k for cosine vs.
Hamming rankings using the synthetic retrieval snapshot (see architecture §4.2).

Retrieval APIs always require an explicit `seed`; `seed=None` must raise ValueError.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval import retrieval as ret


SNAPSHOT_PATH = Path(__file__).resolve().parent / "data" / "phase1_synthetic" / "retrieval_toy_snapshot.npz"


@pytest.fixture(scope="module")
def retrieval_snapshot() -> dict[str, np.ndarray]:
    data = np.load(SNAPSHOT_PATH, allow_pickle=False)
    return {key: data[key] for key in data.files}


def test_topk_neighbor_indices_cosine_excludes_self(retrieval_snapshot: dict[str, np.ndarray]) -> None:
    embeddings = retrieval_snapshot["embeddings"]
    neighbors = ret.topk_neighbor_indices_cosine(embeddings, k=2, seed=3)
    assert neighbors.shape == (embeddings.shape[0], 2)
    for i, row in enumerate(neighbors):
        assert i not in row


def test_topk_neighbor_indices_cosine_requires_seed(retrieval_snapshot: dict[str, np.ndarray]) -> None:
    embeddings = retrieval_snapshot["embeddings"]
    with pytest.raises(ValueError, match="seed.*required"):
        ret.topk_neighbor_indices_cosine(embeddings, k=2, seed=None)


def test_topk_neighbor_indices_cosine_deterministic(retrieval_snapshot: dict[str, np.ndarray]) -> None:
    embeddings = retrieval_snapshot["embeddings"]
    first = ret.topk_neighbor_indices_cosine(embeddings, k=3, seed=5)
    second = ret.topk_neighbor_indices_cosine(embeddings.copy(), k=3, seed=5)
    np.testing.assert_array_equal(first, second)


def test_topk_neighbor_indices_hamming_orders_by_distance(retrieval_snapshot: dict[str, np.ndarray]) -> None:
    codes = retrieval_snapshot["codes_m64"]
    neighbors = ret.topk_neighbor_indices_hamming(codes, k=2, seed=7)
    assert neighbors.shape[1] == 2
    # For the synthetic snapshot, sample 0 should rank 1 before 2 due to lower Hamming distance.
    assert list(neighbors[0]) == [1, 2]


def test_topk_neighbor_indices_hamming_deterministic(retrieval_snapshot: dict[str, np.ndarray]) -> None:
    codes = retrieval_snapshot["codes_m32"]
    first = ret.topk_neighbor_indices_hamming(codes, k=3, seed=9)
    second = ret.topk_neighbor_indices_hamming(codes.copy(), k=3, seed=9)
    np.testing.assert_array_equal(first, second)


def test_topk_neighbor_indices_hamming_requires_seed(retrieval_snapshot: dict[str, np.ndarray]) -> None:
    codes = retrieval_snapshot["codes_m32"]
    with pytest.raises(ValueError, match="seed.*required"):
        ret.topk_neighbor_indices_hamming(codes, k=2, seed=None)


def test_neighbor_overlap_at_k_increases_with_code_bits(retrieval_snapshot: dict[str, np.ndarray]) -> None:
    embeddings = retrieval_snapshot["embeddings"]
    codes32 = retrieval_snapshot["codes_m32"]
    codes64 = retrieval_snapshot["codes_m64"]
    codes128 = retrieval_snapshot["codes_m128"]

    cos_neighbors = ret.topk_neighbor_indices_cosine(embeddings, k=3, seed=11)
    overlap32 = ret.neighbor_overlap_at_k(cos_neighbors, ret.topk_neighbor_indices_hamming(codes32, k=3, seed=13))
    overlap64 = ret.neighbor_overlap_at_k(cos_neighbors, ret.topk_neighbor_indices_hamming(codes64, k=3, seed=13))
    overlap128 = ret.neighbor_overlap_at_k(cos_neighbors, ret.topk_neighbor_indices_hamming(codes128, k=3, seed=13))

    assert overlap32 <= overlap64 <= overlap128


def test_neighbor_overlap_at_k_k_validation(retrieval_snapshot: dict[str, np.ndarray]) -> None:
    embeddings = retrieval_snapshot["embeddings"]
    cos_neighbors = ret.topk_neighbor_indices_cosine(embeddings, k=3, seed=17)
    ham_neighbors = ret.topk_neighbor_indices_hamming(retrieval_snapshot["codes_m32"], k=3, seed=19)
    with pytest.raises(ValueError, match="k.*dataset size"):
        ret.neighbor_overlap_at_k(cos_neighbors, ham_neighbors[:, :2])


def test_topk_allows_k_equal_to_dataset_size(retrieval_snapshot: dict[str, np.ndarray]) -> None:
    embeddings = retrieval_snapshot["embeddings"]
    k = embeddings.shape[0]
    neighbors = ret.topk_neighbor_indices_cosine(embeddings, k=k, seed=43)
    assert neighbors.shape == (k, k)


def test_topk_rejects_k_greater_than_dataset_size(retrieval_snapshot: dict[str, np.ndarray]) -> None:
    embeddings = retrieval_snapshot["embeddings"]
    with pytest.raises(ValueError, match="k.*<=.*dataset size"):
        ret.topk_neighbor_indices_cosine(embeddings, k=embeddings.shape[0] + 1, seed=47)


def test_ndcg_at_k_matches_manual_computation(retrieval_snapshot: dict[str, np.ndarray]) -> None:
    """nDCG@k must normalize DCG by IDCG for each query."""
    relevance = retrieval_snapshot["ground_truth"]
    ranking = ret.topk_neighbor_indices_cosine(retrieval_snapshot["embeddings"], k=3, seed=23)
    ndcg_values = ret.ndcg_at_k(ranking, relevance, k=3)

    assert ndcg_values.shape == (relevance.shape[0],)
    np.testing.assert_allclose(ndcg_values[:3], np.array([1.0, 1.0, 1.0], dtype=np.float32), atol=1e-6)


def test_ndcg_at_k_zero_if_no_relevant_docs() -> None:
    ranking = np.array([[2, 1, 0]], dtype=np.int32)
    relevance = np.zeros((1, 5), dtype=np.float32)
    ndcg = ret.ndcg_at_k(ranking, relevance, k=3)
    np.testing.assert_allclose(ndcg, np.array([0.0], dtype=np.float32))


def test_recall_at_k_matches_ground_truth(retrieval_snapshot: dict[str, np.ndarray]) -> None:
    relevance = retrieval_snapshot["ground_truth"]
    ranking = ret.topk_neighbor_indices_hamming(retrieval_snapshot["codes_m64"], k=3, seed=37)
    recall = ret.recall_at_k(ranking, relevance, k=3)

    expected = np.array([1.0, 1.0, 1.0, 0.5, 1.0], dtype=np.float32)
    np.testing.assert_allclose(recall, expected, atol=1e-6)


def test_recall_at_k_rejects_invalid_k(retrieval_snapshot: dict[str, np.ndarray]) -> None:
    ranking = ret.topk_neighbor_indices_cosine(retrieval_snapshot["embeddings"], k=3, seed=41)
    with pytest.raises(ValueError, match="k.*positive"):
        ret.recall_at_k(ranking, retrieval_snapshot["ground_truth"], k=0)


def test_ndcg_at_k_rejects_bad_shapes(retrieval_snapshot: dict[str, np.ndarray]) -> None:
    ranking = ret.topk_neighbor_indices_cosine(retrieval_snapshot["embeddings"], k=3, seed=31)
    bad_relevance = retrieval_snapshot["ground_truth"][:-1]
    with pytest.raises(ValueError, match="relevance.*queries"):
        ret.ndcg_at_k(ranking, bad_relevance, k=3)


def test_topk_neighbor_indices_cosine_rejects_bad_inputs(retrieval_snapshot: dict[str, np.ndarray]) -> None:
    bad_embeddings = retrieval_snapshot["embeddings"][0]
    with pytest.raises(ValueError, match="embeddings.*2D"):
        ret.topk_neighbor_indices_cosine(bad_embeddings, k=2)
    with pytest.raises(ValueError, match="k.*positive"):
        ret.topk_neighbor_indices_cosine(retrieval_snapshot["embeddings"], k=0)


def test_topk_neighbor_indices_hamming_validates_codes(retrieval_snapshot: dict[str, np.ndarray]) -> None:
    invalid_codes = retrieval_snapshot["codes_m32"].astype(np.int8)
    invalid_codes[0, 0] = 2
    with pytest.raises(ValueError, match=r"codes_01.*\{0,1\}"):
        ret.topk_neighbor_indices_hamming(invalid_codes, k=2)


def test_retrieval_metrics_are_deterministic(retrieval_snapshot: dict[str, np.ndarray]) -> None:
    embeddings = retrieval_snapshot["embeddings"]
    cosine_neighbors_a = ret.topk_neighbor_indices_cosine(embeddings, k=3, seed=101)
    cosine_neighbors_b = ret.topk_neighbor_indices_cosine(embeddings, k=3, seed=101)
    np.testing.assert_array_equal(cosine_neighbors_a, cosine_neighbors_b)

    codes = retrieval_snapshot["codes_m128"]
    hamming_neighbors_a = ret.topk_neighbor_indices_hamming(codes, k=3, seed=202)
    hamming_neighbors_b = ret.topk_neighbor_indices_hamming(codes, k=3, seed=202)
    np.testing.assert_array_equal(hamming_neighbors_a, hamming_neighbors_b)

    overlap_a = ret.neighbor_overlap_at_k(cosine_neighbors_a, hamming_neighbors_a)
    overlap_b = ret.neighbor_overlap_at_k(cosine_neighbors_b, hamming_neighbors_b)
    assert overlap_a == pytest.approx(overlap_b, rel=1e-9)


def test_retrieval_performance_sanity(retrieval_snapshot: dict[str, np.ndarray]) -> None:
    embeddings = np.tile(retrieval_snapshot["embeddings"], (10, 1))
    codes = np.tile(retrieval_snapshot["codes_m64"], (10, 1))
    ranking = ret.topk_neighbor_indices_cosine(embeddings, k=3, seed=59)
    hamming_neighbors = ret.topk_neighbor_indices_hamming(codes, k=3, seed=59)
    overlap = ret.neighbor_overlap_at_k(ranking, hamming_neighbors)
    assert 0.0 <= overlap <= 1.0


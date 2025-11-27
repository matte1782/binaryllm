"""Contract tests for Phase 1 T5 similarity + ranking metrics.

These tests define the required API for src.eval.similarity:
- cosine_similarity_matrix
- hamming_distance_matrix
- spearman_correlation_from_matrices
- topk_neighbor_indices
- neighbor_overlap_at_k
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from src.eval import similarity as sim


SNAPSHOT_PATH = (
    Path(__file__).resolve().parent / "data" / "phase1_synthetic" / "similarity_toy_snapshot.npz"
)


@pytest.fixture(scope="module")
def similarity_snapshot() -> Dict[str, np.ndarray]:
    """Load deterministic synthetic embeddings/codes and golden outputs."""
    data = np.load(SNAPSHOT_PATH, allow_pickle=False)
    return {key: data[key] for key in data.files}


def _manual_average_spearman(cosine_matrix: np.ndarray, hamming_matrix: np.ndarray) -> float:
    """Reference implementation to cross-check module output."""
    n = cosine_matrix.shape[0]
    correlations: list[float] = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        cos_scores = cosine_matrix[i, mask]
        ham_scores = hamming_matrix[i, mask]
        other_indices = np.arange(n)[mask]

        cos_order = np.lexsort((other_indices, -cos_scores))
        ham_order = np.lexsort((other_indices, ham_scores))

        # ranks: position of each neighbor in the ordered list.
        cos_ranks = np.empty_like(cos_order)
        cos_ranks[cos_order] = np.arange(cos_order.size)
        ham_ranks = np.empty_like(ham_order)
        ham_ranks[ham_order] = np.arange(ham_order.size)

        diffs = cos_ranks - ham_ranks
        denom = cos_order.size * (cos_order.size**2 - 1)
        if denom == 0:
            correlations.append(1.0)
        else:
            correlations.append(1.0 - (6.0 * np.sum(diffs**2)) / denom)
    return float(np.mean(correlations))


def test_cosine_similarity_matrix_matches_golden(similarity_snapshot: Dict[str, np.ndarray]) -> None:
    embeddings = similarity_snapshot["embeddings"]
    result = sim.cosine_similarity_matrix(embeddings, assume_normalized=True)
    np.testing.assert_allclose(result, similarity_snapshot["cosine_matrix"], atol=1e-6)
    np.testing.assert_allclose(np.diag(result), np.ones(result.shape[0]), atol=1e-6)
    assert np.allclose(result, result.T, atol=1e-6)


def test_cosine_similarity_normalizes_when_requested(similarity_snapshot: Dict[str, np.ndarray]) -> None:
    """Cosine must match manual L2 normalization; no hidden column heuristics allowed."""
    embeddings = similarity_snapshot["embeddings"] * np.array([[2.0, 1.0, 1.0]], dtype=np.float32)
    manually_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    expected = manually_normalized @ manually_normalized.T
    result = sim.cosine_similarity_matrix(embeddings, assume_normalized=False)
    np.testing.assert_allclose(result, expected.astype(np.float32), atol=1e-6)


def test_cosine_similarity_validates_inputs(similarity_snapshot: Dict[str, np.ndarray]) -> None:
    vector = similarity_snapshot["embeddings"][0]
    with pytest.raises(ValueError, match="embeddings.*2D"):
        sim.cosine_similarity_matrix(vector, assume_normalized=True)
    nan_emb = similarity_snapshot["embeddings"].copy()
    nan_emb[0, 0] = np.nan
    with pytest.raises(ValueError, match="embeddings.*NaN"):
        sim.cosine_similarity_matrix(nan_emb, assume_normalized=True)


def test_hamming_distance_matrix_matches_golden(similarity_snapshot: Dict[str, np.ndarray]) -> None:
    codes = similarity_snapshot["codes_m128"]
    result = sim.hamming_distance_matrix(codes)
    np.testing.assert_allclose(result, similarity_snapshot["hamming128_matrix"], atol=1e-6)
    assert np.allclose(np.diag(result), 0.0)
    assert np.allclose(result, result.T)


def test_hamming_distance_validates_binary_inputs() -> None:
    invalid = np.array([[0, 2, 1, 0]], dtype=np.int8)
    with pytest.raises(ValueError, match=r"codes_01.*\{0,1\}"):
        sim.hamming_distance_matrix(invalid)


def test_hamming_distance_validates_shape() -> None:
    flat = np.array([0, 1, 0, 1], dtype=np.uint8)
    with pytest.raises(ValueError, match="codes_01.*2D"):
        sim.hamming_distance_matrix(flat)


def test_spearman_correlation_matches_reference(similarity_snapshot: Dict[str, np.ndarray]) -> None:
    cosine_matrix = similarity_snapshot["cosine_matrix"]
    hamming_matrix = sim.hamming_distance_matrix(similarity_snapshot["codes_m128"])
    module_value = sim.spearman_correlation_from_matrices(cosine_matrix, hamming_matrix)
    manual_value = _manual_average_spearman(cosine_matrix, hamming_matrix)
    assert module_value == pytest.approx(manual_value, rel=1e-6)
    assert module_value > 0.9


def test_spearman_correlation_rejects_shape_mismatch(similarity_snapshot: Dict[str, np.ndarray]) -> None:
    cosine_matrix = similarity_snapshot["cosine_matrix"]
    smaller = cosine_matrix[:-1, :-1]
    with pytest.raises(ValueError, match="shape"):
        sim.spearman_correlation_from_matrices(cosine_matrix, smaller)


def test_neighbor_overlap_monotonic_improvement(similarity_snapshot: Dict[str, np.ndarray]) -> None:
    cosine_matrix = similarity_snapshot["cosine_matrix"]
    codes32 = similarity_snapshot["codes_m32"]
    codes64 = similarity_snapshot["codes_m64"]
    codes128 = similarity_snapshot["codes_m128"]

    overlap32 = sim.neighbor_overlap_at_k(cosine_matrix, sim.hamming_distance_matrix(codes32), k=3)
    overlap64 = sim.neighbor_overlap_at_k(cosine_matrix, sim.hamming_distance_matrix(codes64), k=3)
    overlap128 = sim.neighbor_overlap_at_k(cosine_matrix, sim.hamming_distance_matrix(codes128), k=3)

    # Spec uses standard overlap fraction only; we only require non-decreasing behavior.
    assert overlap32 <= overlap64 <= overlap128
    assert overlap128 >= 0.8


def test_topk_neighbors_are_deterministic(similarity_snapshot: Dict[str, np.ndarray]) -> None:
    cosine_matrix = similarity_snapshot["cosine_matrix"]
    hamming_matrix = similarity_snapshot["hamming128_matrix"]

    first_cos = sim.topk_neighbor_indices(cosine_matrix, k=3, mode="cosine")
    second_cos = sim.topk_neighbor_indices(cosine_matrix.copy(), k=3, mode="cosine")
    np.testing.assert_array_equal(first_cos, second_cos)
    np.testing.assert_array_equal(first_cos, similarity_snapshot["topk_cosine_k3"])

    first_ham = sim.topk_neighbor_indices(hamming_matrix, k=3, mode="hamming")
    second_ham = sim.topk_neighbor_indices(hamming_matrix.copy(), k=3, mode="hamming")
    np.testing.assert_array_equal(first_ham, second_ham)
    np.testing.assert_array_equal(first_ham, similarity_snapshot["topk_hamming128_k3"])


def test_neighbor_overlap_rejects_invalid_k(similarity_snapshot: Dict[str, np.ndarray]) -> None:
    cosine_matrix = similarity_snapshot["cosine_matrix"]
    hamming_matrix = similarity_snapshot["hamming128_matrix"]
    with pytest.raises(ValueError, match="k"):
        sim.neighbor_overlap_at_k(cosine_matrix, hamming_matrix, k=0)


def test_similarity_pipeline_regression_snapshot(similarity_snapshot: Dict[str, np.ndarray]) -> None:
    cosine_matrix = sim.cosine_similarity_matrix(similarity_snapshot["embeddings"], assume_normalized=True)
    hamming_matrix = sim.hamming_distance_matrix(similarity_snapshot["codes_m128"])
    np.testing.assert_allclose(cosine_matrix, similarity_snapshot["cosine_matrix"], atol=1e-6)
    np.testing.assert_allclose(hamming_matrix, similarity_snapshot["hamming128_matrix"], atol=1e-6)

    cos_neighbors = sim.topk_neighbor_indices(cosine_matrix, k=3, mode="cosine")
    ham_neighbors = sim.topk_neighbor_indices(hamming_matrix, k=3, mode="hamming")
    np.testing.assert_array_equal(cos_neighbors, similarity_snapshot["topk_cosine_k3"])
    np.testing.assert_array_equal(ham_neighbors, similarity_snapshot["topk_hamming128_k3"])


def test_functions_are_deterministic(similarity_snapshot: Dict[str, np.ndarray]) -> None:
    cosine_matrix = sim.cosine_similarity_matrix(similarity_snapshot["embeddings"], assume_normalized=True)
    repeat_cosine = sim.cosine_similarity_matrix(similarity_snapshot["embeddings"], assume_normalized=True)
    np.testing.assert_array_equal(cosine_matrix, repeat_cosine)

    hamming_once = sim.hamming_distance_matrix(similarity_snapshot["codes_m64"])
    hamming_twice = sim.hamming_distance_matrix(similarity_snapshot["codes_m64"])
    np.testing.assert_array_equal(hamming_once, hamming_twice)


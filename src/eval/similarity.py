"""Similarity and ranking utilities for BinaryLLM Phase 1 (T5).

This module operates on logical float embeddings (cosine space) and logical {0,1}
codes (Hamming space). It intentionally avoids packed representations to keep
the API simple for Phase 1 testing.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

MatrixMode = Literal["cosine", "hamming"]


def _validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("embeddings must be a 2D array for cosine similarity.")
    if array.size == 0:
        raise ValueError("embeddings must not be empty for cosine similarity.")
    if not np.isfinite(array).all():
        raise ValueError("embeddings must not contain NaN or Inf values.")
    return array


def _l2_normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embeddings normalization requires non-zero row norms.")
    return array / norms


def cosine_similarity_matrix(embeddings: np.ndarray, *, assume_normalized: bool) -> np.ndarray:
    """Return cosine similarity matrix for embeddings (float32 output)."""
    array = _validate_embeddings(embeddings).astype(np.float64, copy=False)
    if not assume_normalized:
        array = _l2_normalize_rows(array)
    sim_matrix = array @ array.T
    return sim_matrix.astype(np.float32, copy=False)


def _validate_codes_01(codes_01: np.ndarray) -> np.ndarray:
    array = np.asarray(codes_01)
    if array.ndim != 2:
        raise ValueError("codes_01 must be a 2D array for Hamming distance.")
    if array.size == 0:
        raise ValueError("codes_01 must not be empty for Hamming distance.")
    if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_)):
        raise ValueError("codes_01 must be an integer/bool array containing only {0,1}.")
    if not np.isin(array, (0, 1)).all():
        raise ValueError("codes_01 must contain only {0,1} values.")
    return array.astype(np.uint8, copy=False)


def hamming_distance_matrix(codes_01: np.ndarray) -> np.ndarray:
    """Compute pairwise Hamming distances for {0,1} codes."""
    codes = _validate_codes_01(codes_01)
    xor = np.bitwise_xor(codes[:, None, :], codes[None, :, :])
    distances = xor.sum(axis=2, dtype=np.int32)
    return distances.astype(np.float32, copy=False)


def _stable_rank_order(scores: np.ndarray, ascending: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted order and ranks using lexsort for deterministic tie-breaking."""
    indices = np.arange(scores.size)
    if ascending:
        order = np.lexsort((indices, scores))
    else:
        order = np.lexsort((indices, -scores))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.size)
    return order, ranks


def spearman_correlation_from_matrices(
    cosine_matrix: np.ndarray,
    hamming_matrix: np.ndarray,
) -> float:
    """Average Spearman correlation between cosine vs Hamming rankings."""
    cos = np.asarray(cosine_matrix, dtype=np.float64)
    ham = np.asarray(hamming_matrix, dtype=np.float64)
    if cos.shape != ham.shape:
        raise ValueError("cosine_matrix and hamming_matrix must share the same shape.")
    if cos.ndim != 2 or cos.shape[0] != cos.shape[1]:
        raise ValueError("similarity matrices must be square.")
    n = cos.shape[0]
    if n <= 1:
        return 1.0

    correlations = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        cos_scores = cos[i, mask]
        ham_scores = ham[i, mask]

        cos_order, cos_ranks = _stable_rank_order(cos_scores, ascending=False)
        ham_order, ham_ranks = _stable_rank_order(ham_scores, ascending=True)

        diffs = cos_ranks - ham_ranks
        denom = cos_scores.size * (cos_scores.size**2 - 1)
        if denom == 0:
            correlations.append(1.0)
        else:
            correlations.append(1.0 - (6.0 * np.sum(diffs**2)) / denom)

    return float(np.mean(correlations))


def topk_neighbor_indices(
    matrix: np.ndarray,
    *,
    k: int,
    mode: MatrixMode,
) -> np.ndarray:
    """Return deterministic top-k neighbor indices for cosine or Hamming matrices."""
    scores = np.asarray(matrix)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("matrix must be square for neighbor retrieval.")
    n = scores.shape[0]
    if not 0 < k < n:
        raise ValueError("k must satisfy 0 < k < number of samples.")
    if mode not in {"cosine", "hamming"}:
        raise ValueError("mode must be either 'cosine' or 'hamming'.")

    neighbors = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        row_scores = scores[i]
        indices = np.arange(n)
        keep = indices != i
        row_scores = row_scores[keep]
        indices = indices[keep]

        ascending = mode == "hamming"
        order = np.lexsort((indices, row_scores if ascending else -row_scores))
        neighbors[i] = indices[order[:k]]
    return neighbors


def neighbor_overlap_at_k(
    cosine_matrix: np.ndarray,
    hamming_matrix: np.ndarray,
    *,
    k: int,
) -> float:
    """Average top-k neighbor overlap between cosine and Hamming metrics."""
    cos_neighbors = topk_neighbor_indices(cosine_matrix, k=k, mode="cosine")
    ham_neighbors = topk_neighbor_indices(hamming_matrix, k=k, mode="hamming")
    overlaps = [
        len(set(cos_neighbors[i]).intersection(ham_neighbors[i])) / k for i in range(cos_neighbors.shape[0])
    ]
    return float(np.mean(overlaps))


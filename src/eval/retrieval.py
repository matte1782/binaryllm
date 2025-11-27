"""Retrieval utilities for Phase 1 T6 (cosine vs Hamming evaluation)."""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

Mode = Literal["cosine", "hamming"]


def _validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings)
    if array.ndim != 2:
        raise ValueError("embeddings must be a 2D array for retrieval.")
    if not np.isfinite(array).all():
        raise ValueError("embeddings must not contain NaN or Inf values.")
    return array.astype(np.float64, copy=False)


def _validate_codes_01(codes_01: np.ndarray) -> np.ndarray:
    array = np.asarray(codes_01)
    if array.ndim != 2:
        raise ValueError("codes_01 must be a 2D array for Hamming retrieval.")
    if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_)):
        raise ValueError("codes_01 must contain only {0,1} values.")
    if not np.isin(array, (0, 1)).all():
        raise ValueError("codes_01 must contain only {0,1} values.")
    return array.astype(np.uint8, copy=False)


def _validate_k(k: int, n: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > n:
        raise ValueError("k must be <= dataset size.")


def _rng_from_seed(seed: int | None, query_index: int = 0) -> np.random.Generator:
    if seed is None:
        raise ValueError("seed parameter is required for deterministic retrieval.")
    return np.random.default_rng(int(seed) + query_index)


def _deterministic_order(values: np.ndarray, seed: int | None, query_index: int, ascending: bool) -> np.ndarray:
    priority = _rng_from_seed(seed, query_index).permutation(values.shape[0])
    primary = values if ascending else -values
    return np.lexsort((priority, primary))


def topk_neighbor_indices_cosine(
    embeddings: np.ndarray,
    k: int,
    *,
    seed: int | None = None,
    assume_normalized: bool = True,
) -> np.ndarray:
    array = _validate_embeddings(embeddings)
    n = array.shape[0]
    _validate_k(k, n)
    if not assume_normalized:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("embeddings normalization requires non-zero row norms.")
        array = array / norms

    similarities = array @ array.T
    neighbors = np.zeros((n, k), dtype=np.int32)
    for i in range(n):
        scores = similarities[i].copy()
        scores[i] = -np.inf
        order = _deterministic_order(scores, seed, i, ascending=False)
        neighbors[i] = order[:k]
    return neighbors


def topk_neighbor_indices_hamming(
    codes_01: np.ndarray,
    k: int,
    *,
    seed: int | None = None,
) -> np.ndarray:
    codes = _validate_codes_01(codes_01)
    n = codes.shape[0]
    _validate_k(k, n)
    neighbors = np.zeros((n, k), dtype=np.int32)
    for i in range(n):
        xor = np.bitwise_xor(codes[i], codes)
        dist = xor.sum(axis=1).astype(np.float64)
        dist[i] = dist.max() + 1.0
        order = _deterministic_order(dist, seed, i, ascending=True)
        neighbors[i] = order[:k]
    return neighbors


def neighbor_overlap_at_k(cosine_topk: np.ndarray, hamming_topk: np.ndarray) -> float:
    if cosine_topk.ndim != 2 or hamming_topk.ndim != 2:
        raise ValueError("neighbor overlap requires 2D inputs.")
    if cosine_topk.shape[1] != hamming_topk.shape[1]:
        raise ValueError("k must be <= dataset size for both cosine and hamming top-k.")
    if cosine_topk.shape[0] != hamming_topk.shape[0]:
        raise ValueError("neighbor overlap requires matching batch sizes.")
    overlap = [
        len(set(cosine_topk[i]).intersection(hamming_topk[i])) / cosine_topk.shape[1]
        for i in range(cosine_topk.shape[0])
    ]
    return float(np.mean(overlap))


def _validate_relevance(ranked_indices: np.ndarray, relevance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ranks = np.asarray(ranked_indices, dtype=np.int64)
    rel = np.asarray(relevance, dtype=np.float32)
    if ranks.ndim != 2 or rel.ndim != 2:
        raise ValueError("relevance and ranking must be 2D arrays.")
    if ranks.shape[0] != rel.shape[0]:
        raise ValueError("relevance shape must match ranking shape (same number of queries).")
    if ranks.shape[1] > rel.shape[1]:
        raise ValueError("k must be <= number of available relevance columns.")
    return ranks, rel


def ndcg_at_k(ranked_indices: np.ndarray, relevance: np.ndarray, *, k: int) -> np.ndarray:
    ranks, rel = _validate_relevance(ranked_indices, relevance)
    if k <= 0:
        raise ValueError("k must be positive for nDCG.")
    k = min(k, ranks.shape[1], rel.shape[1])
    indices = np.clip(ranks[:, :k], 0, rel.shape[1] - 1)
    gains = np.take_along_axis(rel, indices, axis=1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (gains * discounts).sum(axis=1)

    ideal_sorted = np.sort(rel, axis=1)[:, ::-1][:, :k]
    ideal_dcg = (ideal_sorted * discounts).sum(axis=1)

    ndcg = np.zeros_like(dcg, dtype=np.float64)
    nonzero = ideal_dcg > 0
    ndcg[nonzero] = dcg[nonzero] / ideal_dcg[nonzero]
    return ndcg.astype(np.float32)


def recall_at_k(ranked_indices: np.ndarray, relevance: np.ndarray, *, k: int) -> np.ndarray:
    ranks, rel = _validate_relevance(ranked_indices, relevance)
    if k <= 0:
        raise ValueError("k must be positive for recall@k.")
    k = min(k, ranks.shape[1], rel.shape[1])
    indices = np.clip(ranks[:, :k], 0, rel.shape[1] - 1)
    relevant_in_topk = np.take_along_axis(rel, indices, axis=1).sum(axis=1)
    total_relevant = rel.sum(axis=1)
    total_relevant[total_relevant == 0] = 1.0
    return (relevant_in_topk / total_relevant).astype(np.float32)


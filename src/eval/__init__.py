"""
Evaluation metrics for BinaryLLM Phase 1.

This module provides three families of evaluation metrics:

Similarity Metrics (src.eval.similarity):
    - cosine_similarity_matrix: Pairwise cosine similarities
    - hamming_distance_matrix: Pairwise Hamming distances
    - spearman_correlation_from_matrices: Rank correlation
    - neighbor_overlap_at_k: Top-k neighbor consistency

Retrieval Metrics (src.eval.retrieval):
    - topk_neighbor_indices_cosine/hamming: Top-k neighbor retrieval
    - neighbor_overlap_at_k: Neighbor set overlap
    - ndcg_at_k: Normalized discounted cumulative gain
    - recall_at_k: Recall at k

Classification Metrics (src.eval.classification):
    - evaluate_classification: Centroid classifier with degradation analysis
"""

from src.eval.similarity import (
    cosine_similarity_matrix,
    hamming_distance_matrix,
    spearman_correlation_from_matrices,
    neighbor_overlap_at_k as similarity_neighbor_overlap_at_k,
)
from src.eval.retrieval import (
    topk_neighbor_indices_cosine,
    topk_neighbor_indices_hamming,
    neighbor_overlap_at_k as retrieval_neighbor_overlap_at_k,
    ndcg_at_k,
    recall_at_k,
)
from src.eval.classification import evaluate_classification

__all__ = [
    "cosine_similarity_matrix",
    "hamming_distance_matrix",
    "spearman_correlation_from_matrices",
    "similarity_neighbor_overlap_at_k",
    "topk_neighbor_indices_cosine",
    "topk_neighbor_indices_hamming",
    "retrieval_neighbor_overlap_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "evaluate_classification",
]

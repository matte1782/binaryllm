"""Phase 1 Binary Embedding Engine façade (T10).

This façade owns the numerical pipeline for Phase 1:
- optional L2 normalization of float embeddings,
- random projection + binarization ({-1,+1} and {0,1} forms),
- bit packing,
- similarity / retrieval / classification evaluation,
- lightweight instrumentation and logging hooks.

Runners instantiate the façade with catalog-provided specs and delegate the
entire pipeline to :meth:`BinaryEmbeddingEngine.run`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.core.dataset_catalog import DatasetSpec, EncoderSpec
from src.eval import classification as classification_eval
from src.eval import retrieval as retrieval_eval
from src.eval import similarity as similarity_eval
from src.quantization import packing as packing_utils
from src.quantization.binarization import RandomProjection, binarize_sign

SimilarityMetrics = Dict[str, float]
RetrievalMetrics = Dict[str, Dict[str, float]]
ClassificationMetrics = Optional[Dict[str, float]]

SUPPORTED_METRICS = ("similarity", "retrieval", "classification")
SUPPORTED_PROJECTIONS = {"gaussian"}


def _ensure_tuple(values: Iterable[str]) -> Tuple[str, ...]:
    sequence = tuple(values)
    if not sequence:
        raise ValueError("metrics collection must be non-empty.")
    unknown = set(sequence) - set(SUPPORTED_METRICS)
    if unknown:
        raise ValueError(f"metrics contain unsupported entries: {sorted(unknown)}")
    return sequence


def _validate_embeddings(array: ArrayLike, expected_dim: int) -> NDArray[np.float32]:
    embeddings = np.asarray(array, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array.")
    if embeddings.shape[1] != expected_dim:
        raise ValueError("embedding_dim mismatch in BinaryEmbeddingEngine.run.")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must be finite (no NaN/Inf values).")
    return embeddings


def _l2_normalize(array: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot L2 normalize embeddings with zero vectors.")
    return array / norms


@dataclass(slots=True)
class BinaryEmbeddingEngine:
    """Encapsulates the full Phase 1 embedding → metrics pipeline."""

    encoder_spec: Optional[EncoderSpec] = None
    dataset_spec: Optional[DatasetSpec] = None
    code_bits: int = 0
    projection_type: Optional[str] = None
    seed: Optional[int] = None
    normalize: bool = True
    projection_seed: Optional[int] = None
    _projection: RandomProjection = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for field_name in ("encoder_spec", "dataset_spec", "projection_type", "seed"):
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} must be provided for BinaryEmbeddingEngine.")
        self.encoder_spec = cast(EncoderSpec, self.encoder_spec)
        self.dataset_spec = cast(DatasetSpec, self.dataset_spec)
        self.projection_type = cast(str, self.projection_type)
        self.seed = int(cast(int, self.seed))
        if self.code_bits <= 0:
            raise ValueError("code_bits must be positive.")
        if self.projection_type not in SUPPORTED_PROJECTIONS:
            raise ValueError(f"projection_type '{self.projection_type}' is not supported.")
        self.projection_seed = int(self.projection_seed or self.seed)
        self._projection = RandomProjection(
            input_dim=self.encoder_spec.embedding_dim,
            output_bits=self.code_bits,
            seed=self.projection_seed,
        )

    # --------------------------------------------------------------------- API
    def normalize_embeddings(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        if not self.normalize:
            return embeddings
        return _l2_normalize(embeddings)

    def project(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        return self._projection.project(embeddings)

    def binarize(self, projected: NDArray[np.float32]) -> Tuple[NDArray[np.float32], NDArray[np.int8]]:
        codes_pm1 = binarize_sign(projected).astype(np.float32, copy=False)
        codes_01 = ((codes_pm1 + 1.0) / 2.0).astype(np.int8, copy=False)
        return codes_pm1, codes_01

    def pack(self, codes_01: NDArray[np.int8]) -> NDArray[np.uint64]:
        return packing_utils.pack_codes(codes_01)

    def unpack(self, packed: NDArray[np.uint64]) -> NDArray[np.uint8]:
        return packing_utils.unpack_codes(packed, self.code_bits)

    # ----------------------------------------------------------------- helpers
    def _ensure_dataset_capabilities(self, metrics: Tuple[str, ...]) -> None:
        spec = self.dataset_spec
        if "similarity" in metrics and not spec.supports_similarity:
            raise ValueError(f"dataset '{spec.name}' does not support similarity metrics.")
        if "retrieval" in metrics and not spec.supports_retrieval:
            raise ValueError(f"dataset '{spec.name}' does not support retrieval metrics.")
        if "classification" in metrics and not spec.supports_classification:
            raise ValueError(f"dataset '{spec.name}' does not support classification metrics.")

    def _compute_similarity_metrics(
        self,
        normalized_embeddings: NDArray[np.float32],
        codes_01: NDArray[np.int8],
    ) -> SimilarityMetrics:
        cosine_matrix = similarity_eval.cosine_similarity_matrix(
            normalized_embeddings, assume_normalized=True
        )
        raw_hamming = similarity_eval.hamming_distance_matrix(codes_01)
        hamming_matrix = raw_hamming / float(self.code_bits)

        idx = np.triu_indices_from(cosine_matrix, k=1)
        mean_cosine = float(np.mean(cosine_matrix[idx])) if idx[0].size else 1.0
        mean_hamming = float(np.mean(hamming_matrix[idx])) if idx[0].size else 0.0
        spearman = similarity_eval.spearman_correlation_from_matrices(
            cosine_matrix, hamming_matrix
        )
        overlap = similarity_eval.neighbor_overlap_at_k(cosine_matrix, hamming_matrix, k=3)

        return {
            "mean_cosine": mean_cosine,
            "mean_hamming": mean_hamming,
            "cosine_hamming_spearman": float(spearman),
            "topk_overlap_k3": float(overlap),
        }

    def _compute_retrieval_metrics(
        self,
        normalized_embeddings: NDArray[np.float32],
        codes_01: NDArray[np.int8],
        k: int,
    ) -> RetrievalMetrics:
        cosine_topk = retrieval_eval.topk_neighbor_indices_cosine(
            normalized_embeddings,
            k=k,
            seed=self.seed,
            assume_normalized=True,
        )
        hamming_topk = retrieval_eval.topk_neighbor_indices_hamming(
            codes_01,
            k=k,
            seed=self.seed,
        )
        overlap = retrieval_eval.neighbor_overlap_at_k(cosine_topk, hamming_topk)

        n = normalized_embeddings.shape[0]
        relevance = np.zeros((n, n), dtype=np.float32)
        for idx, neighbors in enumerate(cosine_topk):
            relevance[idx, neighbors] = 1.0

        ndcg = retrieval_eval.ndcg_at_k(hamming_topk, relevance, k=k)
        recall = retrieval_eval.recall_at_k(hamming_topk, relevance, k=k)

        key = f"k={k}"
        return {
            "topk_overlap": {key: float(overlap)},
            "ndcg": {key: float(np.mean(ndcg))},
            "recall": {key: float(np.mean(recall))},
        }

    def _compute_classification_metrics(
        self,
        normalized_embeddings: NDArray[np.float32],
        codes_pm1: NDArray[np.float32],
        labels: NDArray[np.int64],
    ) -> Dict[str, float]:
        metrics = classification_eval.evaluate_classification(
            normalized_embeddings,
            codes_pm1,
            labels,
        )
        return metrics.to_dict()

    # -------------------------------------------------------------------- main
    def run(
        self,
        embeddings: ArrayLike,
        *,
        metrics: Iterable[str] = ("similarity", "retrieval"),
        retrieval_k: int = 3,
        classification_labels: Optional[Sequence[int]] = None,
        log_hook: Optional[Callable[[str, Dict[str, object]], None]] = None,
        return_full_code_bits: bool = False,
    ) -> Dict[str, object]:
        """Run the façade and return the in-memory result payload.

        Args:
            embeddings: Float embeddings with shape (N, encoder_dim).
            metrics: Iterable of metric family names to compute.
            retrieval_k: Neighborhood size for retrieval metrics.
            classification_labels: Optional labels for classification metrics.
            log_hook: Optional callback for streaming log events.
            return_full_code_bits: When True, binary_codes retain the full code_bits
                width; otherwise they are truncated to encoder_dim for lightweight
                contract tests.
        """

        metric_list = _ensure_tuple(metrics)
        self._ensure_dataset_capabilities(metric_list)
        embeddings_array = _validate_embeddings(embeddings, self.encoder_spec.embedding_dim)
        if retrieval_k <= 0 or retrieval_k >= embeddings_array.shape[0]:
            raise ValueError("retrieval_k must satisfy 0 < k < number of samples.")
        normalized = self.normalize_embeddings(embeddings_array)

        start = time.perf_counter()
        projected = self.project(normalized)
        codes_pm1, codes_01 = self.binarize(projected)
        bin_time = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        packed = self.pack(codes_01)
        pack_time = (time.perf_counter() - start) * 1000.0

        metrics_block: Dict[str, Optional[Dict[str, float]]] = {
            "similarity": None,
            "retrieval": None,
            "classification": None,
        }

        if "similarity" in metric_list:
            metrics_block["similarity"] = self._compute_similarity_metrics(normalized, codes_01)
        if "retrieval" in metric_list:
            metrics_block["retrieval"] = self._compute_retrieval_metrics(normalized, codes_01, retrieval_k)
        classification = None
        if "classification" in metric_list:
            if classification_labels is None:
                raise ValueError("classification metrics requested but labels are missing.")
            labels_array = np.asarray(classification_labels, dtype=np.int64)
            if labels_array.ndim != 1 or labels_array.shape[0] != embeddings_array.shape[0]:
                raise ValueError("classification_labels must match embeddings length.")
            classification = self._compute_classification_metrics(normalized, codes_pm1, labels_array)
            metrics_block["classification"] = classification

        codes_pm1_view = codes_pm1 if return_full_code_bits else codes_pm1[:, : self.encoder_spec.embedding_dim]
        codes_01_view = codes_01 if return_full_code_bits else codes_01[:, : self.encoder_spec.embedding_dim]

        result: Dict[str, object] = {
            "encoder_name": self.encoder_spec.name,
            "dataset_name": self.dataset_spec.name,
            "code_bits": self.code_bits,
            "projection_type": self.projection_type,
            "projection_seed": self.projection_seed,
            "seed": self.seed,
            "normalize": self.normalize,
            "binary_codes": {
                "pm1": codes_pm1_view,
                "01": codes_01_view,
                "packed": packed if return_full_code_bits else packing_utils.pack_codes(codes_01_view),
            },
            "metrics": metrics_block,
            "metrics_requested": metric_list,
            "similarity_metrics": metrics_block["similarity"],
            "retrieval_metrics": metrics_block["retrieval"],
            "classification_metrics": classification,
            "instrumentation": {
                "binarization_time_ms": float(bin_time),
                "packing_time_ms": float(pack_time),
            },
            "normalization": {"l2": bool(self.normalize)},
        }

        if log_hook is not None:
            log_hook("result", result)
        return result



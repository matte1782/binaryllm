"""Auxiliary classification metrics for BinaryLLM Phase 1 (T7).

This module intentionally keeps the logic simple and deterministic: we expose a
centroid-based linear classifier that operates on both float embeddings and
binary codes (in {-1,+1} form).  The goal is not to maximize accuracy but to
provide a reproducible signal about how binarization affects downstream
classification quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

__all__ = ["ClassificationResult", "evaluate_classification"]


@dataclass(frozen=True)
class ClassificationResult:
    """Container for float vs. binary classification metrics."""

    float_accuracy: float
    float_f1: float
    binary_accuracy: float
    binary_f1: float
    accuracy_delta: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "float_accuracy": float(self.float_accuracy),
            "float_f1": float(self.float_f1),
            "binary_accuracy": float(self.binary_accuracy),
            "binary_f1": float(self.binary_f1),
            "accuracy_delta": float(self.accuracy_delta),
        }


def _validate_inputs(
    float_embeddings: np.ndarray,
    binary_codes_pm1: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    floats = np.asarray(float_embeddings, dtype=np.float32)
    binaries = np.asarray(binary_codes_pm1, dtype=np.float32)
    lbls = np.asarray(labels)

    if floats.ndim != 2:
        raise ValueError("classification: float embeddings must be 2D.")
    if binaries.ndim != 2:
        raise ValueError("classification: binary codes must be 2D.")
    if lbls.ndim != 1:
        raise ValueError("classification: labels must be 1D.")
    if floats.shape[0] != binaries.shape[0] or floats.shape[0] != lbls.shape[0]:
        raise ValueError("classification: number of samples must match between inputs.")
    if not np.isfinite(floats).all() or not np.isfinite(binaries).all():
        raise ValueError("classification: embeddings must be finite.")

    return floats, binaries, lbls


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return array / norms


def _compute_centroids(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    classes = np.unique(labels)
    centroids = []
    for cls in classes:
        cls_features = features[labels == cls]
        if cls_features.size == 0:
            raise ValueError("classification: every class must have at least one sample.")
        centroid = cls_features.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids.append(centroid)
    return classes, np.stack(centroids, axis=0)


def _predict(features: np.ndarray, centroids: np.ndarray, classes: np.ndarray) -> np.ndarray:
    scores = features @ centroids.T
    indices = np.argmax(scores, axis=1)
    return classes[indices]


def _macro_f1(true_labels: np.ndarray, predicted: np.ndarray, classes: Iterable[int]) -> float:
    f1_scores = []
    for cls in classes:
        cls_mask = true_labels == cls
        pred_mask = predicted == cls
        tp = np.logical_and(cls_mask, pred_mask).sum()
        fp = np.logical_and(~cls_mask, pred_mask).sum()
        fn = np.logical_and(cls_mask, ~pred_mask).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    if not f1_scores:
        return 0.0
    return float(np.mean(f1_scores))


def _accuracy(true_labels: np.ndarray, predicted: np.ndarray) -> float:
    return float((true_labels == predicted).mean())


def evaluate_classification(
    float_embeddings: np.ndarray,
    binary_codes_pm1: np.ndarray,
    labels: np.ndarray,
) -> ClassificationResult:
    """Return accuracy/F1 metrics for float vs. binary features."""

    floats, binaries, lbls = _validate_inputs(float_embeddings, binary_codes_pm1, labels)
    floats = _normalize_rows(floats)
    binaries = _normalize_rows(binaries)

    classes, float_centroids = _compute_centroids(floats, lbls)
    _, binary_centroids = _compute_centroids(binaries, lbls)

    float_predictions = _predict(floats, float_centroids, classes)
    binary_predictions = _predict(binaries, binary_centroids, classes)

    float_acc = _accuracy(lbls, float_predictions)
    bin_acc = _accuracy(lbls, binary_predictions)
    float_f1 = _macro_f1(lbls, float_predictions, classes)
    bin_f1 = _macro_f1(lbls, binary_predictions, classes)
    delta = bin_acc - float_acc

    return ClassificationResult(
        float_accuracy=float_acc,
        float_f1=float_f1,
        binary_accuracy=bin_acc,
        binary_f1=bin_f1,
        accuracy_delta=delta,
    )



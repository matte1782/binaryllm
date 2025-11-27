from __future__ import annotations

import numpy as np
import pytest

from src.eval import classification


def _balanced_dataset() -> tuple[np.ndarray, np.ndarray]:
    floats = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    return floats, labels


def _pm1_codes_from(array: np.ndarray) -> np.ndarray:
    signs = np.where(array >= 0, 1.0, -1.0)
    return signs.astype(np.float32, copy=False)


def test_evaluate_classification_is_deterministic_for_same_inputs() -> None:
    rng = np.random.default_rng(123)
    floats = rng.normal(size=(8, 4), loc=0.0, scale=1.0).astype(np.float32)
    binaries = _pm1_codes_from(floats)
    labels = rng.integers(0, 2, size=8, dtype=np.int64)

    first = classification.evaluate_classification(floats, binaries, labels).to_dict()
    second = classification.evaluate_classification(floats, binaries, labels).to_dict()
    assert first == second, "classification metrics must be deterministic for fixed inputs"


def test_binary_metrics_degrade_relative_to_float_embeddings() -> None:
    floats, labels = _balanced_dataset()
    # Binary codes collapse both classes onto the same centroid to induce degradation.
    binaries = np.ones_like(floats, dtype=np.float32)
    result = classification.evaluate_classification(floats, binaries, labels)

    assert result.float_accuracy >= result.binary_accuracy
    assert result.float_f1 >= result.binary_f1
    assert result.accuracy_delta == pytest.approx(result.binary_accuracy - result.float_accuracy)
    assert result.accuracy_delta <= 0.0


def test_evaluate_classification_requires_matching_label_count() -> None:
    floats, labels = _balanced_dataset()
    binaries = _pm1_codes_from(floats)
    with pytest.raises(ValueError, match="samples"):
        classification.evaluate_classification(floats, binaries, labels[:-1])


def test_evaluate_classification_requires_labels() -> None:
    floats, labels = _balanced_dataset()
    binaries = _pm1_codes_from(floats)
    with pytest.raises(ValueError, match="labels"):
        classification.evaluate_classification(floats, binaries, None)  # type: ignore[arg-type]


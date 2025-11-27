"""Embedding batch abstractions for BinaryLLM Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

NORMALIZATION_EPS = 1e-3


def _to_ndarray(data: ArrayLike, field_name: str) -> NDArray[np.float32]:
    try:
        array = np.asarray(data, dtype=np.float32)
    except Exception as exc:  # pragma: no cover - numpy error surface
        raise ValueError(f"stage='embeddings': failed to convert {field_name} to ndarray") from exc
    return array


@dataclass(slots=True)
class FloatEmbeddingBatch:
    """Represents a batch of float embeddings with validation."""

    data: NDArray[np.float32]
    embedding_dim: int
    encoder_name: str
    dataset_name: str
    ids: Optional[Sequence[str]] = None
    normalized: bool = False
    _data: NDArray[np.float32] = field(init=False, repr=False)
    _ids: Optional[List[str]] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._data = _to_ndarray(self.data, "data")
        if self._data.ndim != 2:
            raise ValueError("embeddings data must be 2D (stage='embeddings').")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive for embeddings (stage='embeddings').")
        if self._data.shape[1] != self.embedding_dim:
            raise ValueError(
                "embedding_dim mismatch for embeddings: data second dimension must equal embedding_dim."
            )

        if self.ids is not None:
            if len(self.ids) != self._data.shape[0]:
                raise ValueError("ids length must match number of embeddings (stage='embeddings').")
            self._ids = list(self.ids)
        else:
            self._ids = None

        if self.normalized:
            norms = np.linalg.norm(self._data, axis=1)
            if norms.size and not np.allclose(norms, 1.0, atol=NORMALIZATION_EPS):
                raise ValueError(
                    "normalized embeddings must have L2 norm within tolerance (stage='embeddings')."
                )

        self.data = self._data
        if self._ids is not None:
            self.ids = self._ids


def _to_int_array(data: ArrayLike, field_name: str) -> NDArray[np.int8]:
    try:
        array = np.asarray(data, dtype=np.int8)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"stage='embeddings': failed to convert {field_name} to ndarray") from exc
    return array


@dataclass(slots=True)
class BinaryCodeBatch:
    """Represents logical binary codes (no packing)."""

    codes_pm1: NDArray[np.int8]
    codes_01: NDArray[np.int8]
    code_bits: int
    encoder_name: str
    dataset_name: str
    _codes_pm1: NDArray[np.int8] = field(init=False, repr=False)
    _codes_01: NDArray[np.int8] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._codes_pm1 = _to_int_array(self.codes_pm1, "codes_pm1")
        self._codes_01 = _to_int_array(self.codes_01, "codes_01")

        if self._codes_pm1.ndim != 2:
            raise ValueError("stage='embeddings': codes_pm1 must be 2D.")
        if self._codes_01.ndim != 2:
            raise ValueError("stage='embeddings': codes_01 must be 2D.")
        if self._codes_pm1.shape != self._codes_01.shape:
            raise ValueError("stage='embeddings': codes_pm1 and codes_01 must share shape.")

        _, bits = self._codes_pm1.shape
        if self.code_bits != bits:
            raise ValueError("stage='embeddings': code_bits must match codes width.")

        if not np.isin(self._codes_pm1, (-1, 1)).all():
            raise ValueError("stage='embeddings': codes_pm1 must contain only {-1,+1}.")
        if not np.isin(self._codes_01, (0, 1)).all():
            raise ValueError("stage='embeddings': codes_01 must contain only {0,1}.")

        self.codes_pm1 = self._codes_pm1
        self.codes_01 = self._codes_01


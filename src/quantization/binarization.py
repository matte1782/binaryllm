"""Binarization and random projection utilities for BinaryLLM Phase 1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def binarize_sign(x: np.ndarray) -> np.ndarray:
    """Return elementwise sign binarization with convention x >= 0 → +1."""
    result = np.ones_like(x, dtype=np.float32)
    result[x < 0] = -1.0
    return result


@dataclass(slots=True)
class RandomProjection:
    """Deterministic random projection using Gaussian weights."""

    input_dim: int
    output_bits: int
    seed: int
    _weights: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.output_bits <= 0:
            raise ValueError("projection dims must be positive.")
        rng = np.random.default_rng(self.seed)
        self._weights = rng.standard_normal((self.input_dim, self.output_bits), dtype=np.float32)

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project inputs onto random hyperplanes."""
        if x.shape[-1] != self.input_dim:
            raise ValueError("projection dim mismatch (stage='projection').")
        if not np.isfinite(x).all():
            raise ValueError("projection received non-finite inputs.")
        x_float = np.asarray(x, dtype=np.float32)
        return x_float @ self._weights

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.project(x)


def project_and_binarize(x: np.ndarray, projection: RandomProjection) -> np.ndarray:
    """Apply projection then binarize with sign convention."""
    projected = projection.project(x)
    return binarize_sign(projected)


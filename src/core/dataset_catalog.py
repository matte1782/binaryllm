"""Dataset and encoder catalog for BinaryLLM Phase 1.

This module centralizes lightweight metadata about supported datasets and
encoders so that later IO/config logic can validate requests deterministically.
The catalog is static for Phase 1 but intentionally structured for easy
extension in future phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

__all__ = [
    "DatasetSpec",
    "EncoderSpec",
    "DATASET_REGISTRY",
    "ENCODER_REGISTRY",
    "UnknownDatasetError",
    "UnknownEncoderError",
    "get_dataset_spec",
    "get_encoder_spec",
]


@dataclass(frozen=True)
class DatasetSpec:
    """Describes a dataset containing precomputed embeddings."""

    name: str
    file_format: str
    embedding_dim: int
    requires_l2_normalization: bool
    required_fields: Tuple[str, ...]
    supports_similarity: bool
    supports_retrieval: bool
    supports_classification: bool
    default_encoder: str | None = None


@dataclass(frozen=True)
class EncoderSpec:
    """Describes a precomputed embedding encoder."""

    name: str
    display_name: str
    embedding_dim: int
    dtype: str
    notes: str | None = None


class UnknownDatasetError(KeyError):
    """Raised when a requested dataset is not found in the registry."""


class UnknownEncoderError(KeyError):
    """Raised when a requested encoder is not found in the registry."""


ENCODER_REGISTRY: Dict[str, EncoderSpec] = {
    "synthetic_encoder_4d": EncoderSpec(
        name="synthetic_encoder_4d",
        display_name="Synthetic Test Encoder (4d)",
        embedding_dim=4,
        dtype="float32",
        notes="Used by tests/data/phase1_synthetic fixtures.",
    ),
    "demo_mini_lm_384": EncoderSpec(
        name="demo_mini_lm_384",
        display_name="Demo MiniLM Encoder (384d)",
        embedding_dim=384,
        dtype="float32",
        notes="Placeholder for realistic MiniLM/BGE-style encoders.",
    ),
}


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "phase1_synthetic_toy": DatasetSpec(
        name="phase1_synthetic_toy",
        file_format="npy",
        embedding_dim=4,
        requires_l2_normalization=True,
        required_fields=("id", "embedding", "label"),
        supports_similarity=True,
        supports_retrieval=True,
        supports_classification=True,
        default_encoder="synthetic_encoder_4d",
    ),
    "demo_beir_msmarco": DatasetSpec(
        name="demo_beir_msmarco",
        file_format="parquet",
        embedding_dim=384,
        requires_l2_normalization=True,
        required_fields=("id", "embedding", "text", "split"),
        supports_similarity=True,
        supports_retrieval=True,
        supports_classification=False,
        default_encoder="demo_mini_lm_384",
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return the DatasetSpec for ``name`` or raise UnknownDatasetError."""
    try:
        spec = DATASET_REGISTRY[name]
    except KeyError as exc:
        raise UnknownDatasetError(f"Unknown dataset '{name}'.") from exc
    _validate_dataset_spec(spec)
    return spec


def get_encoder_spec(name: str) -> EncoderSpec:
    """Return the EncoderSpec for ``name`` or raise UnknownEncoderError."""
    try:
        spec = ENCODER_REGISTRY[name]
    except KeyError as exc:
        raise UnknownEncoderError(f"Unknown encoder '{name}'.") from exc
    _validate_encoder_spec(spec)
    return spec


def _validate_dataset_spec(spec: DatasetSpec) -> None:
    """Internal sanity checks to catch catalog inconsistencies early."""
    if spec.embedding_dim <= 0:
        raise ValueError(f"Dataset '{spec.name}' has non-positive embedding_dim.")
    if spec.default_encoder is not None and spec.default_encoder not in ENCODER_REGISTRY:
        raise ValueError(
            f"Dataset '{spec.name}' references unknown encoder '{spec.default_encoder}'."
        )


def _validate_encoder_spec(spec: EncoderSpec) -> None:
    """Internal sanity checks for encoders."""
    if spec.embedding_dim <= 0:
        raise ValueError(f"Encoder '{spec.name}' has non-positive embedding_dim.")
    if not spec.dtype:
        raise ValueError(f"Encoder '{spec.name}' must define dtype.")



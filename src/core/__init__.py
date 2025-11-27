"""
Core abstractions for BinaryLLM Phase 1.

This module provides:
    - DatasetSpec, EncoderSpec: Metadata specifications for datasets and encoders
    - FloatEmbeddingBatch, BinaryCodeBatch: Validated embedding containers
    - EmbeddingDataset: Dataset wrapper with catalog integration
    - Dataset catalog with registration and lookup functions

Example:
    >>> from src.core.dataset_catalog import get_dataset_spec, get_encoder_spec
    >>> dataset = get_dataset_spec("phase1_synthetic_toy")
    >>> encoder = get_encoder_spec("synthetic_encoder_4d")
"""

from src.core.dataset_catalog import (
    DatasetSpec,
    EncoderSpec,
    get_dataset_spec,
    get_encoder_spec,
    UnknownDatasetError,
    UnknownEncoderError,
)
from src.core.embeddings import FloatEmbeddingBatch, BinaryCodeBatch
from src.core.datasets import EmbeddingDataset

__all__ = [
    "DatasetSpec",
    "EncoderSpec",
    "get_dataset_spec",
    "get_encoder_spec",
    "UnknownDatasetError",
    "UnknownEncoderError",
    "FloatEmbeddingBatch",
    "BinaryCodeBatch",
    "EmbeddingDataset",
]

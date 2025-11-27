"""Contract tests for src.core.datasets abstractions.

Assumptions:
- EmbeddingDataset consumes FloatEmbeddingBatch instances and catalog specs.
- Validation errors must reference both the offending field (embedding_dim,
  normalized, labels, etc.) and the embeddings stage for debuggability.
"""

import numpy as np
import pytest

from src.core.dataset_catalog import DATASET_REGISTRY, ENCODER_REGISTRY
from src.core.embeddings import FloatEmbeddingBatch
from src.core import datasets


def _make_float_batch(dim: int, normalized: bool = True, size: int = 2) -> FloatEmbeddingBatch:
    data = np.eye(dim, dtype=np.float32)[:size]
    if normalized:
        return FloatEmbeddingBatch(
            data=data,
            embedding_dim=dim,
            encoder_name="synthetic_encoder_4d",
            dataset_name="phase1_synthetic_toy",
            normalized=True,
        )
    return FloatEmbeddingBatch(
        data=data * 2,
        embedding_dim=dim,
        encoder_name="synthetic_encoder_4d",
        dataset_name="phase1_synthetic_toy",
        normalized=False,
    )


def test_embedding_dataset_success_with_defaults():
    spec = DATASET_REGISTRY["phase1_synthetic_toy"]
    batch = _make_float_batch(dim=4, normalized=True)
    ds = datasets.EmbeddingDataset(
        name=spec.name,
        encoder_name=None,
        embeddings=batch,
        labels=["pos", "neg"],
    )
    assert ds.name == spec.name
    assert ds.encoder_name == spec.default_encoder
    assert ds.embeddings is batch
    assert ds.labels == ["pos", "neg"]


def test_embedding_dataset_unknown_dataset_raises():
    batch = _make_float_batch(dim=4)
    with pytest.raises(ValueError, match="dataset.*unknown_dataset"):
        datasets.EmbeddingDataset(
            name="unknown_dataset",
            encoder_name="synthetic_encoder_4d",
            embeddings=batch,
        )


def test_embedding_dataset_unknown_encoder_raises():
    spec = DATASET_REGISTRY["phase1_synthetic_toy"]
    batch = _make_float_batch(dim=4)
    with pytest.raises(ValueError, match="encoder.*missing_encoder"):
        datasets.EmbeddingDataset(
            name=spec.name,
            encoder_name="missing_encoder",
            embeddings=batch,
        )


def test_embedding_dataset_dim_mismatch_raises():
    spec = DATASET_REGISTRY["phase1_synthetic_toy"]
    # create embeddings with dim != spec.embedding_dim
    batch = _make_float_batch(dim=3)
    with pytest.raises(ValueError, match="embedding_dim.*embeddings"):
        datasets.EmbeddingDataset(
            name=spec.name,
            encoder_name=spec.default_encoder,
            embeddings=batch,
        )


def test_embedding_dataset_requires_normalized_embeddings():
    spec = DATASET_REGISTRY["phase1_synthetic_toy"]
    batch = _make_float_batch(dim=4, normalized=False)
    with pytest.raises(ValueError, match="normalized.*embeddings"):
        datasets.EmbeddingDataset(
            name=spec.name,
            encoder_name=spec.default_encoder,
            embeddings=batch,
        )


def test_embedding_dataset_labels_length_must_match():
    spec = DATASET_REGISTRY["phase1_synthetic_toy"]
    batch = _make_float_batch(dim=4)
    with pytest.raises(ValueError, match="labels.*embeddings"):
        datasets.EmbeddingDataset(
            name=spec.name,
            encoder_name=spec.default_encoder,
            embeddings=batch,
            labels=["only_one"],
        )


def test_embedding_dataset_labels_require_classification_support():
    spec = DATASET_REGISTRY["demo_beir_msmarco"]
    batch = _make_float_batch(dim=spec.embedding_dim)
    with pytest.raises(ValueError, match="classification.*demo_beir_msmarco"):
        datasets.EmbeddingDataset(
            name=spec.name,
            encoder_name=spec.default_encoder,
            embeddings=batch,
            labels=["irrelevant"] * batch.data.shape[0],
        )


def test_embedding_dataset_split_length_must_match():
    spec = DATASET_REGISTRY["phase1_synthetic_toy"]
    batch = _make_float_batch(dim=4)
    with pytest.raises(ValueError, match="split.*embeddings"):
        datasets.EmbeddingDataset(
            name=spec.name,
            encoder_name=spec.default_encoder,
            embeddings=batch,
            split=["train"],
        )


def test_embedding_dataset_is_deterministic():
    spec = DATASET_REGISTRY["phase1_synthetic_toy"]
    batch = _make_float_batch(dim=4)
    ds_a = datasets.EmbeddingDataset(
        name=spec.name,
        encoder_name=spec.default_encoder,
        embeddings=batch,
    )
    ds_b = datasets.EmbeddingDataset(
        name=spec.name,
        encoder_name=spec.default_encoder,
        embeddings=batch,
    )
    assert ds_a.name == ds_b.name
    assert ds_a.encoder_name == ds_b.encoder_name
    assert np.array_equal(ds_a.embeddings.data, ds_b.embeddings.data)


def test_query_dataset_success():
    spec = DATASET_REGISTRY["phase1_synthetic_toy"]
    queries = _make_float_batch(dim=spec.embedding_dim)
    qset = datasets.QueryDataset(
        queries=queries,
        target_dataset_name=spec.name,
    )
    assert qset.target_dataset_name == spec.name
    assert qset.queries is queries


def test_query_dataset_unknown_target_raises():
    queries = _make_float_batch(dim=4)
    with pytest.raises(ValueError, match="target_dataset.*unknown_dataset"):
        datasets.QueryDataset(
            queries=queries,
            target_dataset_name="unknown_dataset",
        )


# These tests define the Phase 1 EmbeddingDataset / QueryDataset contract:
# - datasets must align with catalog specs (dim, normalization, encoder)
# - labels/splits must match lengths and supported task flags
# - error messages must surface both the offending field and embeddings stage


"""Tests for src.core.dataset_catalog metadata invariants."""

import pytest

from src.core import dataset_catalog


def test_get_dataset_spec_known_dataset_returns_expected_values():
    spec = dataset_catalog.get_dataset_spec("phase1_synthetic_toy")
    assert spec.file_format == "npy"
    assert spec.embedding_dim == 4
    assert spec.requires_l2_normalization is True
    assert spec.required_fields == ("id", "embedding", "label")
    assert spec.supports_similarity is True
    assert spec.supports_retrieval is True
    assert spec.supports_classification is True
    assert spec.default_encoder == "synthetic_encoder_4d"


def test_get_encoder_spec_known_encoder_returns_expected_values():
    spec = dataset_catalog.get_encoder_spec("demo_mini_lm_384")
    assert spec.display_name.startswith("Demo MiniLM")
    assert spec.embedding_dim == 384
    assert spec.dtype == "float32"


def test_get_dataset_spec_unknown_name_raises():
    with pytest.raises(dataset_catalog.UnknownDatasetError):
        dataset_catalog.get_dataset_spec("non_existing_dataset")


def test_get_encoder_spec_unknown_name_raises():
    with pytest.raises(dataset_catalog.UnknownEncoderError):
        dataset_catalog.get_encoder_spec("non_existing_encoder")


def test_get_dataset_spec_is_deterministic():
    first = dataset_catalog.get_dataset_spec("phase1_synthetic_toy")
    second = dataset_catalog.get_dataset_spec("phase1_synthetic_toy")
    assert first is second


def test_get_encoder_spec_is_deterministic():
    first = dataset_catalog.get_encoder_spec("demo_mini_lm_384")
    second = dataset_catalog.get_encoder_spec("demo_mini_lm_384")
    assert first is second


def test_dataset_without_default_encoder_is_allowed(monkeypatch):
    spec = dataset_catalog.DatasetSpec(
        name="no_encoder_dataset",
        file_format="npy",
        embedding_dim=2,
        requires_l2_normalization=False,
        required_fields=("id", "embedding"),
        supports_similarity=True,
        supports_retrieval=False,
        supports_classification=False,
        default_encoder=None,
    )
    monkeypatch.setitem(dataset_catalog.DATASET_REGISTRY, "no_encoder_dataset", spec)
    assert dataset_catalog.get_dataset_spec("no_encoder_dataset") is spec


def test_dataset_default_encoder_must_exist_in_registry(monkeypatch):
    spec = dataset_catalog.DatasetSpec(
        name="bad_dataset",
        file_format="npy",
        embedding_dim=2,
        requires_l2_normalization=False,
        required_fields=("id", "embedding"),
        supports_similarity=True,
        supports_retrieval=False,
        supports_classification=False,
        default_encoder="missing_encoder",
    )
    monkeypatch.setitem(dataset_catalog.DATASET_REGISTRY, "bad_dataset", spec)
    with pytest.raises(ValueError, match="references unknown encoder"):
        dataset_catalog.get_dataset_spec("bad_dataset")


def test_dataset_embedding_dim_must_be_positive(monkeypatch):
    spec = dataset_catalog.DatasetSpec(
        name="bad_dim_dataset",
        file_format="npy",
        embedding_dim=0,
        requires_l2_normalization=False,
        required_fields=("id", "embedding"),
        supports_similarity=True,
        supports_retrieval=False,
        supports_classification=False,
        default_encoder=None,
    )
    monkeypatch.setitem(dataset_catalog.DATASET_REGISTRY, "bad_dim_dataset", spec)
    with pytest.raises(ValueError, match="non-positive embedding_dim"):
        dataset_catalog.get_dataset_spec("bad_dim_dataset")


def test_encoder_embedding_dim_must_be_positive(monkeypatch):
    spec = dataset_catalog.EncoderSpec(
        name="bad_encoder",
        display_name="Bad Encoder",
        embedding_dim=0,
        dtype="float32",
    )
    monkeypatch.setitem(dataset_catalog.ENCODER_REGISTRY, "bad_encoder", spec)
    with pytest.raises(ValueError, match="non-positive embedding_dim"):
        dataset_catalog.get_encoder_spec("bad_encoder")


def test_encoder_dtype_must_be_non_empty(monkeypatch):
    spec = dataset_catalog.EncoderSpec(
        name="empty_dtype_encoder",
        display_name="Empty DType Encoder",
        embedding_dim=8,
        dtype="",
    )
    monkeypatch.setitem(dataset_catalog.ENCODER_REGISTRY, "empty_dtype_encoder", spec)
    with pytest.raises(ValueError, match="must define dtype"):
        dataset_catalog.get_encoder_spec("empty_dtype_encoder")


def test_phase1_synthetic_toy_matches_h1_requirements():
    spec = dataset_catalog.get_dataset_spec("phase1_synthetic_toy")
    assert spec.requires_l2_normalization is True
    assert spec.supports_similarity is True
    assert spec.supports_retrieval is True
    assert spec.supports_classification is True



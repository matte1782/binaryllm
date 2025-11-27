"""Phase 1 – T10 Binary Embedding Engine contract tests.

These pytest cases define the façade contract for
`src/variants/binary_embedding_engine.py`. They focus on constructor validation,
determinism, normalization behavior, projection/binarization/packing invariants,
metrics correctness (similarity, retrieval, optional classification), logging
hooks, and the returned in-memory result schema. The actual engine must satisfy
ALL of these behaviors; no production code is edited here.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pytest

from src.core.dataset_catalog import DatasetSpec, EncoderSpec

# The façade is not implemented yet; skip these tests until it exists, but once
# implemented they become binding.
engine_mod = pytest.importorskip("src.variants.binary_embedding_engine")
BinaryEmbeddingEngine = engine_mod.BinaryEmbeddingEngine


def _make_embeddings(num_samples: int = 12, normalized: bool = False) -> np.ndarray:
    """Deterministic synthetic embeddings with controllable normalization."""
    base = np.linspace(-1.0, 1.0, num_samples * 4, dtype=np.float32).reshape(num_samples, 4)
    # Make rows unique/non-orthogonal by adding a tiny offset per sample.
    base += np.arange(num_samples, dtype=np.float32)[:, None] * 0.01
    if normalized:
        norms = np.linalg.norm(base, axis=1, keepdims=True)
        return base / norms
    return base


def _classification_labels(num_samples: int) -> np.ndarray:
    labels = np.arange(num_samples) % 3
    return labels.astype(np.int64)


@pytest.fixture(scope="module")
def encoder_spec() -> EncoderSpec:
    return EncoderSpec(
        name="synthetic_encoder_4d",
        display_name="Synthetic Test Encoder",
        embedding_dim=4,
        dtype="float32",
    )


@pytest.fixture(scope="module")
def dataset_spec(encoder_spec: EncoderSpec) -> DatasetSpec:
    return DatasetSpec(
        name="phase1_synthetic_contract",
        file_format="npy",
        embedding_dim=encoder_spec.embedding_dim,
        requires_l2_normalization=True,
        required_fields=("id", "embedding", "label"),
        supports_similarity=True,
        supports_retrieval=True,
        supports_classification=True,
        default_encoder=encoder_spec.name,
    )


@pytest.fixture
def engine_args(encoder_spec: EncoderSpec, dataset_spec: DatasetSpec) -> Dict[str, Any]:
    return {
        "encoder_spec": encoder_spec,
        "dataset_spec": dataset_spec,
        "code_bits": 32,
        "projection_type": "gaussian",
        "seed": 4242,
        "normalize": True,
    }


@pytest.fixture
def raw_embeddings() -> np.ndarray:
    return _make_embeddings(normalized=False)


@pytest.fixture
def normalized_embeddings() -> np.ndarray:
    return _make_embeddings(normalized=True)


def test_constructor_requires_all_fields(engine_args: Dict[str, Any]) -> None:
    required = ("encoder_spec", "dataset_spec", "code_bits", "projection_type", "seed")
    for missing in required:
        params = dict(engine_args)
        params.pop(missing)
        with pytest.raises(ValueError, match=missing):
            BinaryEmbeddingEngine(**params)


def test_constructor_validates_positive_code_bits(engine_args: Dict[str, Any]) -> None:
    params = dict(engine_args)
    params["code_bits"] = 0
    with pytest.raises(ValueError, match=r"code_bits.*positive"):
        BinaryEmbeddingEngine(**params)


def test_run_returns_complete_schema(engine_args: Dict[str, Any], raw_embeddings: np.ndarray) -> None:
    engine = BinaryEmbeddingEngine(**engine_args)
    result = engine.run(raw_embeddings, metrics=("similarity", "retrieval"))

    for key in ("encoder_name", "dataset_name", "code_bits", "projection_type", "seed", "normalize"):
        assert key in result, f"{key} missing in result"
    assert result["encoder_name"] == engine_args["encoder_spec"].name
    assert result["dataset_name"] == engine_args["dataset_spec"].name
    assert result["code_bits"] == engine_args["code_bits"]
    assert result["projection_type"] == engine_args["projection_type"]
    assert result["seed"] == engine_args["seed"]
    assert result["normalize"] is True

    metrics = result["metrics"]
    assert set(metrics) == {"similarity", "retrieval", "classification"}
    assert metrics["classification"] is None

    similarity = metrics["similarity"]
    for field in ("mean_cosine", "mean_hamming", "cosine_hamming_spearman"):
        assert field in similarity and isinstance(similarity[field], float)
        assert -1.0 <= similarity[field] <= 1.0

    retrieval = metrics["retrieval"]
    for bucket in ("topk_overlap", "ndcg", "recall"):
        assert "k=3" in retrieval[bucket]
        assert isinstance(retrieval[bucket]["k=3"], float)

    binary_codes = result["binary_codes"]
    assert set(binary_codes) == {"pm1", "01", "packed"}
    assert binary_codes["pm1"].shape == raw_embeddings.shape
    assert binary_codes["01"].shape == raw_embeddings.shape
    assert binary_codes["pm1"].dtype == np.float32
    assert np.issubdtype(binary_codes["01"].dtype, np.integer)
    assert np.all(np.isin(binary_codes["01"], np.array([0, 1], dtype=np.int8)))
    packed = binary_codes["packed"]
    assert packed.dtype == np.uint64
    assert packed.shape[0] == raw_embeddings.shape[0]
    expected_words = math.ceil(engine_args["code_bits"] / 64)
    assert packed.shape[1] == expected_words


def test_run_is_deterministic_for_same_seed(
    engine_args: Dict[str, Any], normalized_embeddings: np.ndarray
) -> None:
    engine_a = BinaryEmbeddingEngine(**engine_args)
    engine_b = BinaryEmbeddingEngine(**engine_args)
    result_a = engine_a.run(normalized_embeddings, metrics=("similarity", "retrieval"))
    result_b = engine_b.run(normalized_embeddings, metrics=("similarity", "retrieval"))
    assert result_a["metrics"] == result_b["metrics"]
    assert np.array_equal(result_a["binary_codes"]["pm1"], result_b["binary_codes"]["pm1"])
    assert np.array_equal(result_a["binary_codes"]["01"], result_b["binary_codes"]["01"])
    assert np.array_equal(result_a["binary_codes"]["packed"], result_b["binary_codes"]["packed"])


def test_projection_changes_with_different_seed(
    engine_args: Dict[str, Any], normalized_embeddings: np.ndarray
) -> None:
    engine_a = BinaryEmbeddingEngine(**engine_args)
    params_b = dict(engine_args)
    params_b["seed"] = engine_args["seed"] + 777
    engine_b = BinaryEmbeddingEngine(**params_b)
    proj_a = engine_a.project(normalized_embeddings)
    proj_b = engine_b.project(normalized_embeddings)
    assert proj_a.shape == proj_b.shape == (normalized_embeddings.shape[0], engine_args["code_bits"])
    assert not np.allclose(proj_a, proj_b)


def test_normalization_flag_controls_behavior(
    engine_args: Dict[str, Any], raw_embeddings: np.ndarray
) -> None:
    params = dict(engine_args)
    params["normalize"] = True
    engine_norm = BinaryEmbeddingEngine(**params)
    normalized = engine_norm.normalize_embeddings(raw_embeddings)
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0, atol=1e-3)

    params["normalize"] = False
    engine_passthrough = BinaryEmbeddingEngine(**params)
    passthrough = engine_passthrough.normalize_embeddings(raw_embeddings)
    assert np.allclose(passthrough, raw_embeddings)


def test_projection_type_validation(engine_args: Dict[str, Any]) -> None:
    params = dict(engine_args)
    params["projection_type"] = "unsupported_projection"
    with pytest.raises(ValueError, match=r"projection_type"):
        BinaryEmbeddingEngine(**params)


def test_binarize_outputs_expected_values(
    engine_args: Dict[str, Any], normalized_embeddings: np.ndarray
) -> None:
    engine = BinaryEmbeddingEngine(**engine_args)
    projected = engine.project(normalized_embeddings)
    codes_pm1, codes_01 = engine.binarize(projected)
    assert codes_pm1.shape == projected.shape
    assert codes_01.shape == projected.shape
    assert np.all(np.isin(codes_pm1, np.array([-1.0, 1.0], dtype=np.float32)))
    assert np.all(np.isin(codes_01, np.array([0, 1], dtype=np.uint8)))


def test_packing_round_trip_and_word_count(
    engine_args: Dict[str, Any], normalized_embeddings: np.ndarray
) -> None:
    engine = BinaryEmbeddingEngine(**engine_args)
    projected = engine.project(normalized_embeddings)
    _, codes_01 = engine.binarize(projected)
    packed = engine.pack(codes_01)
    unpacked = engine.unpack(packed)
    assert np.array_equal(unpacked[:, : engine_args["code_bits"]], codes_01)
    assert packed.shape[1] == math.ceil(engine_args["code_bits"] / 64)


def test_similarity_and_retrieval_metrics_structure(
    engine_args: Dict[str, Any], normalized_embeddings: np.ndarray
) -> None:
    engine = BinaryEmbeddingEngine(**engine_args)
    result = engine.run(normalized_embeddings, retrieval_k=3, metrics=("similarity", "retrieval"))
    similarity = result["metrics"]["similarity"]
    retrieval = result["metrics"]["retrieval"]
    assert similarity["cosine_hamming_spearman"] <= 1.0
    assert "k=3" in retrieval["topk_overlap"]
    assert "k=3" in retrieval["ndcg"]
    assert "k=3" in retrieval["recall"]


def test_classification_metrics_returned_when_requested(
    engine_args: Dict[str, Any], raw_embeddings: np.ndarray
) -> None:
    labels = _classification_labels(raw_embeddings.shape[0])
    engine = BinaryEmbeddingEngine(**engine_args)
    requested_metrics = ("similarity", "retrieval", "classification")
    result = engine.run(
        raw_embeddings,
        metrics=requested_metrics,
        classification_labels=labels,
        retrieval_k=3,
    )
    classification = result["metrics"]["classification"]
    assert classification is not None, "classification metrics must be present when requested"
    for key in ("float_accuracy", "binary_accuracy", "float_f1", "binary_f1", "accuracy_delta"):
        assert key in classification
        assert isinstance(classification[key], float)
    assert -1.0 <= classification["accuracy_delta"] <= 1.0


def test_log_hook_receives_final_payload(
    engine_args: Dict[str, Any], normalized_embeddings: np.ndarray
) -> None:
    events: List[Dict[str, Any]] = []

    def hook(stage: str, payload: Dict[str, Any]) -> None:
        events.append({"stage": stage, "payload": payload})

    engine = BinaryEmbeddingEngine(**engine_args)
    result = engine.run(
        normalized_embeddings,
        metrics=("similarity", "retrieval"),
        retrieval_k=3,
        log_hook=hook,
    )
    assert events, "log hook must be invoked at least once"
    assert events[-1]["stage"] == "result"
    assert events[-1]["payload"] is result


def test_run_rejects_nan_inputs(engine_args: Dict[str, Any], raw_embeddings: np.ndarray) -> None:
    embeddings = raw_embeddings.copy()
    embeddings[0, 0] = np.nan
    engine = BinaryEmbeddingEngine(**engine_args)
    with pytest.raises(ValueError, match=r"finite"):
        engine.run(embeddings)


def test_run_validates_embedding_dimension(engine_args: Dict[str, Any]) -> None:
    bad_embeddings = np.ones((5, 5), dtype=np.float32)
    engine = BinaryEmbeddingEngine(**engine_args)
    with pytest.raises(ValueError, match=r"embedding_dim"):
        engine.run(bad_embeddings)


def test_metrics_subset_controls_outputs(
    engine_args: Dict[str, Any], normalized_embeddings: np.ndarray
) -> None:
    engine = BinaryEmbeddingEngine(**engine_args)
    result = engine.run(normalized_embeddings, metrics=("similarity",))
    assert result["metrics"]["retrieval"] is None
    assert result["metrics"]["classification"] is None
    assert isinstance(result["metrics"]["similarity"]["mean_cosine"], float)


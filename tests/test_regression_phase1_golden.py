"""Phase 1 synthetic golden regression tests (T13 v2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import yaml

from src.core import dataset_catalog
from src.core.dataset_catalog import DatasetSpec

runner = pytest.importorskip("src.experiments.runners.phase1_binary_embeddings")

BASE_DIR = Path(__file__).resolve().parents[1]
GOLDEN_DIR = BASE_DIR / "tests" / "data" / "phase1_golden"
CONFIG_PATH = GOLDEN_DIR / "config_phase1_synthetic_v1.yaml"
GOLDEN_RESULT_PATH = GOLDEN_DIR / "golden_result_phase1_synthetic_v1.json"
GOLDEN_LOG_PATH = GOLDEN_DIR / "golden_log_phase1_synthetic_v1.jsonl"
DATASET_NAME = "phase1_synthetic_golden"
ENCODER_NAME = "synthetic_encoder_4d"
# Phase 1 golden suite must freeze all three metric families simultaneously.
EXPECTED_METRICS = ("similarity", "retrieval", "classification")
CLASSIFICATION_KEYS = frozenset(
    ["float_accuracy", "binary_accuracy", "float_f1", "binary_f1", "accuracy_delta"]
)


def _split_major_minor(version: str) -> tuple[str, str]:
    parts = version.split(".")
    if len(parts) == 1:
        parts.append("0")
    return parts[0], parts[1]


def _assert_system_metadata_close(actual: Dict[str, Any], expected: Dict[str, Any]) -> None:
    for key in ("hostname", "platform"):
        assert isinstance(actual.get(key), str) and actual.get(key)
        assert isinstance(expected.get(key), str) and expected.get(key)
    actual_version = actual.get("python_version", "")
    expected_version = expected.get("python_version", "")
    assert _split_major_minor(actual_version) == _split_major_minor(expected_version)
    for key in ("cpu_model", "gpu_model", "num_gpus"):
        assert key in actual, f"{key} missing from system metadata"
        assert actual.get(key) == expected.get(key), f"{key} mismatch in system metadata"


def _assert_classification_degradation(metrics: Dict[str, float]) -> None:
    for key in CLASSIFICATION_KEYS:
        assert key in metrics, f"{key} missing from classification metrics"
    float_acc = metrics["float_accuracy"]
    binary_acc = metrics["binary_accuracy"]
    delta = metrics["accuracy_delta"]
    assert 0.0 <= float_acc <= 1.0
    assert 0.0 <= binary_acc <= 1.0
    assert float_acc >= binary_acc, "float accuracy must dominate binary accuracy"
    assert delta < 0.0, "accuracy_delta must record degradation (binary - float < 0)"
    assert delta == pytest.approx(binary_acc - float_acc, abs=1e-9)


def _load_yaml_config() -> Dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _load_golden_result() -> Dict[str, Any]:
    return json.loads(GOLDEN_RESULT_PATH.read_text(encoding="utf-8"))


def _load_golden_log() -> Dict[str, Any]:
    raw_line = GOLDEN_LOG_PATH.read_text(encoding="utf-8").strip()
    return json.loads(raw_line)


def _materialize_config(tmp_path: Path) -> Tuple[Path, Path]:
    config_data = _load_yaml_config()
    assert "classification_labels" in config_data, "golden config must include classification labels"
    config_data["embedding_files"] = [
        (BASE_DIR / Path(path)).resolve().as_posix() for path in config_data["embedding_files"]
    ]
    assert all("/" in entry and "\\" not in entry for entry in config_data["embedding_files"])
    config_data["classification_labels"] = (
        (BASE_DIR / Path(config_data["classification_labels"])).resolve().as_posix()
    )
    output_dir = (tmp_path / "phase1_runner_artifacts").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_data["output_dir"] = output_dir.as_posix()
    config_path = tmp_path / "phase1_golden_config.json"
    config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
    return config_path, output_dir


def _load_latest_log(output_dir: Path) -> Dict[str, Any]:
    log_files = sorted(output_dir.glob("*.json"))
    assert log_files, "runner must emit a JSON log entry"
    return json.loads(log_files[-1].read_text(encoding="utf-8"))


def _assert_log_binary_codes_sanitized(binary_codes: Dict[str, Any]) -> None:
    pm1 = binary_codes["pm1"]
    codes_01 = binary_codes["01"]
    packed = binary_codes["packed"]
    assert isinstance(pm1, list)
    assert isinstance(codes_01, list)
    assert isinstance(packed, list)
    assert len(pm1) == len(codes_01) == len(packed)
    for row in pm1:
        assert isinstance(row, list)
        assert all(isinstance(value, float) for value in row)
    for row in codes_01:
        assert isinstance(row, list)
        assert all(isinstance(value, int) for value in row)
    for words in packed:
        assert isinstance(words, list)
        assert all(isinstance(word, int) for word in words)


def _assert_metric_tree_close(
    actual: Dict[str, Any], expected: Dict[str, Any], *, atol: float = 1e-9, rtol: float = 1e-9
) -> None:
    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            assert isinstance(actual_value, dict)
            _assert_metric_tree_close(actual_value, expected_value, atol=atol, rtol=rtol)
        else:
            assert np.isclose(actual_value, expected_value, atol=atol, rtol=rtol), key


def _assert_instrumentation_close(
    actual: Dict[str, float], expected: Dict[str, float], tolerance_ms: float = 5.0
) -> None:
    assert actual.keys() == expected.keys()
    for key in actual:
        assert abs(actual[key] - expected[key]) <= tolerance_ms


def _assert_binary_code_shapes(binary_codes: Dict[str, Any], n_samples: int, code_bits: int) -> None:
    pm1 = np.asarray(binary_codes["pm1"])
    codes_01 = np.asarray(binary_codes["01"])
    packed = np.asarray(binary_codes["packed"])
    assert pm1.shape == (n_samples, code_bits)
    assert codes_01.shape == (n_samples, code_bits)
    assert pm1.dtype in (np.float32, np.float64)
    assert np.issubdtype(codes_01.dtype, np.integer)
    assert packed.shape[0] == n_samples


def _assert_result_schema_v2(result: Dict[str, Any]) -> None:
    assert result["version"] == "phase1-v2"
    assert result["status"] == "success"
    system = result["system"]
    assert isinstance(system, dict)
    for key in ("hostname", "platform", "python_version", "cpu_model", "gpu_model", "num_gpus"):
        assert key in system
    normalization = result["normalization"]
    assert isinstance(normalization, dict) and "l2" in normalization
    instrumentation = result["instrumentation"]
    assert isinstance(instrumentation, dict)
    for key in ("binarization_time_ms", "packing_time_ms"):
        assert key in instrumentation
    assert tuple(result["metrics_requested"]) == EXPECTED_METRICS
    similarity = result["similarity_metrics"]
    assert similarity is not None
    assert 0.0 <= similarity["mean_hamming"] <= 1.0
    assert -1.0 <= similarity["mean_cosine"] <= 1.0
    assert result["retrieval_metrics"] is not None
    assert result["classification_metrics"] is not None


def _assert_log_schema_v2(entry: Dict[str, Any]) -> None:
    required_fields = [
        "version",
        "status",
        "encoder_name",
        "dataset_name",
        "dataset_format",
        "code_bits",
        "projection_type",
        "projection_seed",
        "runner",
        "seed",
        "git_hash",
        "metrics",
        "metrics_requested",
        "hypotheses",
        "system",
        "normalization",
        "instrumentation",
        "retrieval_k",
        "config_fingerprint",
        "similarity_metrics",
        "retrieval_metrics",
        "classification_metrics",
    ]
    for field in required_fields:
        assert field in entry, f"log missing {field}"
    assert tuple(entry["metrics_requested"]) == EXPECTED_METRICS
    system = entry["system"]
    for key in ("python_version", "platform", "hostname", "cpu_model", "gpu_model", "num_gpus"):
        assert key in system, f"{key} missing from system metadata"
    similarity = entry["similarity_metrics"]
    assert 0.0 <= similarity["mean_hamming"] <= 1.0
    assert -1.0 <= similarity["mean_cosine"] <= 1.0
    assert entry["retrieval_metrics"] is not None
    assert entry["classification_metrics"] is not None


@pytest.fixture(scope="module", autouse=True)
def register_phase1_dataset() -> None:
    existing = dataset_catalog.DATASET_REGISTRY.get(DATASET_NAME)
    if existing is None:
        dataset_catalog.DATASET_REGISTRY[DATASET_NAME] = DatasetSpec(
            name=DATASET_NAME,
            file_format="npy",
            embedding_dim=4,
            requires_l2_normalization=True,
            required_fields=("id", "embedding", "label"),
            supports_similarity=True,
            supports_retrieval=True,
            supports_classification=True,
            default_encoder=ENCODER_NAME,
        )
    try:
        yield
    finally:
        if existing is None:
            dataset_catalog.DATASET_REGISTRY.pop(DATASET_NAME, None)


def test_golden_config_paths_are_posix(tmp_path: Path) -> None:
    config_data = _load_yaml_config()
    file_entries = list(config_data["embedding_files"])
    file_entries.append(config_data["classification_labels"])
    for entry in file_entries:
        assert "\\" not in entry, f"path must use POSIX separators: {entry!r}"
        assert entry.startswith("tests/"), f"golden path must stay within repo tree: {entry}"
        assert (BASE_DIR / entry).exists(), f"golden file missing on disk: {entry}"

    output_dir = config_data["output_dir"]
    assert "\\" not in output_dir, f"output_dir must use POSIX separators: {output_dir!r}"
    assert output_dir.startswith("tests/"), "golden output_dir must remain relative to repo root"
    assert tuple(config_data["metrics"]) == EXPECTED_METRICS
    assert tuple(config_data["tasks"]) == EXPECTED_METRICS

    config_path, materialized_output_dir = _materialize_config(tmp_path / "posix_check")
    materialized_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    materialized_paths = list(materialized_config["embedding_files"])
    materialized_paths.append(materialized_config["classification_labels"])
    materialized_paths.append(materialized_config["output_dir"])
    for entry in materialized_paths:
        assert "\\" not in entry, f"materialized config path must be POSIX: {entry!r}"
    # Cleanup temporary artifacts created for this check.
    config_path.unlink()
    for artifact in materialized_output_dir.glob("*"):
        artifact.unlink()
    materialized_output_dir.rmdir()


def test_classification_labels_match_embeddings() -> None:
    config_data = _load_yaml_config()
    embeddings = np.load(BASE_DIR / config_data["embedding_files"][0], allow_pickle=False)
    labels = np.load(BASE_DIR / config_data["classification_labels"], allow_pickle=False)
    assert embeddings.shape[0] == labels.shape[0] > 0
    assert labels.ndim == 1
    assert set(np.unique(labels)) <= {0, 1}, "golden labels must remain binary"


def test_phase1_golden_success_regression(tmp_path: Path) -> None:
    config_path, output_dir = _materialize_config(tmp_path)
    result = runner.run_phase1_experiment(str(config_path))
    golden_result = _load_golden_result()
    _assert_result_schema_v2(result)
    _assert_result_schema_v2(golden_result)

    discrete_fields = [
        "version",
        "status",
        "runner",
        "encoder_name",
        "dataset_name",
        "dataset_format",
        "code_bits",
        "projection_type",
        "seed",
        "projection_seed",
        "hypotheses",
    ]
    for field in discrete_fields:
        assert result[field] == golden_result[field]

    _assert_metric_tree_close(result["similarity_metrics"], golden_result["similarity_metrics"])
    _assert_metric_tree_close(result["retrieval_metrics"], golden_result["retrieval_metrics"])
    _assert_metric_tree_close(result["classification_metrics"], golden_result["classification_metrics"])
    _assert_instrumentation_close(result["instrumentation"], golden_result["instrumentation"])
    assert result["normalization"] == golden_result["normalization"]
    assert tuple(result["metrics_requested"]) == EXPECTED_METRICS
    assert result["version"] == golden_result["version"] == "phase1-v2"
    system = result["system"]
    golden_system = golden_result["system"]
    for key in ("hostname", "platform", "python_version", "cpu_model", "gpu_model", "num_gpus"):
        assert key in system, f"{key} missing from result system metadata"
        assert key in golden_system, f"{key} missing from golden system metadata"
    _assert_system_metadata_close(system, golden_system)
    assert result["retrieval_k"] == golden_result["retrieval_k"]
    classification = result["classification_metrics"]
    assert set(classification.keys()) == CLASSIFICATION_KEYS
    _assert_classification_degradation(classification)
    _assert_classification_degradation(golden_result["classification_metrics"])

    num_samples = len(result["binary_codes"]["pm1"])
    _assert_binary_code_shapes(result["binary_codes"], num_samples, result["code_bits"])


def test_phase1_golden_log_matches_reference(tmp_path: Path) -> None:
    config_path, output_dir = _materialize_config(tmp_path)
    runner_result = runner.run_phase1_experiment(str(config_path))
    live_log = _load_latest_log(output_dir)
    golden_log = _load_golden_log()
    golden_result = _load_golden_result()
    _assert_log_schema_v2(live_log)
    _assert_log_schema_v2(golden_log)

    for field in ("status", "encoder_name", "dataset_name", "code_bits", "projection_type"):
        assert live_log[field] == runner_result[field]
        assert golden_log[field] == golden_result[field]
    assert live_log["config_fingerprint"] == runner_result["config_fingerprint"]
    assert golden_log["config_fingerprint"] == golden_result["config_fingerprint"]
    assert isinstance(live_log.get("timestamp"), str) and live_log["timestamp"]
    assert live_log["status"] == "success"
    assert live_log.get("normalization") in (None, golden_result["normalization"])

    live_system = live_log["system"]
    golden_system = golden_log["system"]
    for key in ("hostname", "platform", "python_version", "cpu_model", "gpu_model", "num_gpus"):
        assert key in live_system
        assert key in golden_system
    _assert_system_metadata_close(live_system, golden_system)

    _assert_metric_tree_close(live_log["similarity_metrics"], golden_log["similarity_metrics"])
    _assert_metric_tree_close(live_log["retrieval_metrics"], golden_log["retrieval_metrics"])
    _assert_metric_tree_close(live_log["classification_metrics"], golden_log["classification_metrics"])
    assert set(live_log["metrics"].keys()) == set(EXPECTED_METRICS)
    assert set(golden_log["metrics"].keys()) == set(EXPECTED_METRICS)
    _assert_metric_tree_close(live_log["metrics"], golden_log["metrics"])
    _assert_log_binary_codes_sanitized(live_log["binary_codes"])
    _assert_classification_degradation(live_log["classification_metrics"])
    _assert_classification_degradation(golden_log["classification_metrics"])


def test_phase1_golden_is_deterministic(tmp_path: Path) -> None:
    config_path_one, _ = _materialize_config(tmp_path / "first")
    first = runner.run_phase1_experiment(str(config_path_one))
    config_path_two, _ = _materialize_config(tmp_path / "second")
    second = runner.run_phase1_experiment(str(config_path_two))
    _assert_result_schema_v2(first)
    _assert_result_schema_v2(second)

    _assert_metric_tree_close(first["similarity_metrics"], second["similarity_metrics"])
    _assert_metric_tree_close(first["retrieval_metrics"], second["retrieval_metrics"])
    _assert_metric_tree_close(first["classification_metrics"], second["classification_metrics"])
    _assert_classification_degradation(first["classification_metrics"])
    _assert_classification_degradation(second["classification_metrics"])
    np.testing.assert_array_equal(first["binary_codes"]["pm1"], second["binary_codes"]["pm1"])
    np.testing.assert_array_equal(first["binary_codes"]["01"], second["binary_codes"]["01"])
    np.testing.assert_array_equal(first["binary_codes"]["packed"], second["binary_codes"]["packed"])


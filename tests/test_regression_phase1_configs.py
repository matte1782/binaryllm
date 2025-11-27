"""Table-driven regression tests for Phase 1 configs (T13 coverage restoration)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pytest
import yaml

from src.core import dataset_catalog
from src.core.dataset_catalog import DatasetSpec

runner = pytest.importorskip("src.experiments.runners.phase1_binary_embeddings")

BASE_DIR = Path(__file__).resolve().parents[1]
GOLDEN_CONFIG_PATH = BASE_DIR / "tests" / "data" / "phase1_golden" / "config_phase1_synthetic_v1.yaml"
DATASET_NAME = "phase1_synthetic_golden"
ENCODER_NAME = "synthetic_encoder_4d"


def _load_template_config() -> Dict[str, Any]:
    return yaml.safe_load(GOLDEN_CONFIG_PATH.read_text(encoding="utf-8"))


def _materialize_config(tmp_path: Path, name: str, overrides: Dict[str, Any]) -> Path:
    config_data = _load_template_config()
    config_data.update(overrides)
    embedding_files = [
        (BASE_DIR / Path(path)).resolve().as_posix() for path in config_data["embedding_files"]
    ]
    assert all("/" in entry and "\\" not in entry for entry in embedding_files)
    config_data["embedding_files"] = embedding_files
    config_data["classification_labels"] = (
        (BASE_DIR / Path(config_data["classification_labels"])).resolve().as_posix()
    )
    output_dir = (tmp_path / f"artifacts_{name}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_data["output_dir"] = output_dir.as_posix()

    config_path = tmp_path / f"{name}.json"
    config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
    return config_path


@pytest.fixture(scope="module", autouse=True)
def register_phase1_dataset() -> Iterable[None]:
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


def _assert_result_schema(result: Dict[str, Any]) -> None:
    assert result["version"] == "phase1-v2"
    assert result["status"] == "success"
    assert tuple(result["metrics_requested"])
    assert isinstance(result["normalization"], dict)
    similarity = result["similarity_metrics"]
    assert similarity is not None
    assert 0.0 <= similarity["mean_hamming"] <= 1.0
    assert -1.0 <= similarity["mean_cosine"] <= 1.0
    assert result["retrieval_metrics"] is not None
    assert result["classification_metrics"] is not None
    instrumentation = result["instrumentation"]
    for key in ("binarization_time_ms", "packing_time_ms"):
        assert key in instrumentation
    system = result["system"]
    for key in ("hostname", "platform", "python_version", "cpu_model", "gpu_model", "num_gpus"):
        assert key in system, f"{key} missing from system metadata"


def _assert_runs_deterministic(first: Dict[str, Any], second: Dict[str, Any]) -> None:
    assert first["metrics"] == second["metrics"]
    np.testing.assert_array_equal(np.asarray(first["binary_codes"]["pm1"]), np.asarray(second["binary_codes"]["pm1"]))
    np.testing.assert_array_equal(np.asarray(first["binary_codes"]["01"]), np.asarray(second["binary_codes"]["01"]))
    np.testing.assert_array_equal(np.asarray(first["binary_codes"]["packed"]), np.asarray(second["binary_codes"]["packed"]))


def test_phase1_configs_remain_regression_stable(tmp_path: Path) -> None:
    scenarios = [
        {"name": "gaussian_32bits", "overrides": {}},
        {"name": "gaussian_64bits", "overrides": {"code_bits": 64}},
        {"name": "gaussian_custom_seed", "overrides": {"seed": 777}},
    ]

    for scenario in scenarios:
        config_path = _materialize_config(tmp_path, scenario["name"], scenario["overrides"])
        first = runner.run_phase1_experiment(str(config_path))
        second = runner.run_phase1_experiment(str(config_path))

        _assert_result_schema(first)
        _assert_result_schema(second)
        _assert_runs_deterministic(first, second)


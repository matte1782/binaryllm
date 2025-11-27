"""Phase 1 runner contract tests (T9).

These tests define the behavioral contract for `run_phase1_experiment` before
the implementation exists. They rely on the Phase 1 architecture spec and on
T8 utilities (config, seed, IO, logging) to ensure that the runner:

- loads configs via the shared loader (with deterministic fingerprint),
- seeds every stochastic component,
- wires dataset/encoder specs and IO correctly,
- produces similarity + retrieval metrics deterministically,
- writes structured JSON logs with the required metadata,
- surfaces clear errors for malformed configs.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from src.core import dataset_catalog
from src.utils import config as config_utils

runner = importlib.import_module("src.experiments.runners.phase1_binary_embeddings")


def _write_embeddings(tmp_path: Path) -> Path:
    raw = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    normalized = raw / norms
    path = tmp_path / "synthetic_embeddings.npy"
    np.save(path, normalized)
    return path


def _write_config(
    tmp_path: Path,
    *,
    embeddings_path: Path,
    output_dir: Path,
    overrides: Dict[str, Any] | None = None,
) -> Path:
    base_config: Dict[str, Any] = {
        "runner": "phase1_binary_embeddings",
        "encoder_name": "synthetic_encoder_4d",
        "dataset_name": "phase1_synthetic_toy",
        "code_bits": 64,
        "projection_type": "gaussian",
        "tasks": ["similarity", "retrieval"],
        "seed": 4242,
        "metrics": ["similarity", "retrieval"],
        "hypotheses": ["H1"],
        "embedding_files": [str(embeddings_path)],
        "dataset_format": "npy",
        "output_dir": str(output_dir),
    }
    if overrides:
        base_config.update(overrides)
    fingerprint_source = json.dumps(base_config, sort_keys=True).encode("utf-8")
    fingerprint = hashlib.sha256(fingerprint_source).hexdigest()[:8]
    config_path = tmp_path / f"phase1_config_{fingerprint}.json"
    config_path.write_text(json.dumps(base_config, indent=2))
    return config_path


def _load_log(output_dir: Path) -> Dict[str, Any]:
    log_files = sorted(output_dir.glob("*.json"))
    assert log_files, "runner must emit at least one JSON log file"
    latest = log_files[-1]
    return json.loads(latest.read_text(encoding="utf-8"))


def _write_embeddings_with_nan(tmp_path: Path) -> Path:
    data = np.array(
        [
            [np.nan, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    path = tmp_path / "nan_embeddings.npy"
    np.save(path, data)
    return path


def _assert_error_payload(result: Dict[str, Any], *, stage: str) -> None:
    assert result["status"] == "error", "runner must not raise raw exceptions"
    assert result["similarity_metrics"] is None
    assert result["retrieval_metrics"] is None
    assert result["classification_metrics"] is None
    error = result.get("error")
    assert isinstance(error, dict), "error block must be present in result schema v2"
    assert error.get("stage") == stage
    assert "message" in error
    assert error.get("exception_type") is not None


def test_run_phase1_experiment_logs_and_returns_metrics(tmp_path: Path) -> None:
    embeddings_path = _write_embeddings(tmp_path)
    output_dir = tmp_path / "artifacts_success"
    config_path = _write_config(tmp_path, embeddings_path=embeddings_path, output_dir=output_dir)

    result = runner.run_phase1_experiment(str(config_path))
    assert result["status"] == "success"
    for key in ("encoder_name", "dataset_name", "code_bits", "projection_type", "hypotheses", "seed"):
        assert key in result
    assert "H1" in result["hypotheses"]

    assert "metrics" in result and set(result["metrics"]) >= {"similarity", "retrieval"}
    similarity_metrics = result["metrics"]["similarity"]
    retrieval_metrics = result["metrics"]["retrieval"]
    for field in ("cosine_hamming_spearman", "mean_cosine", "mean_hamming"):
        assert field in similarity_metrics
        assert isinstance(similarity_metrics[field], float)
    for bucket in ("topk_overlap", "ndcg", "recall"):
        assert bucket in retrieval_metrics and isinstance(retrieval_metrics[bucket], dict)
        assert "k=3" in retrieval_metrics[bucket]

    instrumentation = result.get("instrumentation", {})
    assert "binarization_time_ms" in instrumentation
    assert "packing_time_ms" in instrumentation

    cfg_loaded = config_utils.load_config(config_path)
    log_entry = _load_log(output_dir)
    assert log_entry["encoder_name"] == result["encoder_name"]
    assert log_entry["dataset_name"] == result["dataset_name"]
    assert log_entry["code_bits"] == result["code_bits"]
    assert log_entry["projection_type"] == result["projection_type"]
    assert "timestamp" in log_entry
    assert "git_hash" in log_entry
    assert "metrics" in log_entry
    assert log_entry["hypotheses"] == result["hypotheses"]
    assert log_entry.get("config_fingerprint") == cfg_loaded.fingerprint


def test_runner_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    embeddings_path = _write_embeddings(tmp_path)
    output_dir_one = tmp_path / "artifacts_run_one"
    config_path_one = _write_config(tmp_path, embeddings_path=embeddings_path, output_dir=output_dir_one)
    first = runner.run_phase1_experiment(str(config_path_one))

    output_dir_two = tmp_path / "artifacts_run_two"
    config_path_two = _write_config(
        tmp_path,
        embeddings_path=embeddings_path,
        output_dir=output_dir_two,
    )
    second = runner.run_phase1_experiment(str(config_path_two))
    assert first["metrics"] == second["metrics"]
    assert set(first["instrumentation"]) == set(second["instrumentation"])
    for key in first["instrumentation"]:
        assert isinstance(first["instrumentation"][key], float)
        assert isinstance(second["instrumentation"][key], float)
        assert abs(first["instrumentation"][key] - second["instrumentation"][key]) < 5.0


def test_runner_reports_error_when_seed_missing(tmp_path: Path) -> None:
    embeddings_path = _write_embeddings(tmp_path)
    output_dir = tmp_path / "artifacts_missing_seed"
    config_path = _write_config(tmp_path, embeddings_path=embeddings_path, output_dir=output_dir)

    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    config_data.pop("seed", None)
    config_path.write_text(json.dumps(config_data))

    result = runner.run_phase1_experiment(str(config_path))
    _assert_error_payload(result, stage="seed_extraction")


def test_runner_rejects_unknown_metric(tmp_path: Path) -> None:
    embeddings_path = _write_embeddings(tmp_path)
    output_dir = tmp_path / "artifacts_bad_metric"
    config_path = _write_config(
        tmp_path,
        embeddings_path=embeddings_path,
        output_dir=output_dir,
        overrides={"metrics": ["similarity", "unknown_metric"]},
    )

    with pytest.raises(ValueError, match="metric.*unsupported"):
        runner.run_phase1_experiment(str(config_path))


def test_runner_errors_on_unknown_dataset(tmp_path: Path) -> None:
    embeddings_path = _write_embeddings(tmp_path)
    output_dir = tmp_path / "artifacts_bad_dataset"
    config_path = _write_config(
        tmp_path,
        embeddings_path=embeddings_path,
        output_dir=output_dir,
        overrides={"dataset_name": "nonexistent_dataset"},
    )

    with pytest.raises(dataset_catalog.UnknownDatasetError):
        runner.run_phase1_experiment(str(config_path))


def test_runner_rejects_non_integer_seed(tmp_path: Path) -> None:
    embeddings_path = _write_embeddings(tmp_path)
    output_dir = tmp_path / "artifacts_bad_seed_type"
    config_path = _write_config(
        tmp_path,
        embeddings_path=embeddings_path,
        output_dir=output_dir,
        overrides={"seed": "abc"},
    )

    # Config schema catches non-integer seeds during load, so we expect a structured
    # error payload (no exception) tagged with the config_load stage.
    result = runner.run_phase1_experiment(str(config_path))
    assert result["status"] == "error"
    assert result["error"]["stage"] == "config_load"
    assert "seed" in result["error"]["message"]


def test_runner_rejects_unsupported_projection_type(tmp_path: Path) -> None:
    embeddings_path = _write_embeddings(tmp_path)
    output_dir = tmp_path / "artifacts_bad_projection"
    config_path = _write_config(
        tmp_path,
        embeddings_path=embeddings_path,
        output_dir=output_dir,
        overrides={"projection_type": "bogus_projection"},
    )

    # Phase-1 spec: unsupported projection must fail schema validation and raise
    # a ValueError instead of returning a structured error payload.
    with pytest.raises(ValueError, match=r"projection_type.*must be one of"):
        runner.run_phase1_experiment(str(config_path))


def test_runner_handles_nan_embeddings_with_load_error(tmp_path: Path) -> None:
    embeddings_path = _write_embeddings_with_nan(tmp_path)
    output_dir = tmp_path / "artifacts_nan_embeddings"
    config_path = _write_config(tmp_path, embeddings_path=embeddings_path, output_dir=output_dir)

    # Embedding ingestion happens after config succeeds, so NaNs must surface via the
    # structured load_embeddings stage rather than raising upstream exceptions.
    result = runner.run_phase1_experiment(str(config_path))
    _assert_error_payload(result, stage="load_embeddings")


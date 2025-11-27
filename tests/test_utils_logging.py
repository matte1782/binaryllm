from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.utils import logging as logging_utils


def _valid_log_entry() -> dict:
    return {
        "version": "phase1-v2",
        "status": "success",
        "encoder_name": "phase1_encoder",
        "dataset_name": "synthetic_retrieval",
        "dataset_format": "npy",
        "code_bits": 128,
        "projection_type": "gaussian",
        "projection_seed": 11,
        "runner": "phase1_runner",
        "seed": 11,
        "git_hash": "abc123",
        "metrics": {
            "similarity": {"mean_hamming": 0.5},
            "retrieval": {"topk_overlap": {"k=3": 1.0}},
            "classification": {"float_accuracy": 0.9, "binary_accuracy": 0.8},
        },
        "metrics_requested": ["similarity", "retrieval", "classification"],
        "hypotheses": ["H1"],
        "system": {
            "hostname": "test-host",
            "platform": "Linux",
            "python_version": "3.12.1",
            "cpu_model": "Test CPU",
            "gpu_model": "none",
            "num_gpus": 0,
        },
        "normalization": {"l2": True},
        "instrumentation": {"binarization_time_ms": 0.1, "packing_time_ms": 0.2},
        "retrieval_k": 3,
        "config_fingerprint": "abc123",
        "similarity_metrics": {"mean_hamming": 0.5, "mean_cosine": 0.1},
        "retrieval_metrics": {"topk_overlap": {"k=3": 1.0}},
        "classification_metrics": {
            "float_accuracy": 0.9,
            "binary_accuracy": 0.8,
            "accuracy_delta": -0.1,
            "float_f1": 0.9,
            "binary_f1": 0.8,
        },
        "binary_codes": {
            "pm1": [[1.0, -1.0]],
            "01": [[1, 0]],
            "packed": [[7]],
        },
    }


def test_write_log_entry_creates_sorted_json(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    entry = _valid_log_entry()
    log_path = logging_utils.write_log_entry(log_dir, entry)

    assert log_path.parent == log_dir
    payload = json.loads(log_path.read_text())
    assert list(payload.keys()) == sorted(payload.keys())
    datetime.fromisoformat(payload["timestamp"])
    for key in _valid_log_entry():
        assert payload[key] == entry[key]
    expected_system_keys = ("hostname", "platform", "python_version", "cpu_model", "gpu_model", "num_gpus")
    assert set(payload["system"]) == set(expected_system_keys)
    for key in expected_system_keys:
        assert payload["system"][key] == entry["system"][key]


def test_write_log_entry_creates_directory(tmp_path: Path) -> None:
    log_dir = tmp_path / "nested" / "logs"
    logging_utils.write_log_entry(log_dir, _valid_log_entry())
    assert log_dir.exists()


def test_write_log_entry_rejects_missing_required_field(tmp_path: Path) -> None:
    entry = _valid_log_entry()
    entry.pop("encoder_name")
    with pytest.raises(ValueError, match="encoder_name"):
        logging_utils.write_log_entry(tmp_path / "logs", entry)


def test_write_log_entry_rejects_incomplete_system_metadata(tmp_path: Path) -> None:
    entry = _valid_log_entry()
    entry["system"] = dict(entry["system"])
    entry["system"].pop("cpu_model")
    with pytest.raises(ValueError, match="cpu_model"):
        logging_utils.write_log_entry(tmp_path / "logs", entry)


def test_write_log_entry_preserves_v2_success_payload(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs_success"
    entry = _valid_log_entry()
    log_path = logging_utils.write_log_entry(log_dir, entry)
    payload = json.loads(log_path.read_text(encoding="utf-8"))

    assert payload["version"] == "phase1-v2"
    assert payload["status"] == "success"
    assert payload["normalization"] == entry["normalization"]
    assert payload["metrics_requested"] == entry["metrics_requested"]
    assert payload["binary_codes"] == entry["binary_codes"]
    system = payload["system"]
    for key in ("hostname", "platform", "python_version", "cpu_model", "gpu_model", "num_gpus"):
        assert key in system
    assert "timestamp" in payload


def test_write_log_entry_handles_v2_error_payload(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs_error"
    entry = _valid_log_entry()
    entry["status"] = "error"
    entry["error"] = {
        "stage": "load_embeddings",
        "message": "failed to load",
        "variant": "binary_embedding_engine_v0.1",
        "exception_type": "ValueError",
    }
    log_path = logging_utils.write_log_entry(log_dir, entry)
    payload = json.loads(log_path.read_text(encoding="utf-8"))

    assert payload["status"] == "error"
    assert payload["version"] == "phase1-v2"
    system = payload["system"]
    expected_system_keys = ("hostname", "platform", "python_version", "cpu_model", "gpu_model", "num_gpus")
    assert set(system) == set(expected_system_keys)
    assert "timestamp" in payload
    assert payload["normalization"] == entry["normalization"]
    assert payload["metrics_requested"] == entry["metrics_requested"]
    assert payload["binary_codes"] == entry["binary_codes"]
    assert payload["error"]["stage"] == entry["error"]["stage"]
    assert payload["error"]["exception_type"] == entry["error"]["exception_type"]
    assert payload["error"]["variant"] == entry["error"]["variant"]


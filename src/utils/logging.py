"""Structured logging helpers for Phase 1 runs."""

from __future__ import annotations

import json
import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

REQUIRED_LOG_FIELDS = [
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
    "normalization",
    "instrumentation",
    "system",
]


def init_run_log(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _validate_entry(entry: Dict[str, Any]) -> None:
    for field in REQUIRED_LOG_FIELDS:
        if field not in entry:
            raise ValueError(f"logging: missing '{field}' in log entry")

    if not isinstance(entry["metrics"], dict):
        raise ValueError("logging: metrics must be a dict")
    if not isinstance(entry["hypotheses"], list):
        raise ValueError("logging: hypotheses must be a list")
    if not isinstance(entry["metrics_requested"], list):
        raise ValueError("logging: metrics_requested must be a list")
    if not isinstance(entry["normalization"], dict):
        raise ValueError("logging: normalization must be a dict")
    if not isinstance(entry["instrumentation"], dict):
        raise ValueError("logging: instrumentation must be a dict")
    system_block = entry.get("system")
    if not isinstance(system_block, dict):
        raise ValueError("logging: system metadata must be a dict")
    for key in ("hostname", "platform", "python_version", "cpu_model", "gpu_model", "num_gpus"):
        if key not in system_block:
            raise ValueError(f"logging: system metadata missing '{key}'")
    error_block = entry.get("error")
    if error_block is not None and not isinstance(error_block, dict):
        raise ValueError("logging: error block must be a dict or null")


def write_log_entry(directory: Path, entry: Dict[str, Any]) -> Path:
    init_run_log(directory)
    _validate_entry(entry)
    payload = copy.deepcopy(entry)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    log_path = directory / f"log_{payload['timestamp'].replace(':', '_')}.json"
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
    return log_path


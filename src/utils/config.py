"""Configuration loading and validation utilities for Phase 1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


# Phase 1: only 'gaussian' is implemented end-to-end (config + engine + runner).
# Other projections (e.g. 'rademacher', 'prelearned_linear') are reserved for Phase 2.
SUPPORTED_PROJECTIONS: set[str] = {"gaussian"}
SUPPORTED_CODE_BITS: set[int] = {32, 64, 128, 256}
REQUIRED_FIELDS: Dict[str, type] = {
    "encoder_name": str,
    "dataset_name": str,
    "code_bits": int,
    "projection_type": str,
    "runner": str,
    "tasks": list,
    "seed": int,
}
OPTIONAL_FIELDS: Dict[str, type] = {
    "embedding_files": list,
    "dataset_format": str,
    "metrics": list,
    "hypotheses": list,
    "output_dir": str,
    "classification_labels": str,
}


@dataclass(frozen=True)
class FrozenConfig(Mapping[str, Any]):
    """Immutable mapping wrapper with deterministic fingerprint attribute."""

    _data: Dict[str, Any]
    fingerprint: str

    def __getitem__(self, key: str) -> Any:  # pragma: no cover - Mapping interface
        return self._data[key]

    def __iter__(self) -> Iterator[str]:  # pragma: no cover - Mapping interface
        return iter(self._data)

    def __len__(self) -> int:  # pragma: no cover - Mapping interface
        return len(self._data)


def _read_config_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text())
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text())
    raise ValueError(f"config: unsupported extension '{suffix}' (expected .json/.yaml)")


def _validate_schema(payload: Dict[str, Any]) -> Dict[str, Any]:
    for field, field_type in REQUIRED_FIELDS.items():
        if field not in payload:
            raise ValueError(f"config: missing required field '{field}'")
        if not isinstance(payload[field], field_type):
            raise ValueError(f"config: '{field}' must be of type {field_type.__name__}")

    optional_allowed = set(OPTIONAL_FIELDS)
    unknown_keys = set(payload) - set(REQUIRED_FIELDS) - optional_allowed
    if unknown_keys:
        raise ValueError(f"config: unknown fields {sorted(unknown_keys)} present")

    code_bits = payload["code_bits"]
    if code_bits not in SUPPORTED_CODE_BITS:
        raise ValueError(f"config: code_bits must be in {sorted(SUPPORTED_CODE_BITS)} (received {code_bits})")

    projection = payload["projection_type"]
    if projection not in SUPPORTED_PROJECTIONS:
        raise ValueError(
            f"config: projection_type must be one of {sorted(SUPPORTED_PROJECTIONS)} (received '{projection}')"
        )

    tasks = payload["tasks"]
    if not isinstance(tasks, list) or any(not isinstance(task, str) for task in tasks):
        raise ValueError("config: tasks must be a list[str]")
    payload["tasks"] = list(tasks)

    for field, field_type in OPTIONAL_FIELDS.items():
        if field not in payload:
            continue
        value = payload[field]
        if not isinstance(value, field_type):
            raise ValueError(f"config: '{field}' must be of type {field_type.__name__}")
        if field == "embedding_files":
            if not value or any(not isinstance(path, str) for path in value):
                raise ValueError("config: embedding_files must be a non-empty list of strings")
            payload[field] = list(value)
        elif field in {"metrics", "hypotheses"}:
            if any(not isinstance(item, str) for item in value):
                raise ValueError(f"config: {field} must contain only strings")
            payload[field] = list(value)

    return payload


def _stable_fingerprint(data: Dict[str, Any]) -> str:
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> FrozenConfig:
    resolved = Path(path)
    payload = _read_config_file(resolved)
    validated = _validate_schema(payload)
    fingerprint = _stable_fingerprint(validated)
    return FrozenConfig(validated, fingerprint)


def config_hash(config: Mapping[str, Any]) -> str:
    """Expose deterministic hash for logging."""
    return _stable_fingerprint(dict(config))


from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from src.utils import config as config_utils

FIXTURE_DIR = Path(__file__).resolve().parent / "data" / "phase1_synthetic"
VALID_JSON = FIXTURE_DIR / "t8_config_valid.json"
VALID_YAML = FIXTURE_DIR / "t8_config_valid.yaml"


def test_load_config_from_json_returns_immutable_mapping() -> None:
    config = config_utils.load_config(VALID_JSON)
    assert isinstance(config, Mapping)
    assert config["encoder_name"] == "phase1_encoder"
    assert config["tasks"] == ["similarity", "retrieval"]
    with pytest.raises(TypeError):
        config["encoder_name"] = "other"  # type: ignore[index]


def test_load_config_from_yaml_matches_json() -> None:
    json_config = config_utils.load_config(VALID_JSON)
    yaml_config = config_utils.load_config(VALID_YAML)
    assert dict(json_config) == dict(yaml_config)


def test_load_config_fingerprint_is_deterministic() -> None:
    first = config_utils.load_config(VALID_JSON).fingerprint
    second = config_utils.load_config(VALID_JSON).fingerprint
    assert isinstance(first, str)
    assert first == second


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    payload = json.loads(VALID_JSON.read_text())
    payload["unexpected"] = "boom"
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unknown.*unexpected"):
        config_utils.load_config(path)


def test_load_config_requires_all_fields(tmp_path: Path) -> None:
    payload = json.loads(VALID_JSON.read_text())
    payload.pop("dataset_name", None)
    path = tmp_path / "missing.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="dataset_name"):
        config_utils.load_config(path)


def test_load_config_rejects_invalid_projection_type(tmp_path: Path) -> None:
    payload = json.loads(VALID_JSON.read_text())
    payload["projection_type"] = "invalid_projection"
    path = tmp_path / "bad_projection.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="projection_type"):
        config_utils.load_config(path)


def test_load_config_rejects_invalid_code_bits(tmp_path: Path) -> None:
    payload = json.loads(VALID_JSON.read_text())
    payload["code_bits"] = 17
    path = tmp_path / "bad_code_bits.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="code_bits"):
        config_utils.load_config(path)


def test_load_config_requires_list_of_tasks(tmp_path: Path) -> None:
    payload = json.loads(VALID_JSON.read_text())
    payload["tasks"] = "similarity"
    path = tmp_path / "tasks_string.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="tasks"):
        config_utils.load_config(path)



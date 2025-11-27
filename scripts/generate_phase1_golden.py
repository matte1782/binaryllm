"""Generate Phase 1 synthetic golden artifacts for the regression suite (T13 v4).

This helper script is intentionally lightweight and deterministic so that
engineers can refresh the golden dataset/config/result/log whenever the Phase 1
pipeline evolves. It performs the following actions:

1. Emit a small synthetic embedding dataset plus classification labels.
2. Write the canonical YAML config using POSIX-style relative paths.
3. Run the real Phase 1 runner to capture the reference result + log payloads.
4. Enforce Requirement F by ensuring the binary classifier degrades relative to the float baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Tuple

import numpy as np
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.core import dataset_catalog
from src.core.dataset_catalog import DatasetSpec
from src.experiments.runners.phase1_binary_embeddings import run_phase1_experiment

GOLDEN_DIR = BASE_DIR / "tests" / "data" / "phase1_golden"
EMBEDDINGS_PATH = GOLDEN_DIR / "embeddings_phase1_synthetic.npy"
LABELS_PATH = GOLDEN_DIR / "labels_phase1_synthetic.npy"
CONFIG_PATH = GOLDEN_DIR / "config_phase1_synthetic_v1.yaml"
GOLDEN_RESULT_PATH = GOLDEN_DIR / "golden_result_phase1_synthetic_v1.json"
GOLDEN_LOG_PATH = GOLDEN_DIR / "golden_log_phase1_synthetic_v1.jsonl"
RESOLVED_CONFIG_PATH = GOLDEN_DIR / "config_phase1_resolved.json"

DATASET_NAME = "phase1_synthetic_golden"
ENCODER_NAME = "synthetic_encoder_4d"


def _ensure_directories() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)


def _generate_embeddings_and_labels(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create three tight clusters in 4-D space with linearly separable labels."""
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.2],
            [-0.8, 0.0, 0.6, 0.0],
        ],
        dtype=np.float32,
    )
    cluster_size = 4
    embeddings = []
    labels = []
    for idx, center in enumerate(centers):
        noise = rng.normal(loc=0.0, scale=0.05, size=(cluster_size, center.size)).astype(np.float32)
        block = center + noise
        embeddings.append(block)
        # Alternate labels so clusters are linearly separable but non-trivial.
        label_value = idx % 2
        labels.append(np.full(cluster_size, label_value, dtype=np.int64))
    embedding_array = np.vstack(embeddings).astype(np.float32)
    label_array = np.concatenate(labels).astype(np.int64)
    return embedding_array, label_array


def _persist_embeddings(embeddings: np.ndarray, labels: np.ndarray) -> None:
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(LABELS_PATH, labels)


def _register_dataset_spec() -> None:
    """Ensure the synthetic dataset is known to the catalog before runner invocation."""
    if DATASET_NAME in dataset_catalog.DATASET_REGISTRY:
        return
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


def _as_posix_relative(path: Path) -> str:
    try:
        return path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def _write_portable_config(artifacts_dir: Path) -> None:
    """Write the YAML config with POSIX-style relative paths."""
    if artifacts_dir.is_absolute():
        abs_artifacts = artifacts_dir
    else:
        abs_artifacts = (BASE_DIR / artifacts_dir).resolve()
    abs_artifacts.mkdir(parents=True, exist_ok=True)
    portable_config = {
        "runner": "phase1_binary_embeddings",
        "encoder_name": ENCODER_NAME,
        "dataset_name": DATASET_NAME,
        "dataset_format": "npy",
        "code_bits": 32,
        "projection_type": "gaussian",
        "seed": 123,
        "metrics": ["similarity", "retrieval", "classification"],
        "tasks": ["similarity", "retrieval", "classification"],
        "hypotheses": ["H1"],
        "embedding_files": [_as_posix_relative(EMBEDDINGS_PATH)],
        "classification_labels": _as_posix_relative(LABELS_PATH),
        "output_dir": _as_posix_relative(abs_artifacts),
    }
    CONFIG_PATH.write_text(yaml.safe_dump(portable_config, sort_keys=True), encoding="utf-8")


def _prepare_runner_config(portable_artifacts: Path) -> Tuple[Path, Path]:
    """Resolve config paths to absolute strings for the runner invocation."""
    config_data: Dict[str, Any] = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    resolved = dict(config_data)
    resolved["embedding_files"] = [
        (BASE_DIR / Path(PurePosixPath(path))).resolve().as_posix()
        for path in config_data["embedding_files"]
    ]
    resolved["classification_labels"] = (
        (BASE_DIR / Path(PurePosixPath(config_data["classification_labels"]))).resolve().as_posix()
    )
    if portable_artifacts.is_absolute():
        abs_output_dir = portable_artifacts
    else:
        abs_output_dir = (BASE_DIR / portable_artifacts).resolve()
    abs_output_dir.mkdir(parents=True, exist_ok=True)
    resolved["output_dir"] = abs_output_dir.as_posix()
    RESOLVED_CONFIG_PATH.write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    return RESOLVED_CONFIG_PATH, abs_output_dir


def _sanitize_for_json(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _sanitize_for_json(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_sanitize_for_json(value) for value in payload]
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    if isinstance(payload, (np.integer, np.floating)):
        return payload.item()
    return payload


def _write_golden_artifacts(result: Dict[str, Any], log_dir: Path) -> None:
    GOLDEN_RESULT_PATH.write_text(
        json.dumps(_sanitize_for_json(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    log_files = sorted(log_dir.glob("*.json"))
    if not log_files:
        raise RuntimeError(f"No log files were emitted under {log_dir}.")
    latest_log = log_files[-1].read_text(encoding="utf-8").strip()
    GOLDEN_LOG_PATH.write_text(latest_log + "\n", encoding="utf-8")


def _cleanup_artifact_dir(directory: Path) -> None:
    if not directory.exists():
        return
    for artifact in directory.glob("*"):
        if artifact.is_dir():
            _cleanup_artifact_dir(artifact)
            artifact.rmdir()
        else:
            artifact.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate Phase 1 synthetic golden artifacts.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("tests/data/phase1_golden/artifacts"),
        help="Directory (relative to repo root) where runner artifacts are temporarily stored.",
    )
    args = parser.parse_args()

    _ensure_directories()
    _write_portable_config(args.artifacts_dir)
    _register_dataset_spec()

    degradation_result: Dict[str, Any] | None = None
    degradation_artifacts: Path | None = None
    selected_seed: int | None = None

    for offset in range(64):
        data_seed = 1337 + 977 * offset
        embeddings, labels = _generate_embeddings_and_labels(data_seed)
        _persist_embeddings(embeddings, labels)
        resolved_config_path, abs_artifacts = _prepare_runner_config(args.artifacts_dir)

        try:
            candidate_result = run_phase1_experiment(str(resolved_config_path))
        finally:
            if resolved_config_path.exists():
                resolved_config_path.unlink()

        metrics = candidate_result["classification_metrics"]
        if metrics["accuracy_delta"] < 0.0 and metrics["binary_accuracy"] < metrics["float_accuracy"]:
            degradation_result = candidate_result
            degradation_artifacts = abs_artifacts
            selected_seed = data_seed
            break

        _cleanup_artifact_dir(abs_artifacts)
        abs_artifacts.rmdir()

    if degradation_result is None or degradation_artifacts is None or selected_seed is None:
        raise RuntimeError(
            "Unable to synthesize golden data with classification degradation; expand search space."
        )

    print(
        "[generate_phase1_golden] Selected dataset seed "
        f"{selected_seed} (accuracy_delta={degradation_result['classification_metrics']['accuracy_delta']:.4f})"
    )
    _write_golden_artifacts(degradation_result, degradation_artifacts)
    _cleanup_artifact_dir(degradation_artifacts)
    degradation_artifacts.rmdir()


if __name__ == "__main__":
    main()


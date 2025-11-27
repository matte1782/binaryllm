"""Phase 1 experiment runner (T11/T12 schema v2)."""

from __future__ import annotations

import json
import os
import platform
import socket
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.core import dataset_catalog
from src.utils import config as config_utils
from src.utils import logging as logging_utils
from src.utils import seed as seed_utils
from src.utils.config import SUPPORTED_PROJECTIONS as CONFIG_SUPPORTED_PROJECTIONS
from src.variants.binary_embedding_engine import BinaryEmbeddingEngine

ALLOWED_METRICS = {"similarity", "retrieval", "classification"}
SCHEMA_SUPPORTED_PROJECTIONS = tuple(sorted(CONFIG_SUPPORTED_PROJECTIONS))
DEFAULT_RUNNER_NAME = "phase1_binary_embeddings"
DEFAULT_SUCCESS_OUTPUT_SUBDIR = "phase1_runner_artifacts"
DEFAULT_ERROR_OUTPUT_SUBDIR = "phase1_runner_errors"


def _validate_projection_type(value: Any) -> str:
    """Ensure projection_type adheres to the Phase-1 schema contract."""
    if not isinstance(value, str):
        raise ValueError("projection_type must be provided as a string.")
    allowed = CONFIG_SUPPORTED_PROJECTIONS
    if value not in allowed:
        raise ValueError(
            f"projection_type must be one of {list(SCHEMA_SUPPORTED_PROJECTIONS)} (received '{value}')"
        )
    return value


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    normalized = str(PurePosixPath(path_str.replace("\\", "/")))
    path = Path(normalized)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _load_embeddings(files: Sequence[str], expected_dim: int, base_dir: Path) -> np.ndarray:
    if not files:
        raise ValueError("run_phase1_experiment requires at least one embedding file.")
    arrays: List[np.ndarray] = []
    for entry in files:
        path = _resolve_path(entry, base_dir)
        if not path.exists():
            raise FileNotFoundError(f"embedding file not found: {path}")
        arrays.append(np.load(path, allow_pickle=False).astype(np.float32))
    data = np.concatenate(arrays, axis=0)
    if data.ndim != 2 or data.shape[1] != expected_dim:
        raise ValueError("loaded embeddings do not match encoder dimension.")
    if not np.isfinite(data).all():
        raise ValueError("loaded embeddings must contain only finite values.")
    return data


def _load_classification_labels(
    path_str: Optional[str],
    expected_rows: int,
    base_dir: Path,
) -> Optional[np.ndarray]:
    if not path_str:
        return None
    path = _resolve_path(path_str, base_dir)
    if not path.exists():
        raise FileNotFoundError(f"classification labels file not found: {path}")
    labels = np.load(path, allow_pickle=False)
    if labels.ndim != 1 or labels.shape[0] != expected_rows:
        raise ValueError("classification_labels must be a 1D array aligned with embeddings.")
    return labels.astype(np.int64)


def _cpu_model(uname: platform.uname_result) -> str:
    candidates = [
        platform.processor(),
        getattr(uname, "processor", ""),
        getattr(uname, "machine", ""),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return "unknown"


def _gpu_metadata() -> Tuple[str, int]:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                name = torch.cuda.get_device_name(0) or "cuda"
                return str(name), int(gpu_count)
    except Exception:
        pass

    env_name = os.environ.get("GPU_MODEL") or os.environ.get("GPU_NAME")
    if env_name:
        try:
            count = int(os.environ.get("NUM_GPUS", "1"))
        except ValueError:
            count = 1
        return env_name, max(count, 0)

    return "none", 0


def _system_metadata() -> Dict[str, Any]:
    uname = platform.uname()
    gpu_model, num_gpus = _gpu_metadata()
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": _cpu_model(uname),
        "gpu_model": gpu_model,
        "num_gpus": num_gpus,
    }


def _git_hash() -> str:
    git_dir = Path(".git")
    if not git_dir.exists():
        return "unknown"
    head = git_dir / "HEAD"
    try:
        ref = head.read_text().strip()
        if ref.startswith("ref:"):
            ref_path = git_dir / ref.split(" ", 1)[1].strip()
            return ref_path.read_text().strip()
        return ref
    except OSError:
        return "unknown"


def _sanitize_for_logging(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy arrays to Python lists so json.dump can serialize the payload."""

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    return convert(payload)


def _metrics_from_config(config_data: Dict[str, Any]) -> Sequence[str]:
    metrics = config_data.get("metrics") or config_data.get("tasks")
    if not metrics:
        raise ValueError("config must specify metrics or tasks.")
    invalid = set(metrics) - ALLOWED_METRICS
    if invalid:
        raise ValueError(f"config metrics contain unsupported entries: {sorted(invalid)}")
    return list(metrics)


def _initialize_result(runner_name: str) -> Dict[str, Any]:
    return {
        "version": "phase1-v2",
        "status": "error",
        "runner": runner_name,
        "encoder_name": None,
        "dataset_name": None,
        "dataset_format": None,
        "code_bits": None,
        "projection_type": None,
        "projection_seed": None,
        "seed": None,
        "hypotheses": [],
        "metrics_requested": (),
        "metrics": {},
        "similarity_metrics": None,
        "retrieval_metrics": None,
        "classification_metrics": None,
        "normalization": {"l2": None},
        "instrumentation": {},
        "system": _system_metadata(),
        "git_hash": _git_hash(),
        "config_fingerprint": None,
        "output_dir": None,
        "retrieval_k": None,
        "binary_codes": None,
        "error": None,
    }


def _resolve_output_dir(config_dir: Path, raw_value: Optional[str], *, fallback_subdir: str) -> Path:
    candidate = Path(raw_value) if raw_value else Path(fallback_subdir)
    if not candidate.is_absolute():
        candidate = (config_dir / candidate).resolve()
    return candidate


def _apply_error(result: Dict[str, Any], stage: str, exc: Exception) -> Dict[str, Any]:
    result["status"] = "error"
    result["metrics"] = {}
    result["similarity_metrics"] = None
    result["retrieval_metrics"] = None
    result["classification_metrics"] = None
    result["binary_codes"] = None
    result["instrumentation"] = {}
    result["error"] = {
        "stage": stage,
        "message": str(exc),
        "exception_type": exc.__class__.__name__,
    }
    return result


def _log_result_payload(result: Dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": result["version"],
        "status": result["status"],
        "encoder_name": result["encoder_name"],
        "dataset_name": result["dataset_name"],
        "dataset_format": result["dataset_format"],
        "code_bits": result["code_bits"],
        "projection_type": result["projection_type"],
        "projection_seed": result["projection_seed"],
        "runner": result["runner"],
        "seed": result["seed"],
        "git_hash": result["git_hash"],
        "metrics": result["metrics"],
        "metrics_requested": list(result["metrics_requested"]),
        "hypotheses": result["hypotheses"],
        "system": result["system"],
        "normalization": result["normalization"],
        "instrumentation": result["instrumentation"],
        "retrieval_k": result["retrieval_k"],
        "config_fingerprint": result["config_fingerprint"],
        "similarity_metrics": result["similarity_metrics"],
        "retrieval_metrics": result["retrieval_metrics"],
        "classification_metrics": result["classification_metrics"],
        "binary_codes": result["binary_codes"],
        "error": result["error"],
    }
    logging_utils.write_log_entry(directory, _sanitize_for_logging(payload))


def _finalize_error(
    result: Dict[str, Any],
    stage: str,
    exc: Exception,
    *,
    config_dir: Path,
    output_dir_hint: Optional[str],
) -> Dict[str, Any]:
    _apply_error(result, stage, exc)
    output_dir = _resolve_output_dir(
        config_dir,
        output_dir_hint or result.get("output_dir"),
        fallback_subdir=DEFAULT_ERROR_OUTPUT_SUBDIR,
    )
    result["output_dir"] = output_dir.as_posix()
    try:
        _log_result_payload(result, output_dir)
    except Exception:
        # Logging failures at this point would cause a loop; surface best-effort result.
        pass
    return result


def _run_phase1_experiment_impl(config_path: str | Path) -> Dict[str, Any]:
    """Internal implementation for the public Phase 1 runner."""
    config_dir = Path(config_path).resolve().parent
    result = _initialize_result(DEFAULT_RUNNER_NAME)
    try:
        config = config_utils.load_config(config_path)
    except Exception as exc:  # config schema failures
        stage = "config_load"
        message = str(exc)
        if isinstance(exc, ValueError):
            if "projection_type must be one of" in message:
                raise
            if "missing required field 'seed'" in message:
                stage = "seed_extraction"
        return _finalize_error(result, stage, exc, config_dir=config_dir, output_dir_hint=None)

    config_data = dict(config)
    result["runner"] = config_data.get("runner", DEFAULT_RUNNER_NAME)
    result["encoder_name"] = config_data.get("encoder_name")
    result["dataset_name"] = config_data.get("dataset_name")
    result["dataset_format"] = config_data.get("dataset_format", "npy")
    result["code_bits"] = config_data.get("code_bits")
    projection_type = _validate_projection_type(config_data.get("projection_type"))
    result["projection_type"] = projection_type
    result["hypotheses"] = list(config_data.get("hypotheses", []))
    result["config_fingerprint"] = config.fingerprint

    try:
        metrics_requested = tuple(_metrics_from_config(config_data))
        result["metrics_requested"] = metrics_requested
    except ValueError as exc:
        message = str(exc)
        if "unsupported" in message and "metric" in message:
            raise
        return _finalize_error(
            result,
            "config_load",
            exc,
            config_dir=config_dir,
            output_dir_hint=config_data.get("output_dir"),
        )
    except Exception as exc:
        return _finalize_error(
            result,
            "config_load",
            exc,
            config_dir=config_dir,
            output_dir_hint=config_data.get("output_dir"),
        )

    try:
        seed_value = int(config_data["seed"])
    except Exception as exc:
        return _finalize_error(result, "seed_extraction", exc, config_dir=config_dir, output_dir_hint=config_data.get("output_dir"))
    result["seed"] = seed_value
    seed_utils.set_global_seed(seed_value)

    try:
        dataset_spec = dataset_catalog.get_dataset_spec(config_data["dataset_name"])
        encoder_spec = dataset_catalog.get_encoder_spec(config_data["encoder_name"])
    except (dataset_catalog.UnknownDatasetError, dataset_catalog.UnknownEncoderError):
        raise
    except Exception as exc:
        return _finalize_error(result, "catalog_lookup", exc, config_dir=config_dir, output_dir_hint=config_data.get("output_dir"))

    result["normalization"] = {"l2": dataset_spec.requires_l2_normalization}

    output_dir = _resolve_output_dir(
        config_dir,
        config_data.get("output_dir"),
        fallback_subdir=DEFAULT_SUCCESS_OUTPUT_SUBDIR,
    )
    result["output_dir"] = output_dir.as_posix()

    try:
        retrieval_k = int(config_data.get("retrieval_k", 3))
    except Exception as exc:
        return _finalize_error(result, "config_load", exc, config_dir=config_dir, output_dir_hint=result["output_dir"])
    result["retrieval_k"] = retrieval_k

    try:
        embeddings = _load_embeddings(
            config_data.get("embedding_files", []),
            encoder_spec.embedding_dim,
            config_dir,
        )
        labels = _load_classification_labels(
            config_data.get("classification_labels"),
            embeddings.shape[0],
            config_dir,
        )
    except Exception as exc:
        return _finalize_error(result, "load_embeddings", exc, config_dir=config_dir, output_dir_hint=result["output_dir"])

    try:
        engine = BinaryEmbeddingEngine(
            encoder_spec=encoder_spec,
            dataset_spec=dataset_spec,
            code_bits=int(config_data["code_bits"]),
            projection_type=config_data["projection_type"],
            seed=seed_value,
            normalize=dataset_spec.requires_l2_normalization,
            projection_seed=int(config_data.get("projection_seed", seed_value)),
        )
    except ValueError as exc:
        if "projection_type" in str(exc):
            raise
        return _finalize_error(result, "engine_init", exc, config_dir=config_dir, output_dir_hint=result["output_dir"])
    except Exception as exc:
        return _finalize_error(result, "engine_init", exc, config_dir=config_dir, output_dir_hint=result["output_dir"])

    try:
        engine_result = engine.run(
            embeddings,
            metrics=metrics_requested,
            retrieval_k=retrieval_k,
            classification_labels=labels,
            return_full_code_bits=True,
        )
    except Exception as exc:
        return _finalize_error(result, "engine_run", exc, config_dir=config_dir, output_dir_hint=result["output_dir"])

    result.update(
        {
            "status": "success",
            "encoder_name": engine_result["encoder_name"],
            "dataset_name": engine_result["dataset_name"],
            "code_bits": engine_result["code_bits"],
            "projection_type": engine_result["projection_type"],
            "projection_seed": engine_result["projection_seed"],
            "seed": engine_result["seed"],
            "metrics": engine_result["metrics"],
            "metrics_requested": tuple(engine_result["metrics_requested"]),
            "similarity_metrics": engine_result["similarity_metrics"],
            "retrieval_metrics": engine_result["retrieval_metrics"],
            "classification_metrics": engine_result["classification_metrics"],
            "normalization": engine_result["normalization"],
            "instrumentation": engine_result["instrumentation"],
            "binary_codes": engine_result["binary_codes"],
            "error": None,
        }
    )

    try:
        _log_result_payload(result, output_dir)
    except Exception as exc:
        return _finalize_error(result, "logging", exc, config_dir=config_dir, output_dir_hint=output_dir.as_posix())

    return result


def run_phase1_experiment(config_path: str | Path) -> Dict[str, Any]:
    """Main entrypoint for Phase 1 experiments."""
    return _run_phase1_experiment_impl(config_path)

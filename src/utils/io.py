"""IO utilities for embeddings and metadata loading in Phase 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


def save_embeddings_npy(path: Path, embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings.astype(np.float32), allow_pickle=False)


def load_embeddings_npy(path: Path, *, expected_dim: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"embedding file not found: {path}")
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.float32:
        raise ValueError("embeddings dtype must be float32 for IO utilities")
    if array.ndim != 2 or array.shape[1] != expected_dim:
        raise ValueError("embedding_dim mismatch in config/IO")
    return array


def load_embeddings_parquet(path: Path, *, expected_dim: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"embedding parquet file not found: {path}")
    frame = pd.read_parquet(path)
    array = frame.to_numpy(dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != expected_dim:
        raise ValueError("embedding_dim mismatch in parquet embeddings")
    return array


def load_metadata_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"metadata file not found: {path}")
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


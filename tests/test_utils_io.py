from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils import io as io_utils

FIXTURE_DIR = Path(__file__).resolve().parent / "data" / "phase1_synthetic"


def _fixture_embeddings() -> np.ndarray:
    data = json.loads((FIXTURE_DIR / "t8_embeddings.json").read_text())
    return np.asarray(data, dtype=np.float32)


def test_save_and_load_embeddings_npy_round_trip(tmp_path: Path) -> None:
    embeddings = _fixture_embeddings()
    npy_path = tmp_path / "embeddings.npy"
    io_utils.save_embeddings_npy(npy_path, embeddings)
    loaded = io_utils.load_embeddings_npy(npy_path, expected_dim=embeddings.shape[1])
    np.testing.assert_allclose(loaded, embeddings)


def test_load_embeddings_npy_validates_shape_and_dtype(tmp_path: Path) -> None:
    embeddings = _fixture_embeddings().astype(np.int32)
    bad_path = tmp_path / "bad.npy"
    np.save(bad_path, embeddings)
    with pytest.raises(ValueError, match="dtype"):
        io_utils.load_embeddings_npy(bad_path, expected_dim=4)

    float_path = tmp_path / "float.npy"
    arr = _fixture_embeddings()
    np.save(float_path, arr[:, :3])
    with pytest.raises(ValueError, match="embedding_dim"):
        io_utils.load_embeddings_npy(float_path, expected_dim=4)


def test_load_embeddings_parquet_round_trip(tmp_path: Path) -> None:
    embeddings = _fixture_embeddings()
    parquet_path = tmp_path / "embeddings.parquet"
    df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings.shape[1])])
    df.to_parquet(parquet_path, index=False)
    loaded = io_utils.load_embeddings_parquet(parquet_path, expected_dim=embeddings.shape[1])
    np.testing.assert_allclose(loaded, embeddings)


def test_load_embeddings_parquet_validates_files() -> None:
    with pytest.raises(FileNotFoundError):
        io_utils.load_embeddings_parquet(Path("missing.parquet"), expected_dim=4)


def test_load_metadata_jsonl_reads_records(tmp_path: Path) -> None:
    metadata_path = tmp_path / "meta.jsonl"
    metadata_path.write_text((FIXTURE_DIR / "t8_metadata.jsonl").read_text())
    records = io_utils.load_metadata_jsonl(metadata_path)
    assert len(records) == 3
    assert records[0]["id"] == "doc-0"


"""Contract tests for FloatEmbeddingBatch and BinaryCodeBatch abstractions.

Assumptions:
- Embedding batches use numpy arrays internally (or array-likes convertible
  to numpy) with 2D shape (N, d).
- Validation occurs at construction time; failures raise ValueError with
  messages referencing the problematic field and the embeddings stage.
"""

import numpy as np
import pytest

from src.core.embeddings import BinaryCodeBatch, FloatEmbeddingBatch


def test_float_embedding_batch_construction_success():
    data = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    batch = FloatEmbeddingBatch(
        data=data,
        embedding_dim=4,
        encoder_name="synthetic_encoder_4d",
        dataset_name="phase1_synthetic_toy",
        ids=["a", "b"],
        normalized=True,
    )

    assert batch.embedding_dim == 4
    assert batch.encoder_name == "synthetic_encoder_4d"
    assert batch.dataset_name == "phase1_synthetic_toy"
    assert batch.normalized is True
    assert batch.ids == ["a", "b"]
    assert np.array_equal(batch.data, data)


def test_float_embedding_batch_requires_2d_data():
    data = np.ones((4,), dtype=np.float32)
    # Error message must mention the embeddings stage and that data is 2D.
    with pytest.raises(ValueError, match="embeddings.*2D"):
        FloatEmbeddingBatch(
            data=data,
            embedding_dim=4,
            encoder_name="enc",
            dataset_name="ds",
            normalized=False,
        )


def test_float_embedding_batch_validates_embedding_dim():
    data = np.ones((2, 3), dtype=np.float32)
    # Error message must contain both 'embedding_dim' and 'embeddings'
    # so logs clearly identify the failing field and stage.
    with pytest.raises(ValueError, match="embedding_dim.*embeddings"):
        FloatEmbeddingBatch(
            data=data,
            embedding_dim=4,
            encoder_name="enc",
            dataset_name="ds",
            normalized=False,
        )


def test_float_embedding_batch_embedding_dim_positive():
    data = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="embedding_dim.*positive"):
        FloatEmbeddingBatch(
            data=data,
            embedding_dim=0,
            encoder_name="enc",
            dataset_name="ds",
            normalized=False,
        )


def test_float_embedding_batch_normalized_flag_checks_norm():
    data = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    # Normalization invariant: when normalized=True, vectors must be ~unit norm;
    # message must mention 'normalized' and 'embeddings' for debugging.
    with pytest.raises(ValueError, match="normalized.*embeddings"):
        FloatEmbeddingBatch(
            data=data,
            embedding_dim=2,
            encoder_name="enc",
            dataset_name="ds",
            normalized=True,
        )


def test_float_embedding_batch_ids_length_matches_samples():
    data = np.ones((3, 2), dtype=np.float32)
    # IDs length mismatch must be surfaced with 'ids' and 'embeddings'
    # in the error message.
    with pytest.raises(ValueError, match="ids.*embeddings"):
        FloatEmbeddingBatch(
            data=data,
            embedding_dim=2,
            encoder_name="enc",
            dataset_name="ds",
            ids=["a", "b"],
            normalized=False,
        )


def test_binary_code_batch_construction_success():
    codes_pm1 = np.array([[1, -1], [-1, 1]], dtype=np.int8)
    codes_01 = np.array([[1, 0], [0, 1]], dtype=np.int8)
    batch = BinaryCodeBatch(
        codes_pm1=codes_pm1,
        codes_01=codes_01,
        code_bits=2,
        encoder_name="enc",
        dataset_name="ds",
    )

    assert batch.code_bits == 2
    assert np.array_equal(batch.codes_pm1, codes_pm1)
    assert np.array_equal(batch.codes_01, codes_01)


def test_binary_code_batch_requires_matching_shapes():
    codes_pm1 = np.ones((2, 2), dtype=np.int8)
    codes_01 = np.ones((3, 2), dtype=np.int8)
    with pytest.raises(ValueError, match="codes_pm1.*shape"):
        BinaryCodeBatch(
            codes_pm1=codes_pm1,
            codes_01=codes_01,
            code_bits=2,
            encoder_name="enc",
            dataset_name="ds",
        )


def test_binary_code_batch_validates_pm1_values():
    codes_pm1 = np.array([[1, 0]], dtype=np.int8)
    codes_01 = np.array([[1, 0]], dtype=np.int8)
    # Diagnostic contract: message must mention 'codes_pm1' and the literal set '{-1,+1}'.
    with pytest.raises(ValueError, match=r"codes_pm1.*\{-1,\+1\}"):
        BinaryCodeBatch(
            codes_pm1=codes_pm1,
            codes_01=codes_01,
            code_bits=2,
            encoder_name="enc",
            dataset_name="ds",
        )


def test_binary_code_batch_validates_01_values():
    codes_pm1 = np.array([[1, -1]], dtype=np.int8)
    codes_01 = np.array([[2, 0]], dtype=np.int8)
    # Error must flag the field name and the allowed set literal {0,1}.
    with pytest.raises(ValueError, match=r"codes_01.*\{0,1\}"):
        BinaryCodeBatch(
            codes_pm1=codes_pm1,
            codes_01=codes_01,
            code_bits=2,
            encoder_name="enc",
            dataset_name="ds",
        )


def test_binary_code_batch_code_bits_matches_last_dim():
    codes_pm1 = np.array([[1, -1, 1]], dtype=np.int8)
    codes_01 = np.array([[1, 0, 1]], dtype=np.int8)
    with pytest.raises(ValueError, match="code_bits"):
        BinaryCodeBatch(
            codes_pm1=codes_pm1,
            codes_01=codes_01,
            code_bits=2,
            encoder_name="enc",
            dataset_name="ds",
        )


def test_binary_code_batch_rejects_non_2d_inputs():
    codes_pm1 = np.array([1, -1], dtype=np.int8)
    codes_01 = np.array([1, 0], dtype=np.int8)
    with pytest.raises(ValueError, match="codes_pm1.*2D"):
        BinaryCodeBatch(
            codes_pm1=codes_pm1,
            codes_01=codes_01,
            code_bits=2,
            encoder_name="enc",
            dataset_name="ds",
        )


def test_embedding_batches_are_deterministic():
    data = np.eye(2, dtype=np.float32)
    batch_a = FloatEmbeddingBatch(
        data=data,
        embedding_dim=2,
        encoder_name="enc",
        dataset_name="ds",
        normalized=True,
    )
    batch_b = FloatEmbeddingBatch(
        data=data,
        embedding_dim=2,
        encoder_name="enc",
        dataset_name="ds",
        normalized=True,
    )
    assert np.array_equal(batch_a.data, batch_b.data)


def test_binary_code_batches_are_deterministic():
    codes_pm1 = np.array([[1, -1]], dtype=np.int8)
    codes_01 = np.array([[1, 0]], dtype=np.int8)
    batch_a = BinaryCodeBatch(
        codes_pm1=codes_pm1,
        codes_01=codes_01,
        code_bits=2,
        encoder_name="enc",
        dataset_name="ds",
    )
    batch_b = BinaryCodeBatch(
        codes_pm1=codes_pm1,
        codes_01=codes_01,
        code_bits=2,
        encoder_name="enc",
        dataset_name="ds",
    )
    assert np.array_equal(batch_a.codes_pm1, batch_b.codes_pm1)
    assert np.array_equal(batch_a.codes_01, batch_b.codes_01)


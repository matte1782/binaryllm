"""Contract tests for LSB-first uint64 bit packing in Phase 1 T4.

Assumptions enforced here follow `binaryllm_phase1_architecture_v1.md`:
- `pack_codes(codes_01)` accepts uint8 {0,1} matrices with shape (N, m).
- Bits are packed row-major into 64-bit words using LSB-first layout.
- `unpack_codes(packed, code_bits)` reverses the operation exactly.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from src.quantization.packing import pack_codes, unpack_codes


DATA_DIR = Path(__file__).resolve().parent / "data" / "phase1_synthetic"
SNAPSHOT_FILE = DATA_DIR / "packing_roundtrip_snapshot.npy"


def _ceil64(bits: int) -> int:
    return (bits + 63) // 64


@pytest.fixture(scope="module")
def packing_snapshot() -> tuple[np.ndarray, np.ndarray]:
    """Load the regression snapshot (codes + expected packed words)."""
    data = np.load(SNAPSHOT_FILE, allow_pickle=True).item()
    return data["codes_01"], data["packed"]


@pytest.mark.parametrize(
    "shape",
    [
        (1, 7),
        (3, 64),
        (2, 128),
        (5, 77),
    ],
)
def test_pack_unpack_roundtrip_preserves_bits(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(42 + shape[0] * shape[1])
    codes = rng.integers(0, 2, size=shape, dtype=np.uint8)

    packed = pack_codes(codes)
    unpacked = unpack_codes(packed, shape[1])

    assert packed.dtype == np.uint64
    assert unpacked.dtype == np.uint8
    assert packed.shape == (shape[0], _ceil64(shape[1]))
    np.testing.assert_array_equal(unpacked, codes)


@pytest.mark.parametrize(
    "codes",
    [
        np.zeros((1, 1), dtype=np.uint8),  # single bit
        np.zeros((4, 65), dtype=np.uint8),  # all zeros across word boundary
        np.ones((3, 64), dtype=np.uint8),  # all ones exactly one word
        np.tile(np.array([[0, 1]], dtype=np.uint8), (2, 32)),  # alternating 01
        np.tile(np.array([[1, 0]], dtype=np.uint8), (3, 33)),  # alternating 10
        np.array([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.uint8),  # single row alternating
    ],
)
def test_pack_unpack_edge_cases(codes: np.ndarray) -> None:
    packed = pack_codes(codes)
    unpacked = unpack_codes(packed, codes.shape[1])
    np.testing.assert_array_equal(unpacked, codes)


def test_pack_codes_is_deterministic() -> None:
    codes = np.tile(np.array([[1, 0, 0, 1, 1, 0, 1, 0]], dtype=np.uint8), (4, 16))
    baseline = codes.copy()
    first = pack_codes(codes)
    second = pack_codes(codes.copy())
    assert np.array_equal(first, second)
    np.testing.assert_array_equal(codes, baseline)


def test_pack_codes_bit_layout_single_word() -> None:
    codes = np.array([[1, 0, 1, 1, 0, 0, 1, 0]], dtype=np.uint8)
    expected_value = (1 << 0) + (1 << 2) + (1 << 3) + (1 << 6)
    expected = np.array([[expected_value]], dtype=np.uint64)

    packed = pack_codes(codes)
    np.testing.assert_array_equal(packed, expected)
    np.testing.assert_array_equal(unpack_codes(expected, codes.shape[1]), codes)


def test_pack_codes_bit_layout_multi_word() -> None:
    codes = np.zeros((1, 130), dtype=np.uint8)
    codes[0, 0] = 1
    codes[0, 63] = 1
    codes[0, 64] = 1
    codes[0, 129] = 1

    word0 = (np.uint64(1) << np.uint64(0)) | (np.uint64(1) << np.uint64(63))
    word1 = np.uint64(1) << np.uint64(0)
    word2 = np.uint64(1) << np.uint64(1)
    expected = np.array([[word0, word1, word2]], dtype=np.uint64)

    packed = pack_codes(codes)
    np.testing.assert_array_equal(packed, expected)
    unpacked = unpack_codes(expected, codes.shape[1])
    assert unpacked.shape == (1, 130)
    np.testing.assert_array_equal(unpacked[:, [0, 63, 64, 129]], 1)
    np.testing.assert_array_equal(unpacked, codes)


def test_pack_codes_rejects_invalid_values() -> None:
    invalid = np.array([[0, 2]], dtype=np.int8)
    with pytest.raises(ValueError, match=r"codes_01.*\{0,1\}"):
        pack_codes(invalid)


def test_pack_codes_rejects_non_2d_inputs() -> None:
    invalid = np.array([0, 1, 0], dtype=np.uint8)
    with pytest.raises(ValueError, match=r"codes_01.*\{0,1\}"):
        pack_codes(invalid)


def test_pack_codes_rejects_empty_arrays() -> None:
    empty = np.empty((0, 0), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"codes_01.*\{0,1\}"):
        pack_codes(empty)


def test_pack_codes_rejects_wrong_dtype() -> None:
    floats = np.zeros((2, 8), dtype=np.float32)
    with pytest.raises(ValueError, match=r"codes_01.*\{0,1\}"):
        pack_codes(floats)


def test_unpack_codes_rejects_non_positive_code_bits() -> None:
    packed = np.zeros((1, 1), dtype=np.uint64)
    with pytest.raises(ValueError, match="code_bits.*positive"):
        unpack_codes(packed, 0)


def test_unpack_codes_requires_uint64_matrix() -> None:
    packed = np.zeros((2, 1), dtype=np.int64)
    with pytest.raises(ValueError, match="packed.*uint64"):
        unpack_codes(packed, 1)


def test_unpack_codes_shape_mismatch_detection() -> None:
    packed = np.zeros((1, 1), dtype=np.uint64)
    with pytest.raises(ValueError, match="packed.*code_bits"):
        unpack_codes(packed, 65)


def test_pack_unpack_performance_sanity() -> None:
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 2, size=(64, 256), dtype=np.uint8)
    start = time.perf_counter()
    packed = pack_codes(codes)
    unpacked = unpack_codes(packed, codes.shape[1])
    duration = time.perf_counter() - start
    np.testing.assert_array_equal(unpacked, codes)
    assert duration < 0.1, f"pack/unpack exceeded 0.1s (took {duration:.4f}s)"


def test_pack_codes_matches_snapshot(packing_snapshot: tuple[np.ndarray, np.ndarray]) -> None:
    codes, expected_packed = packing_snapshot
    packed = pack_codes(codes)
    np.testing.assert_array_equal(packed, expected_packed)
    np.testing.assert_array_equal(unpack_codes(expected_packed, codes.shape[1]), codes)


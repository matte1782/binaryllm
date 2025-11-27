"""Bit packing utilities for BinaryLLM Phase 1.

Packing uses an LSB-first, row-major layout:
- For sample i and bit index j, with code bits in {0,1},
  the packed word index is w = j // 64, and bit position b = j % 64.
- Packed words are uint64; unused high bits stay zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _validate_codes_01(codes_01: Any) -> np.ndarray:
    """Return a uint8 view of codes_01 after enforcing T4 invariants."""
    array = np.asarray(codes_01)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("codes_01 must be a non-empty 2D array containing only {0,1}.")
    if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_)):
        raise ValueError("codes_01 must be a non-empty 2D array containing only {0,1}.")
    if not np.isin(array, (0, 1)).all():
        raise ValueError("codes_01 must be a non-empty 2D array containing only {0,1}.")
    return array.astype(np.uint8, copy=False)


def pack_codes(codes_01: np.ndarray) -> np.ndarray:
    """Pack {0,1} codes into uint64 words with deterministic LSB-first layout."""
    codes = _validate_codes_01(codes_01)
    n_rows, code_bits = codes.shape
    if code_bits <= 0:
        raise ValueError("codes_01 must be a non-empty 2D array containing only {0,1}.")

    num_words = (code_bits + 63) // 64
    packed = np.zeros((n_rows, num_words), dtype=np.uint64)

    for bit in range(code_bits):
        word_index = bit // 64
        bit_index = bit % 64
        mask = np.uint64(1) << np.uint64(bit_index)
        row_bits = codes[:, bit].astype(np.uint64)
        packed[:, word_index] |= row_bits * mask

    return packed


def unpack_codes(packed: np.ndarray, code_bits: int) -> np.ndarray:
    """Unpack uint64 words back into {0,1} codes using LSB-first layout."""
    if code_bits <= 0:
        raise ValueError("code_bits must be positive for unpacking.")

    packed_array = np.asarray(packed)
    if packed_array.ndim != 2 or packed_array.dtype != np.uint64:
        raise ValueError("packed must be a 2D array with dtype=uint64.")

    n_rows, num_words = packed_array.shape
    expected_words = (code_bits + 63) // 64
    if num_words != expected_words:
        raise ValueError("packed shape incompatible with code_bits.")

    codes = np.zeros((n_rows, code_bits), dtype=np.uint8)

    for bit in range(code_bits):
        word_index = bit // 64
        bit_index = bit % 64
        bits = (packed_array[:, word_index] >> np.uint64(bit_index)) & np.uint64(1)
        codes[:, bit] = bits.astype(np.uint8)

    return codes


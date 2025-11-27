"""
Quantization pipeline for BinaryLLM Phase 1.

This module implements the binary transformation pipeline:

Binarization (src.quantization.binarization):
    - RandomProjection: Gaussian random projection with deterministic weights
    - binarize_sign: Sign-based binarization (x >= 0 -> +1, x < 0 -> -1)
    - project_and_binarize: Combined projection and binarization

Packing (src.quantization.packing):
    - pack_codes: Pack {0,1} codes into uint64 words (LSB-first layout)
    - unpack_codes: Unpack uint64 words to logical bit arrays

The packing format uses LSB-first, row-major layout optimized for
XNOR-popcount GPU kernels in future phases.
"""

from src.quantization.binarization import (
    RandomProjection,
    binarize_sign,
    project_and_binarize,
)
from src.quantization.packing import pack_codes, unpack_codes

__all__ = [
    "RandomProjection",
    "binarize_sign",
    "project_and_binarize",
    "pack_codes",
    "unpack_codes",
]

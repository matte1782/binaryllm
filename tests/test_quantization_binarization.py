"""Tests for src.quantization.binarization projection + binarization utilities."""

import numpy as np
import pytest

from src.quantization import binarization


def test_binarize_sign_basic():
    x = np.array([-2.0, -0.0, 0.0, 3.5], dtype=np.float32)
    result = binarization.binarize_sign(x)
    assert result.dtype == np.float32
    assert result.shape == x.shape
    assert np.array_equal(result, np.array([-1.0, 1.0, 1.0, 1.0], dtype=np.float32))


def test_random_projection_deterministic():
    proj_a = binarization.RandomProjection(input_dim=4, output_bits=8, seed=42)
    proj_b = binarization.RandomProjection(input_dim=4, output_bits=8, seed=42)
    proj_c = binarization.RandomProjection(input_dim=4, output_bits=8, seed=7)

    x = np.ones((2, 4), dtype=np.float32)
    out_a = proj_a.project(x)
    out_b = proj_b.project(x)
    out_c = proj_c.project(x)

    assert np.array_equal(out_a, out_b)
    assert not np.array_equal(out_a, out_c)


def test_random_projection_shape_and_type():
    proj = binarization.RandomProjection(input_dim=4, output_bits=6, seed=1)
    x = np.zeros((3, 4), dtype=np.float32)
    out = proj.project(x)
    assert out.shape == (3, 6)
    assert out.dtype == np.float32


def test_project_and_binarize_behavior():
    proj = binarization.RandomProjection(input_dim=2, output_bits=8, seed=0)
    x = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    codes = binarization.project_and_binarize(x, proj)
    assert codes.shape == (2, 8)
    assert np.isin(codes, (-1.0, 1.0)).all()
    codes_repeat = binarization.project_and_binarize(x, proj)
    assert np.array_equal(codes, codes_repeat)


def test_hamming_matches_angle_sanity():
    angle = np.pi / 4  # 45 degrees
    u = np.array([[1.0, 0.0]], dtype=np.float32)
    v = np.array([[np.cos(angle), np.sin(angle)]], dtype=np.float32)
    u_norm = u / np.linalg.norm(u)
    v_norm = v / np.linalg.norm(v)
    x = np.vstack([u_norm, v_norm])

    proj = binarization.RandomProjection(input_dim=2, output_bits=2048, seed=123)
    codes = binarization.project_and_binarize(x, proj)
    disagreements = (codes[0] != codes[1]).sum()
    frac = disagreements / codes.shape[1]

    assert abs(frac - (angle / np.pi)) < 0.05


def test_random_projection_dimension_mismatch_raises():
    proj = binarization.RandomProjection(input_dim=3, output_bits=4, seed=0)
    x = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="projection.*dim"):
        proj.project(x)


def test_random_projection_handles_empty_batch():
    proj = binarization.RandomProjection(input_dim=3, output_bits=5, seed=0)
    x = np.zeros((0, 3), dtype=np.float32)
    out = proj.project(x)
    assert out.shape == (0, 5)


def test_random_projection_non_finite_inputs_raise():
    proj = binarization.RandomProjection(input_dim=2, output_bits=4, seed=0)
    x = np.array([[np.nan, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        proj.project(x)


# These tests define the projection+binarization contract:
# - sign convention (-1 for negative, +1 for zero/positive)
# - deterministic random projections with seed control
# - project_and_binarize outputs {-1,+1} codes
# - Hamming distance tracks angle within tolerance
# - error handling for dimension mismatch and non-finite inputs


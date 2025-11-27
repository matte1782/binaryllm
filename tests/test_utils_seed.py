from __future__ import annotations

import random

import numpy as np
import pytest

from src.utils import seed as seed_utils

try:
    import torch

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    _TORCH_AVAILABLE = False


def test_set_global_seed_creates_deterministic_streams() -> None:
    seed_utils.set_global_seed(1234)
    python_first = random.randint(0, 10_000)
    numpy_first = np.random.rand()
    torch_first = torch.rand(1).item() if _TORCH_AVAILABLE else None

    seed_utils.set_global_seed(1234)
    assert python_first == random.randint(0, 10_000)
    assert numpy_first == np.random.rand()
    if _TORCH_AVAILABLE:
        assert torch_first == torch.rand(1).item()


def test_set_global_seed_requires_integer() -> None:
    with pytest.raises(ValueError, match="seed.*integer"):
        seed_utils.set_global_seed("abc")  # type: ignore[arg-type]



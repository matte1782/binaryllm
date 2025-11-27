"""Global seed handling that synchronizes Python, NumPy, and Torch (if installed)."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    _TORCH_AVAILABLE = False


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if installed) Torch RNGs."""
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer for deterministic runs")

    random.seed(seed)
    np.random.seed(seed)

    if _TORCH_AVAILABLE:  # pragma: no cover - depends on env
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)


# Backwards-compatible alias -------------------------------------------------
set_seed = set_global_seed


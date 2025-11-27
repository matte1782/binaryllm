"""
Utility modules for BinaryLLM Phase 1.

This module provides infrastructure utilities:

Configuration (src.utils.config):
    - load_config: Load and validate YAML/JSON configs
    - FrozenConfig: Immutable config with SHA-256 fingerprint

Logging (src.utils.logging):
    - write_log_entry: Structured JSON logging with Schema v2

Seeding (src.utils.seed):
    - set_global_seed: Set Python, NumPy, and Torch RNGs

I/O (src.utils.io):
    - File loading and path normalization utilities
"""

from src.utils.config import load_config, FrozenConfig
from src.utils.logging import write_log_entry
from src.utils.seed import set_global_seed

__all__ = [
    "load_config",
    "FrozenConfig",
    "write_log_entry",
    "set_global_seed",
]

"""Smoke test guarding the T1 directory skeleton integrity."""

import importlib


def test_package_imports():
    """Ensure all Phase 1 packages import cleanly."""
    module_names = [
        "src",
        "src.core",
        "src.quantization",
        "src.eval",
        "src.utils",
        "src.experiments",
        "src.variants",
        "src.experiments.configs",
        "src.experiments.runners",
    ]
    for name in module_names:
        importlib.import_module(name)



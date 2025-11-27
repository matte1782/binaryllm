"""
Experiment runners for BinaryLLM.

Available Runners:
    phase1_binary_embeddings: Main Phase 1 experiment runner
        - Config-driven execution
        - Deterministic seeding
        - Structured JSON logging
        - Full error pipeline

Usage:
    >>> from src.experiments.runners.phase1_binary_embeddings import run_phase1_experiment
    >>> result = run_phase1_experiment("path/to/config.yaml")
"""

from src.experiments.runners.phase1_binary_embeddings import run_phase1_experiment

__all__ = ["run_phase1_experiment"]

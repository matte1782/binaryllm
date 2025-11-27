"""
Experiment infrastructure for BinaryLLM.

This module provides experiment runners and configuration management.

Runners (src.experiments.runners):
    - phase1_binary_embeddings: Main Phase 1 experiment runner
      Handles config loading, seeding, I/O, and structured logging.

Usage:
    # From command line:
    python -m src.experiments.runners.phase1_binary_embeddings --config path/to/config.yaml
    
    # Programmatically:
    >>> from src.experiments.runners.phase1_binary_embeddings import run_phase1_experiment
    >>> result = run_phase1_experiment("path/to/config.yaml")
"""

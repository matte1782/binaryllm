"""
BinaryLLM — Towards 1-Bit Latent Spaces for Efficient Large Language Models.

This package implements Phase 1 of the BinaryLLM research program: a deterministic
binary embedding engine that converts float embeddings to binary codes while
preserving semantic neighborhood structure.

Modules:
    core: Dataset catalog, embedding containers, and data abstractions
    quantization: Random projection, sign binarization, and bit packing
    eval: Similarity, retrieval, and classification evaluation metrics
    experiments: Experiment runners and configuration management
    variants: Engine implementations (BinaryEmbeddingEngine façade)
    utils: Configuration, logging, seeding, and I/O utilities

Quick Start:
    >>> from src.experiments.runners.phase1_binary_embeddings import run_phase1_experiment
    >>> result = run_phase1_experiment("path/to/config.yaml")

For detailed documentation, see README.md and docs/TECHNICAL_REFERENCE.md.
"""

__version__ = "1.0.0"
__author__ = "Matteo Panzeri"
__email__ = "matteo1782@gmail.com"
__license__ = "MIT"

"""
Engine variants for BinaryLLM.

This module provides façade implementations that orchestrate the
binary embedding pipeline. Currently Phase 1 only.

BinaryEmbeddingEngine (src.variants.binary_embedding_engine):
    Pure in-memory façade that executes:
    1. L2 normalization of float embeddings
    2. Gaussian random projection
    3. Sign binarization ({-1,+1} and {0,1} forms)
    4. Bit packing to uint64
    5. Similarity, retrieval, and classification evaluation

Example:
    >>> from src.variants.binary_embedding_engine import BinaryEmbeddingEngine
    >>> engine = BinaryEmbeddingEngine(encoder_spec, dataset_spec, code_bits=64, ...)
    >>> result = engine.run(embeddings, metrics=["similarity", "retrieval"])
"""

from src.variants.binary_embedding_engine import BinaryEmbeddingEngine

__all__ = ["BinaryEmbeddingEngine"]

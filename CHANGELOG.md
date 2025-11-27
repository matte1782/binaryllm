# Changelog

All notable changes to BinaryLLM are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2025-11-27

### Phase 1: Binary Embedding Engine — Stability Seal

This release marks the completion and freezing of Phase 1. All 157 tests pass, 
schemas are stable, and golden regression artifacts are locked.

### Added

#### Core Components
- **BinaryEmbeddingEngine** (`src/variants/binary_embedding_engine.py`)
  - Pure in-memory façade for the binary embedding pipeline
  - L2 normalization, Gaussian random projection, sign binarization
  - Bit packing to uint64 with LSB-first layout
  - Instrumentation hooks for timing measurements

- **Experiment Runner** (`src/experiments/runners/phase1_binary_embeddings.py`)
  - YAML/JSON configuration loading with strict validation
  - Deterministic seed management
  - Structured error pipeline with named stages
  - JSON logging with Schema v2 compliance

#### Quantization Module
- `binarize_sign()` — Sign-based binarization with x ≥ 0 → +1 convention
- `RandomProjection` — Deterministic Gaussian projection matrix
- `pack_codes()` / `unpack_codes()` — LSB-first bit packing

#### Evaluation Metrics
- **Similarity**: Cosine/Hamming matrices, Spearman correlation, neighbor overlap
- **Retrieval**: Top-k indices, nDCG@k, recall@k with deterministic tie-breaking
- **Classification**: Centroid classifier with float vs binary accuracy comparison

#### Dataset Infrastructure
- `DatasetSpec` / `EncoderSpec` dataclasses
- Dataset catalog with registration and lookup
- Embedding containers with validation (`FloatEmbeddingBatch`, `BinaryCodeBatch`)

#### Configuration System
- YAML/JSON config loader with `FrozenConfig`
- SHA-256 fingerprinting for reproducibility
- Strict validation: no unknown keys, supported projections only

#### Logging System
- Structured JSON logging (Schema v2)
- System metadata capture (hostname, platform, CPU, GPU)
- Timestamped entries with full audit trail

#### Test Suite
- **157 tests** covering all modules
- Golden regression tests with frozen artifacts
- Determinism tests ensuring reproducibility
- Structural integrity tests

### Technical Specifications

| Specification | Value |
|--------------|-------|
| Schema Version | `phase1-v2` |
| Supported Projections | `gaussian` |
| Supported Code Bits | 32, 64, 128, 256 |
| Bit Layout | LSB-first, row-major uint64 |
| Sign Convention | x ≥ 0 → +1, x < 0 → -1 |

### Invariants Established

- **INV-01**: projection_type ∈ {"gaussian"} (Phase 1 only)
- **INV-02**: Runner performs no numerical operations
- **INV-03**: Façade performs no I/O
- **INV-04**: Same seed + inputs = identical outputs
- **INV-05**: Classification accuracy_delta < 0 on golden dataset

### Known Limitations

- Only Gaussian random projection supported (Rademacher reserved for Phase 2)
- No GPU acceleration (CPU-only NumPy implementation)
- Minimal CLI surface area (single --config flag)

### Documentation

- Architecture specification (`docs/architecture/`)
- Phase 1 stability seal artifacts (`docs/artifacts/`)
- Research report v2 (`docs/binaryllm_report_v2.md`)

---

## [Unreleased]

### Phase 2: Binary KV-Cache (Planned)

- Binary attention key/value representations
- XNOR-popcount attention kernels
- Memory bandwidth benchmarks

---

## Version History Summary

| Version | Date | Milestone |
|---------|------|-----------|
| 1.0.0 | 2025-11-27 | Phase 1 Stability Seal |

---

## Author

**Matteo Panzeri** — AI Bachelor Student, University of Pavia  
📧 [matteo1782@gmail.com](mailto:matteo1782@gmail.com) | [matteo.panzeri01@universitadipavia.it](mailto:matteo.panzeri01@universitadipavia.it)

---

*For detailed development history, see the git log.*


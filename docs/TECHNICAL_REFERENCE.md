# BinaryLLM Phase 1 — Technical Reference

> **Status**: Phase 1 Frozen at Stability Seal (T15)  
> **Audience**: Systems and research engineers  
> **Scope**: Complete technical specification for the binary embedding engine

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Runner](#3-runner)
4. [Façade — BinaryEmbeddingEngine](#4-façade--binaryembeddingengine)
5. [Quantization](#5-quantization)
6. [Evaluation](#6-evaluation)
7. [Configuration](#7-configuration)
8. [Logging](#8-logging)
9. [Result Schema](#9-result-schema)
10. [Error Pipeline](#10-error-pipeline)
11. [Determinism](#11-determinism)
12. [Golden Regression](#12-golden-regression)
13. [Invariants](#13-invariants)

---

## 1. Overview

### 1.1 What Phase 1 Implements

A deterministic, batch-only binary embedding and similarity engine that:

- Converts precomputed float embeddings into `{0,1}` binary codes
- Uses Gaussian random projection and sign binarization
- Packs codes into 64-bit words (LSB-first layout)
- Evaluates similarity, retrieval, and classification metrics

### 1.2 Key Invariants

| Invariant | Description |
|-----------|-------------|
| Result Schema v2 | Binding output format for all experiments |
| Logging v2 | Structured JSON with system metadata |
| Error Pipeline | Named stages for debugging |
| Determinism | Same inputs + seeds = identical outputs |

---

## 2. Architecture

### 2.1 System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Precomputed Float Embeddings                │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Phase 1 Binary Embedding Engine                     │
│  ┌─────────┐  ┌─────────┐  ┌────────────┐  ┌─────────────────┐  │
│  │ Runner  │→ │ Façade  │→ │Quantization│→ │   Evaluation    │  │
│  │ (I/O)   │  │(Pipeline)│  │(Binarize)  │  │(Sim/Ret/Class) │  │
│  └─────────┘  └─────────┘  └────────────┘  └─────────────────┘  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                   ┌────────────┴────────────┐
                   │                         │
                   ▼                         ▼
          Result Dict (RAM)          Structured Logs (JSON)
```

### 2.2 Layer Responsibilities

| Layer | Responsibilities | Key Invariants |
|-------|------------------|----------------|
| Runner | Config, seed, I/O, logging | No numerical ops |
| Façade | Normalize, project, binarize, pack, metrics | No I/O |
| Quantization | Projection, binarization, packing | Sign: x≥0→+1; LSB-first |
| Evaluation | Similarity, retrieval, classification | Deterministic ranking |
| Core | Dataset catalog, embedding containers | Metadata enforcement |
| Utilities | Config, logging, seed, IO | Schema validation |

---

## 3. Runner

**Location**: `src/experiments/runners/phase1_binary_embeddings.py`

### 3.1 Purpose

Orchestration layer that handles:
- Config ingestion and validation
- Deterministic seeding
- Dataset/encoder catalog lookup
- Embedding I/O
- Façade invocation
- Structured logging

### 3.2 Public API

```python
def run_phase1_experiment(config_path: str | Path) -> Dict[str, Any]:
    """Main entrypoint for Phase 1 experiments."""
```

### 3.3 Runner Invariants

| ID | Invariant | Enforcement |
|----|-----------|-------------|
| R-01 | No numerical ops | Structural tests |
| R-02 | Strict configs | load_config validation |
| R-03 | Explicit seeds | Pipeline tests |
| R-04 | Structured errors | Named error stages |
| R-05 | Logging v2 | write_log_entry |

---

## 4. Façade — BinaryEmbeddingEngine

**Location**: `src/variants/binary_embedding_engine.py`

### 4.1 Purpose

Pure in-memory orchestrator for the numerical pipeline:
1. Normalize embeddings (L2)
2. Project via Gaussian random hyperplanes
3. Binarize to `{-1,+1}` and `{0,1}`
4. Pack into `uint64` words
5. Compute metrics
6. Record instrumentation

### 4.2 Constructor

```python
@dataclass
class BinaryEmbeddingEngine:
    encoder_spec: EncoderSpec
    dataset_spec: DatasetSpec
    code_bits: int            # 32, 64, 128, 256
    projection_type: str      # "gaussian" only
    seed: int
    normalize: bool = True
    projection_seed: int      # Defaults to seed
```

### 4.3 Run Method

```python
def run(
    embeddings: ArrayLike,
    *,
    metrics: Iterable[str] = ("similarity", "retrieval"),
    retrieval_k: int = 3,
    classification_labels: Optional[Sequence[int]] = None,
    log_hook: Optional[Callable] = None,
    return_full_code_bits: bool = False,
) -> Dict[str, object]:
```

### 4.4 Façade Invariants

| ID | Invariant |
|----|-----------|
| F-01 | Constructor validates all fields |
| F-02 | No I/O operations |
| F-03 | Dataset capability enforcement |
| F-04 | Deterministic metrics |
| F-05 | Full-width codes when requested |

---

## 5. Quantization

### 5.1 Binarization

**Location**: `src/quantization/binarization.py`

```python
def binarize_sign(x: np.ndarray) -> np.ndarray:
    """Return +1 for x >= 0, -1 otherwise."""
```

```python
@dataclass
class RandomProjection:
    input_dim: int
    output_bits: int
    seed: int
    
    def project(self, x: np.ndarray) -> np.ndarray:
        """Project inputs onto random hyperplanes."""
```

### 5.2 Packing

**Location**: `src/quantization/packing.py`

**Bit Layout**: LSB-first, row-major across `uint64` words.

```
codes_01 row: b0 b1 b2 ... b63 | b64 b65 ... b127 | ...
word 0   : bit0<-b0 ... bit63<-b63
word 1   : bit0<-b64 ...
```

```python
def pack_codes(codes_01: np.ndarray) -> np.ndarray:
    """Pack {0,1} codes into uint64 words."""

def unpack_codes(packed: np.ndarray, code_bits: int) -> np.ndarray:
    """Unpack uint64 words to logical view."""
```

### 5.3 Quantization Invariants

| ID | Invariant |
|----|-----------|
| Q-01 | Sign: x ≥ 0 → +1, x < 0 → -1 |
| Q-02 | Gaussian weights (seed-deterministic) |
| Q-03 | Finite inputs only |
| Q-04 | LSB-first packing (frozen for GPU kernels) |
| Q-05 | Exact round-trip: unpack(pack(codes)) == codes |

---

## 6. Evaluation

### 6.1 Similarity

**Location**: `src/eval/similarity.py`

- `cosine_similarity_matrix(embeddings, assume_normalized)`
- `hamming_distance_matrix(codes_01)`
- `spearman_correlation_from_matrices(cosine, hamming)`
- `neighbor_overlap_at_k(cosine_matrix, hamming_matrix, k)`

### 6.2 Retrieval

**Location**: `src/eval/retrieval.py`

- `topk_neighbor_indices_cosine(embeddings, k, seed)`
- `topk_neighbor_indices_hamming(codes, k, seed)`
- `neighbor_overlap_at_k(cosine_topk, hamming_topk)`
- `ndcg_at_k(predictions, relevance, k)`
- `recall_at_k(predictions, relevance, k)`

### 6.3 Classification

**Location**: `src/eval/classification.py`

Deterministic centroid classifier:
1. Normalize features
2. Compute class centroids
3. Predict via cosine similarity
4. Return accuracy, F1, and degradation delta

**Degradation Contract**: `accuracy_delta = binary_accuracy - float_accuracy < 0`

---

## 7. Configuration

**Location**: `src/utils/config.py`

### 7.1 Supported Values

- `SUPPORTED_PROJECTIONS = {"gaussian"}`
- `SUPPORTED_CODE_BITS = {32, 64, 128, 256}`

### 7.2 Required Fields

| Field | Type |
|-------|------|
| encoder_name | str |
| dataset_name | str |
| code_bits | int |
| projection_type | str |
| runner | str |
| tasks | list[str] |
| seed | int |

### 7.3 Optional Fields

| Field | Type |
|-------|------|
| embedding_files | list[str] |
| dataset_format | str |
| metrics | list[str] |
| hypotheses | list[str] |
| output_dir | str |
| classification_labels | str |

### 7.4 Example Config

```yaml
runner: phase1_binary_embeddings
encoder_name: my_encoder
dataset_name: my_dataset
code_bits: 64
projection_type: gaussian
seed: 42
tasks:
  - similarity
  - retrieval
  - classification
embedding_files:
  - path/to/embeddings.npy
classification_labels: path/to/labels.npy
output_dir: runs/
```

---

## 8. Logging

**Location**: `src/utils/logging.py`

### 8.1 Required Log Fields

- `version`, `status`
- `encoder_name`, `dataset_name`, `dataset_format`
- `code_bits`, `projection_type`, `projection_seed`
- `runner`, `seed`, `git_hash`
- `metrics`, `metrics_requested`
- `hypotheses`, `normalization`, `instrumentation`, `system`

### 8.2 System Metadata Block

| Key | Description |
|-----|-------------|
| hostname | Node identifier |
| platform | OS descriptor |
| python_version | Python version string |
| cpu_model | Processor string |
| gpu_model | CUDA device name or "none" |
| num_gpus | CUDA device count |

---

## 9. Result Schema

### 9.1 Schema v2 Structure

```
result
├── version (str: "phase1-v2")
├── status (str: "success" | "error")
├── runner (str)
├── encoder_name (str|null)
├── dataset_name (str|null)
├── dataset_format (str|null)
├── code_bits (int|null)
├── projection_type (str|null)
├── projection_seed (int|null)
├── seed (int|null)
├── hypotheses (list[str])
├── retrieval_k (int|null)
├── config_fingerprint (str|null)
├── git_hash (str)
├── output_dir (str|null)
├── normalization
│   └── l2 (bool|null)
├── instrumentation
│   ├── binarization_time_ms (float)
│   └── packing_time_ms (float)
├── system {...}
├── metrics_requested (tuple[str,...])
├── metrics
│   ├── similarity (dict|null)
│   ├── retrieval (dict|null)
│   └── classification (dict|null)
├── similarity_metrics (dict|null)
├── retrieval_metrics (dict|null)
├── classification_metrics (dict|null)
├── binary_codes (dict|null)
│   ├── pm1 (array)
│   ├── 01 (array)
│   └── packed (array)
└── error (dict|null)
    ├── stage (str)
    ├── message (str)
    ├── exception_type (str)
    └── variant (str)
```

---

## 10. Error Pipeline

### 10.1 Stage Definitions

| Stage | Description |
|-------|-------------|
| config_load | Config parsing and validation |
| seed_extraction | Seed conversion to int |
| catalog_lookup | Dataset/encoder metadata |
| load_embeddings | Loading embeddings/labels |
| engine_init | BinaryEmbeddingEngine construction |
| engine_run | Façade pipeline execution |
| logging | Writing JSON log entry |

### 10.2 Error Behavior

- Structured errors include: stage, message, exception_type, variant
- Unknown dataset/encoder errors propagate as raw exceptions
- Metrics and binary_codes set to null on error

---

## 11. Determinism

### 11.1 Seed Sources

- **Global seed**: Sets Python, NumPy, and optional Torch RNGs
- **Projection seed**: Controls Gaussian weight matrix
- **Retrieval seed**: Controls tie-breaking

### 11.2 Guarantees

Same config + same inputs + same seeds always yield:
- Identical `metrics` trees
- Identical `binary_codes` (pm1, 01, packed)
- Identical core metadata

---

## 12. Golden Regression

### 12.1 Golden Artifacts

Located in `tests/data/phase1_golden/`:

| File | Purpose |
|------|---------|
| config_phase1_synthetic_v1.yaml | Golden config |
| embeddings_phase1_synthetic.npy | Float embeddings |
| labels_phase1_synthetic.npy | Classification labels |
| golden_result_phase1_synthetic_v1.json | Canonical result |
| golden_log_phase1_synthetic_v1.jsonl | Canonical log |

### 12.2 Classification Degradation Lock

- `float_accuracy = 1.0`
- `binary_accuracy ≈ 0.9167`
- `accuracy_delta ≈ -0.0833`

---

## 13. Invariants

### 13.1 Consolidated Invariants Table

| ID | Invariant | Enforced By |
|----|-----------|-------------|
| INV-01 | projection_type ∈ {"gaussian"} | Config, façade, runner |
| INV-02 | Runner has no numerical ops | Structural tests |
| INV-03 | Façade performs no I/O | Manual audit |
| INV-04 | Binarization only in quantization | Module ownership |
| INV-05 | Same seed ⇒ same results | Golden tests |
| INV-06 | binary_codes shapes (N, code_bits) | Façade tests |
| INV-07 | Packed layout LSB-first | Packing tests |
| INV-08 | Result schema v2 fields present | Logging tests |
| INV-09 | System metadata 6 keys | Logging tests |
| INV-10 | Classification degradation negative | Golden tests |

---

## Appendix A: File Index

| Path | Purpose |
|------|---------|
| src/experiments/runners/phase1_binary_embeddings.py | Runner |
| src/variants/binary_embedding_engine.py | Façade |
| src/quantization/binarization.py | Projection & binarization |
| src/quantization/packing.py | Bit packing |
| src/eval/similarity.py | Similarity metrics |
| src/eval/retrieval.py | Retrieval metrics |
| src/eval/classification.py | Centroid classifier |
| src/core/dataset_catalog.py | Dataset registry |
| src/utils/config.py | Config loader |
| src/utils/logging.py | Logging v2 |

---

*End of Technical Reference*


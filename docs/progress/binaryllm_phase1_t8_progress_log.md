# BinaryLLM – Phase 1 Progress Log  
## T8 – Config, Seed, IO, Logging Infrastructure

**Date:** 2025-11-23  
**Phase:** 1 – Binary Embedding & Similarity Engine  
**Task ID:** T8 – Config / Seed / IO / Logging

---

## 1. Scope of T8

T8 introduces the **infrastructure layer** that everything else in Phase 1 depends on:

- **Config loading & validation**
- **Global seed setting**
- **Embedding/metadata IO**
- **Structured experiment logging**

Goal:  
Provide a **deterministic, spec-aligned infra backbone** so that T9 (runner) and all later experiments are reproducible and auditable.

---

## 2. Modules & Responsibilities

### 2.1 Config Utilities

**Location:** `src/utils/config.py`

**Responsibilities:**

- Load experiment configs (YAML / JSON) for Phase 1.
- Validate:
  - `encoder_name`, `dataset_name` against the **dataset/encoder catalog**.
  - `code_bits` against supported bit-widths.
  - `projection_type`, `metrics`, `k`, `seed` and other Phase 1 knobs.
- Enforce:
  - **No unknown keys** (no silent ignoring).
  - Required fields must be present.
- Compute a **config fingerprint**:
  - `json.dumps(config, sort_keys=True, separators=(',', ':'))`
  - `SHA256` hash.
- Expose fingerprint + normalized config to runners/logging.

**Invariants:**

- Any invalid field → `ValueError` with:
  - offending key name,
  - clear message (no vague “invalid config”).
- Configs are **pure data**: no side effects, no lazy IO.

---

### 2.2 Seed Utilities

**Location:** `src/utils/seed.py`

**Responsibilities:**

- Single entrypoint `set_global_seed(seed: int)` that **must** be used by all runners.
- Seed:
  - `random`
  - `numpy.random`
  - `torch` (if installed; otherwise noop with documented behavior).

**Invariants:**

- Same seed → same random sequences for:
  - Python RNG,
  - NumPy,
  - Torch (if present).
- Tests confirm:
  - re-seeding with same integer reproduces sequences,
  - different seeds produce different sequences.

---

### 2.3 IO Utilities

**Location:** `src/utils/io.py`

**Responsibilities:**

- Load and save **precomputed embeddings** and **metadata** using the catalog contract.
- Supported Phase 1 formats:
  - `.npy` (dense arrays),
  - `.parquet` (tabular embeddings + fields),
  - JSONL metadata.
- Keep IO **pure & deterministic**:
  - no shuffling,
  - no dropping rows silently,
  - no type guessing beyond defined schema.

**Invariants:**

- Round-trip:
  - `save → load` for supported formats yields identical arrays / tables.
- Metadata:
  - order-preserving,
  - fields match `DatasetSpec.required_fields`.
- Errors:
  - Malformed files → explicit `ValueError` / IO errors with:
    - dataset name,
    - stage (`"io"` / `"load_embeddings"`).

---

### 2.4 Logging Utilities

**Location:** `src/utils/logging.py`

**Responsibilities:**

- Structured logging for **Phase 1 experiments**:
  - metrics,
  - config snapshot or fingerprint,
  - seed,
  - `encoder_name`, `dataset_name`,
  - hypotheses (`["H1"]` for Phase 1),
  - Git commit hash if available.
- Write JSON logs with:
  - `sort_keys=True`,
  - stable key ordering,
  - ISO8601 timestamps.

**Invariants:**

- Every experiment run produces:
  - at least one **metrics JSON** entry with:
    - `encoder_name`
    - `dataset_name`
    - `code_bits`
    - `projection_type`
    - `metrics` block (similarity / retrieval / classification if used)
    - `seed`
    - `hypotheses`
    - `config_fingerprint`
    - `git_hash` (`null` if not available).
- No ad-hoc prints or unstructured logs.
- No randomness in log formatting (sorting enforced).

---

## 3. Test Suite – T8

**Location:** `tests/` (exact file names may vary, but cover these areas)

### 3.1 Config Tests

- Load valid YAML/JSON configs and:
  - confirm normalized data structures,
  - confirm identical fingerprints across formats.
- Invalid configs:
  - unknown keys → `ValueError` containing `"unknown"` + field name.
  - missing required fields → `ValueError` with clear message.
  - unsupported `code_bits` / `projection_type` → explicit failures.

### 3.2 Seed Tests

- `set_global_seed(seed)`:
  - two separate runs with same seed → same NumPy/random sequences.
  - different seeds → different sequences.
- If `torch` is installed:
  - deterministic behavior confirmed for torch RNG.

### 3.3 IO Tests

- `.npy` and `.parquet` round-trips:
  - saved → loaded arrays equal elementwise.
- JSONL metadata:
  - order preserved,
  - fields match catalog spec.
- Error paths:
  - corrupted/malformed files → explicit failures.

### 3.4 Logging Tests

- Logging a dummy experiment:
  - JSON output contains all mandatory fields.
  - `config_fingerprint` length and format validated.
  - `seed`, `encoder_name`, `dataset_name`, `hypotheses` present.
  - Key ordering stable across runs.

---

## 4. Hostile Review – Summary

**Hostile verdict for T8:**

- `verdict: "PASS"`
- **No critical issues**
- **No spec violations**
- **Coverage OK**
- Determinism validated:
  - config fingerprints stable across formats,
  - RNG seeding deterministic,
  - IO round-trips deterministic,
  - logging stable and non-random.

**Conclusion:**  
T8 infra is **frozen** for Phase 1.  
No further iterations required unless the architecture spec is extended.

---

## 5. Status & Dependencies

- **Status:** ✅ T8 COMPLETE & FROZEN
- **Depends on:**  
  - T1 (skeleton)  
  - T2 (embeddings/datasets)  
  - T3 (binarization)  
  - T4 (bit packing)  
  - T5 (similarity)  
  - T6 (retrieval)  
  - Architecture v1 + Research Report v2

- **Provides infra for:**
  - T9 – Phase 1 Experiment Runner
  - Any future Phase 2+ modules needing:
    - config parsing,
    - seeding,
    - IO,
    - logging.

---

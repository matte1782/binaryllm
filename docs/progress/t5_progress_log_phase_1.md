# BinaryLLM — Phase 1 Progress Log  
## T5 – Similarity Module (Cosine, Hamming, Neighbor Overlap@k)

**Status:** ✅ Completed & Frozen  
**Scope:** `src/eval/similarity.py`, `tests/test_eval_similarity.py`

---

### 1. Goal of T5

Design and implement the **Similarity Module** for Phase 1:

- Compute **cosine similarity** on float embeddings (with/without normalization).
- Compute **Hamming distance** on binary `{0,1}` codes.
- Implement **neighbor-overlap@k** between float and binary similarity rankings.
- Ensure **determinism, clear invariants, and spec alignment** with:
  - `binaryllm_report_v2.md` (H1: similarity and neighborhood preservation),
  - `binaryllm_phase1_architecture_v1.md` (T5 spec).

The module must be **pure evaluation**:
- No training, no heuristics, no packing logic.
- Only consumes:
  - `FloatEmbeddingBatch` (Phase 1 core),
  - `BinaryCodeBatch` / `codes_01` (from T3/T4).

---

### 2. Implemented Interfaces

**File:** `src/eval/similarity.py`  

Key functions (conceptual):

- `cosine_similarity_matrix(embeddings, assume_normalized: bool = False) -> np.ndarray`
  - Input: 2D float array `(N, d)` (or `FloatEmbeddingBatch` backing data).
  - Behavior:
    - Validates finite values (no NaN/Inf).
    - If `assume_normalized=False` → **row-wise L2 normalization**, then `X @ X.T`.
    - If `assume_normalized=True` → assumes vectors already L2-normalized, performs `X @ X.T` with no additional normalization.
  - Output:
    - `(N, N)` float32 matrix.
    - Symmetric, ones on diagonal (up to numerical tolerance for normalized inputs).

- `hamming_distance_matrix(codes_01) -> np.ndarray`
  - Input: 2D `{0,1}` integer/bool array `(N, m)` from logical binary codes.
  - Behavior:
    - Validates:
      - Non-empty,
      - 2D,
      - Values ∈ `{0,1}`.
    - Computes pairwise Hamming distance via XOR + sum over bits.
  - Output:
    - `(N, N)` float32 distance matrix.
    - Symmetric, zeros on diagonal.

- `topk_neighbor_indices(similarity_matrix, k: int) -> np.ndarray`
  - Deterministic **top-k retrieval** for each query (row):
    - Handles ties deterministically (e.g., stable sort + fixed order).
  - Used as a building block for neighbor overlap.

- `neighbor_overlap_at_k(cosine_sim, hamming_sim, k: int) -> float`
  - Defines pure **overlap metric**:
    - For each query `i`:
      - `N_cos(i)` = top-k neighbors under cosine.
      - `N_ham(i)` = top-k neighbors under Hamming (by converting distances to similarity ordering).
      - Overlap@k for `i` = `|N_cos(i) ∩ N_ham(i)| / k`.
    - Global metric = mean over all queries.
  - No bonuses, no heuristic terms, no fudge factors.

All implementations are **deterministic**, pure CPU/Numpy logic, no side effects.

---

### 3. Invariants & Numerical Conventions

- **Cosine:**
  - When `assume_normalized=True`, caller is responsible for L2-normalization.
  - When `assume_normalized=False`, module normalizes rows internally.
  - Outputs are float32; diagonal ≈ 1 for normalized embeddings.

- **Hamming:**
  - Input: {0,1} codes (logical).
  - Output: float32 distances, but conceptually integer-valued.
  - Strict `{0,1}` validation: any other value → `ValueError`.

- **Neighbor Overlap@k:**
  - Pure set overlap: no positional weighting.
  - Deterministic tie-handling via fully controlled sorting.

---

### 4. Test Coverage (T5)

**File:** `tests/test_eval_similarity.py`  

Main categories:

1. **Cosine tests**
   - Checks:
     - Correct shape `(N, N)`.
     - Symmetry & diagonal ≈ 1 for normalized inputs.
     - `assume_normalized=False` reproduces cosine of a manually normalized baseline.
     - Optional (future): `assume_normalized=True` consistency test.

2. **Hamming tests**
   - Valid Hamming distance on small synthetic codes (identity, inverses).
   - Zero on diagonal, max at full mismatch.
   - `{0,1}` validation (invalid values → `ValueError`).

3. **Neighbor Overlap tests**
   - Deterministic top-k retrieval.
   - Overlap@k defined as Intersection/k.
   - Monotonicity scenarios (e.g., increasing `code_bits` doesn’t break behavior), but **without forcing metric redefinition**.

4. **Error-path tests**
   - Invalid shapes (1D, empty arrays).
   - NaNs/Infs in cosine inputs.
   - Non-binary codes for Hamming.

5. **Determinism tests**
   - Re-running similarity and overlap computations with the same input yields identical results.

---

### 5. Hostile Review – Final Verdict (T5)

Hostile reviewer output (essenza):

- ✅ **Verdict:** PASS  
- ✅ Cosine = standard cosine (no hidden scaling logic).  
- ✅ Hamming = correct XOR+sum over `{0,1}` codes.  
- ✅ Neighbor overlap@k = pure overlap metric, no heuristics.  
- ✅ Determinism, coverage, and spec alignment all confirmed.  
- Remaining suggestions are **optional micro-improvements**, not blocking.

**Decision:**  
→ T5 is **frozen** as a solid foundation for Phase 1.

---

# Plan for T6 — Retrieval Metrics (Design Before Prompts)

We now design **T6 – Retrieval Metrics**, so that next step we can write the 3 enterprise prompts:

- `/tester_binaryllm` (T6)  
- `/engineer_binaryllm` (T6)  
- `/hostile_binaryllm` (T6)

## 1. T6 Scope & Goals

**Goal:** Build a clean, deterministic **retrieval evaluation layer** on top of cosine/Hamming, using T5 as a foundation.

T6 responsibilities (Phase 1):

1. **Top-k retrieval APIs** (per-query):
   - `topk_cosine(...)` using cosine similarity matrices or on-the-fly similarity.
   - `topk_hamming(...)` using Hamming distances or precomputed distance matrix.

2. **Ranking-based metrics:**
   - **Recall@k** (per query and averaged).
   - **nDCG@k** (per query and averaged).
   - Optionally: MRR@k (if needed, but can be postponed).

3. **Ground truth integration:**
   - Work with `EmbeddingDataset`, `QueryDataset`, and labels / relevance judgments.
   - Use simple, explicit representations for:
     - `relevant_docs_per_query` (sets or lists of IDs),
     - Or numeric relevance levels for nDCG (0/1 or multi-level).

4. **Deterministic behavior & invariants:**
   - Top-k selection must be deterministic under ties.
   - Metrics must be fully deterministic given:
     - dataset,
     - similarity/distance matrix,
     - k,
     - relevance labels.

5. **Separation of concerns:**
   - T6 does *not*:
     - Compute embeddings (T2).
     - Binarize (T3).
     - Pack bits (T4).
     - Compute similarity matrices (T5).
   - It **only** consumes:
     - similarity/distances from T5,  
     - dataset/label structures from T2.

---

## 2. Expected Module & File Layout for T6

**Implementation target file:**

- `src/eval/retrieval.py`

**Key conceptual functions/classes:**

- `compute_topk_indices(sim_or_dist, k: int, largest: bool = True) -> np.ndarray`
  - Generic top-k indexing utility; deterministic tie-breaking.

- `recall_at_k(retrieved_indices, relevant_sets, k: int) -> float`
  - Inputs:
    - `retrieved_indices`: `(n_queries, k)` array of doc indices.
    - `relevant_sets`: list/array of sets or 1D arrays of relevant doc IDs per query.
  - Output:
    - Mean Recall@k over queries.

- `ndcg_at_k(retrieved_indices, relevance_scores, k: int) -> float`
  - Inputs:
    - `retrieved_indices` as above.
    - `relevance_scores`: e.g., 2D array `(n_queries, n_docs)` or sparse structure mapping (query, doc) → gain.
  - Output:
    - Mean nDCG@k over queries.

- Optional helper:
  - `build_relevance_from_dataset(query_dataset, corpus_dataset) -> (relevant_sets or relevance_scores)`
    - Uses metadata from T2 dataset abstractions.

**Test file:**

- `tests/test_eval_retrieval.py`

---

## 3. Invariants & Contracts for T6

- **Input validation:**
  - Similarity/distance matrices:
    - 2D, finite.
    - Sizes consistent with dataset cardinality.
  - `k`:
    - Integer, `1 ≤ k ≤ n_docs`.
  - Relevance structures:
    - Every query has a defined relevance set or score vector.
    - No inconsistent IDs (e.g., relevance referencing out-of-range doc indices).

- **Determinism:**
  - Same inputs → same top-k, same metrics.
  - Tie-handling:
    - Define tie-breaking rule (e.g., stable ordering by doc index or by existing sort).
    - Tests must explicitly check determinism under ties.

- **Metric semantics:**
  - **Recall@k**:
    - For each query: `|retrieved ∩ relevant| / |relevant|`.
    - Global = mean over queries (or support sum+count; spec must fix one).
  - **nDCG@k**:
    - Uses standard DCG@k with gains from relevance labels.
    - IDCG@k computed per query from ideal ranking.
    - nDCG@k = DCG@k / IDCG@k (with convention `0/0 → 0`).
  - No heuristic fudge factors; pure textbook definitions.

---

## 4. T6 – Prompt Architecture Plan

We won’t write the full prompts here, but this is the **skeleton** they must follow.

### 4.1 `/tester_binaryllm` – T6 Test Agent

Responsibilities:

1. **Design tests first**:
   - Small synthetic datasets with known ground truth:
     - One where each query has a single relevant item (easy checks).
     - One with multiple relevance levels (for nDCG).
   - Test categories:
     - Correctness of Recall@k and nDCG@k.
     - Deterministic top-k results with ties.
     - Edge cases:
       - No relevant docs,
       - All docs relevant,
       - k > number of relevant docs,
       - k = 1 and k = n_docs.
     - Error-path tests:
       - Invalid k,
       - Mismatched shapes,
       - Out-of-range doc IDs in relevance sets.

2. **File boundaries:**
   - Only `tests/test_eval_retrieval.py` (plus possible tiny fixtures under `tests/data/phase1_synthetic/` if spec already allows it).
   - No changes to `src/` in tester role.

3. **Expected output:**
   - A clear patch plan:
     - Exact test functions to add.
     - What each test asserts.
     - How they map to H1 / Phase 1 goals.

### 4.2 `/engineer_binaryllm` – T6 Implementation Agent

Responsibilities:

1. **Implement only what tests require**:
   - In `src/eval/retrieval.py`:
     - Top-k index function.
     - Recall@k.
     - nDCG@k.
   - No extra features (e.g., MRR) unless explicitly requested by tests/arch.

2. **Respect invariants:**
   - Strict input validation (fail fast).
   - Deterministic tie-handling.
   - Verified shapes and index ranges.

3. **No scope creep:**
   - No GPU-specific shortcuts.
   - No caching or state.
   - No assumptions about future architecture phases.

4. **Document assumptions:**
   - e.g., how relevance is represented:
     - A list of sets per query,
     - Or dense 2D array.

### 4.3 `/hostile_binaryllm` – T6 Review Agent

Responsibilities:

1. **Metric semantics verification:**
   - Check Recall@k and nDCG@k definitions against:
     - Information retrieval literature,
     - Architecture spec for Phase 1.

2. **Test/implementation alignment:**
   - Ensure tests and code agree on:
     - tie-breaking policy,
     - 0/0 nDCG handling,
     - representation of “no relevant docs” cases.

3. **Determinism & regression:**
   - Confirm:
     - No randomness without seeds.
     - Simpler top-k logic is sufficient.
   - Inspect if tests cover:
     - ties,
     - edge queries,
     - error paths.

4. **Scope & architecture compliance:**
   - Reject:
     - any hidden heuristics,
     - any attempt to re-define metrics,
     - any logic that belongs to similarity (T5) or packing (T4).

---

## 5. Next Concrete Step

You are now ready to:

1. **Add T5 section** to `binary_llm_phase_1_progress_log.md` using the log above.
2. Then say:
   - **“Create the full T6 prompts now.”**  
   and we’ll turn this plan into the actual three NVIDIA-grade prompts for:
   - `/tester_binaryllm` (T6),
   - `/engineer_binaryllm` (T6),
   - `/hostile_binaryllm` (T6).

This keeps the discipline:  
**spec → tests → code → hostile review**, step by step.

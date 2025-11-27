# BinaryLLM Phase‑2 — Scientific Experiment Plan (Iteration 3)

**Author:** Senior Scientist Agent  
**Date:** 2025-11-26  
**Status:** Iteration 3 — Post-Hostile, Post-Strategist Convergence

---

## Executive Scientific Summary

The Strategist (Iteration 3) has converged on a single Phase‑2 direction:

> **Binary Retrieval with LLM Embeddings: An Empirical Study of Quality–Memory Trade‑offs**

This document translates that strategic choice into a rigorous, executable scientific experiment plan. The goal is **not** to claim that binary retrieval beats state‑of‑the‑art ANN systems (Faiss/ScaNN/PQ), but to **characterize precisely** under what conditions binary codes from Phase‑1 are competitive, acceptable, or unacceptable—producing publishable‑quality trade‑off curves and actionable guidance for practitioners.

**Key scientific contributions targeted:**

1. **First modern, LLM‑centric empirical map** of binary retrieval quality across multiple embedding models, datasets, and bit‑lengths.
2. **Controlled comparison** of Phase‑1 binary codes against PQ and INT8 baselines under matched memory budgets.
3. **Diagnostic analysis** linking embedding geometry (anisotropy, outliers, intrinsic dimensionality) to binarization performance.
4. **Explicit failure‑mode documentation**: where binary retrieval collapses and why.

The study is scoped to fit within **≤100 A100‑hours** (well under the 150–200 hour cap) and a **10–14 week** part‑time schedule for a single student.

---

## 1. Restated Phase‑2 Direction

### 1.1 Scientific Framing

Phase‑2 is an **empirical study** that measures how retrieval quality degrades when LLM embeddings are compressed to 1‑bit binary codes (via Phase‑1's deterministic random‑projection + sign binarization pipeline), compared against:

- **Float32 brute‑force** (oracle baseline).
- **INT8 quantization** (8× memory reduction, minimal quality loss).
- **Product Quantization (PQ)** at various sub‑vector configurations (8–64 bytes/vector).

The study spans multiple:

- **Bit‑lengths** for binary codes: 64, 128, 256, 512 bits.
- **Embedding models**: 2–3 open‑source LLM text encoders of different sizes and training regimes.
- **Datasets**: 3–4 retrieval / similarity benchmarks covering different domains and difficulty levels.

### 1.2 What Is Being Measured

| Metric | Definition | Purpose |
|--------|------------|---------|
| **Recall@k** | Fraction of true top‑k neighbors (under float cosine) recovered by the compressed index. | Primary quality measure. |
| **nDCG@k** | Normalized discounted cumulative gain using float‑cosine relevance as ground truth. | Captures ranking quality beyond set overlap. |
| **Overlap@k** | Set intersection size between float‑cosine top‑k and compressed top‑k, normalized. | Direct Phase‑1 metric; H1 alignment. |
| **Memory (bytes/vector)** | Storage cost per embedding under each scheme. | Enables fair iso‑memory comparisons. |
| **Latency / QPS** | Query latency and throughput (CPU baseline; optional GPU prototype). | Secondary; no unrealistic speedup claims. |

### 1.3 Hypotheses Under Test

We frame the study around **three falsifiable hypotheses**:

- **H2.1 (Quality–Bit‑Length Scaling):**  
  For a fixed embedding model and dataset, Recall@10 under binary Hamming search increases monotonically with bit‑length (64 → 128 → 256 → 512), with diminishing returns beyond a model‑ and dataset‑dependent saturation point.

- **H2.2 (Iso‑Memory Competitiveness):**  
  Under matched memory budgets (e.g., 64 bits/vector), binary codes achieve Recall@10 within **0.85×** of PQ‑compressed embeddings on at least one mainstream retrieval benchmark.

- **H2.3 (Geometry–Quality Correlation):**  
  Embedding distributions with higher effective dimensionality (lower anisotropy, fewer outliers) exhibit better binarization quality (higher Recall@k at fixed bit‑length).

### 1.4 What Is NOT Being Claimed

- **No claim** that binary retrieval will **dominate** PQ/ScaNN/HNSW in general.
- **No claim** of "48×" or even "10×" speedups; any throughput gains must be measured, not assumed.
- **No product promises**: Phase‑2 delivers research artifacts, not a production vector DB.
- **No training of LLM encoders**: all embeddings come from frozen, pre‑trained models.

---

## 2. Core Scientific Questions

The following **primary research questions (RQs)** structure the experiment design:

### RQ1: How does Recall@k degrade as a function of bit‑length for LLM embeddings?

- **Metrics:** Recall@{1, 5, 10, 50, 100}, nDCG@10, Overlap@10.
- **Datasets:** All selected benchmarks.
- **Variables:** Bit‑length ∈ {64, 128, 256, 512}, embedding model, dataset.
- **Expected output:** Family of curves: Recall@k vs bit‑length, one per (model, dataset) pair.

### RQ2: How does binary retrieval compare to PQ under matched memory?

- **Metrics:** Recall@10, nDCG@10.
- **Baselines:** PQ with 8, 16, 32, 64 bytes/vector (sub‑vector configurations tuned per dataset).
- **Comparison:** Binary codes at equivalent memory (e.g., 64 bits = 8 bytes, 128 bits = 16 bytes).
- **Expected output:** Scatter plots and tables: Recall@10 vs memory (bytes/vector), binary vs PQ.

### RQ3: How do embedding properties correlate with binarization performance?

- **Geometric diagnostics:**
  - **Anisotropy:** Ratio of top singular value to median singular value of embedding matrix.
  - **Outlier prevalence:** Fraction of embeddings with L2 norm > 2× median norm (after centering).
  - **Intrinsic dimensionality:** Estimated via PCA variance explained at 90% threshold.
  - **Angular spread:** Mean pairwise cosine similarity (lower = more spread).
- **Correlation:** Spearman correlation between each geometric measure and Recall@10 at 128 bits across (model, dataset) pairs.
- **Expected output:** Correlation table + scatter plots.

### RQ4: On which tasks does binary retrieval fail catastrophically?

- **Definition of failure:** Recall@10 < 0.50 at 256 bits.
- **Analysis:** Identify datasets / models where failure occurs; characterize their geometry.
- **Expected output:** Failure case studies with qualitative explanations.

### RQ5: What is the realistic throughput of binary Hamming search vs float cosine and PQ?

- **Measurement:** Queries per second (QPS) on CPU (and optionally GPU prototype) for:
  - Brute‑force float cosine.
  - Brute‑force binary Hamming (using Phase‑1 packing + popcount).
  - Faiss PQ index.
- **Caveats:** No claims of production‑level speedups; measurements are indicative only.
- **Expected output:** QPS table with hardware specs and caveats.

### RQ6: What is the minimum bit‑length required to achieve "acceptable" recall on each benchmark?

- **Definition of acceptable:** Recall@10 ≥ 0.70.
- **Sweep:** For each (model, dataset), find smallest bit‑length in {64, 128, 256, 512} achieving threshold (or mark as "not achieved").
- **Expected output:** Table of minimum acceptable bit‑lengths per benchmark.

### RQ7 (Optional): Does re‑ranking with float embeddings recover lost recall?

- **Protocol:** Retrieve top‑100 via binary Hamming, re‑rank with float cosine, report Recall@10 of re‑ranked list.
- **Purpose:** Quantify practical two‑stage retrieval benefit.
- **Expected output:** Recall@10 improvement from re‑ranking, per benchmark.

---

## 3. Datasets & Benchmark Plan

### 3.1 Selection Criteria

- **Diversity:** Cover different domains (general web, QA, scientific, short/long text).
- **Size:** Small enough to fit in memory and run full brute‑force evaluation, but large enough to be meaningful (10K–1M documents).
- **Public availability:** All datasets must be freely downloadable.
- **Embedding availability:** Either pre‑computed embeddings are available, or embeddings can be generated with reasonable GPU time.

### 3.2 Selected Benchmarks

| Dataset | Domain | Corpus Size (N) | Query Count | Justification |
|---------|--------|-----------------|-------------|---------------|
| **MS MARCO (dev small)** | Web passage retrieval | ~8.8M (subsample to 500K–1M) | 6,980 | Industry‑standard retrieval benchmark; well‑studied. |
| **Natural Questions (NQ)** | QA retrieval | ~21M (subsample to 500K) | 3,610 | Wikipedia‑based QA; different query distribution. |
| **BEIR/SciFact** | Scientific claim verification | 5,183 | 300 | Small, specialized; tests domain shift. |
| **STS Benchmark** | Semantic similarity | ~8,000 pairs | N/A (pairwise) | Controlled similarity task; Phase‑1 style evaluation. |

**Subsampling strategy:** For large corpora (MS MARCO, NQ), we uniformly subsample to 500K–1M documents to keep brute‑force evaluation tractable. Subsampling seed is fixed and documented.

### 3.3 Embedding Models

| Model | Dim | Size | Training | Justification |
|-------|-----|------|----------|---------------|
| **all-MiniLM-L6-v2** | 384 | 22M params | Contrastive (sentence-transformers) | Small, fast, widely used baseline. |
| **e5-base-v2** | 768 | 110M params | Contrastive + weak supervision | Strong open‑source retriever; larger dim. |
| **bge-large-en-v1.5** | 1024 | 335M params | Contrastive | High‑quality, large embedding model. |

**Embedding generation:**

- Use Hugging Face `sentence-transformers` or `transformers` library.
- Batch inference on A100; cache embeddings to disk (NPY format).
- Estimated GPU time: ~10–20 A100‑hours for all (model, dataset) combinations.

### 3.4 Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│  1. Download corpus + queries (public sources)                  │
│  2. Subsample corpus (fixed seed)                               │
│  3. Generate float embeddings (GPU, cache to disk)              │
│  4. L2‑normalize embeddings (Phase‑1 requirement for H1 theory) │
│  5. Binarize via Phase‑1 pipeline (CPU, deterministic)          │
│  6. Build PQ/INT8 indices via Faiss (CPU/GPU)                   │
│  7. Run retrieval evaluation (CPU)                              │
│  8. Log results (JSON, Phase‑2 schema)                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Metrics & Analysis Plan

### 4.1 Primary Metrics

| Metric | Formula / Definition | Implementation |
|--------|----------------------|----------------|
| **Recall@k** | \(\frac{|\text{retrieved}_k \cap \text{relevant}_k|}{k}\) where \(\text{relevant}_k\) = true top‑k under float cosine. | `src/eval/retrieval.py::recall_at_k` (Phase‑1). |
| **nDCG@k** | Standard nDCG using float‑cosine similarity as relevance score. | `src/eval/retrieval.py::ndcg_at_k` (Phase‑1). |
| **Overlap@k** | Set intersection normalized by k. | `src/eval/retrieval.py::neighbor_overlap_at_k` (Phase‑1). |
| **Memory (bytes/vector)** | \(\lceil \text{code\_bits} / 8 \rceil\) for binary; varies for PQ. | Computed analytically. |

### 4.2 Secondary Metrics

| Metric | Definition | Purpose |
|--------|------------|---------|
| **MRR** | Mean reciprocal rank of first relevant document. | Alternative ranking metric. |
| **QPS** | Queries per second (brute‑force, single‑threaded CPU baseline). | Throughput indication. |
| **Latency p50/p99** | Median and 99th percentile query latency. | Tail latency characterization. |

### 4.3 Geometric Diagnostics

| Diagnostic | Computation | Interpretation |
|------------|-------------|----------------|
| **Anisotropy** | \(\sigma_1 / \sigma_{\text{median}}\) from SVD of centered embedding matrix. | High anisotropy → embeddings cluster along few directions → binary hashing may fail. |
| **Outlier fraction** | Fraction of embeddings with \(\|x\|_2 > 2 \cdot \text{median}(\|x\|_2)\) after centering. | High outlier fraction → sign binarization unstable. |
| **Intrinsic dim** | Number of PCA components for 90% variance. | Low intrinsic dim → binary codes may suffice with fewer bits. |
| **Angular spread** | Mean pairwise cosine similarity (sampled). | Low spread (high mean cosine) → neighbors are hard to separate. |

### 4.4 Baseline Index Configurations

| Method | Config | Memory (bytes/vector) | Notes |
|--------|--------|----------------------|-------|
| **Float32 brute‑force** | N/A | 4 × dim | Oracle baseline. |
| **INT8** | Per‑dimension quantization | dim | 4× compression vs float32. |
| **PQ‑8** | 8 sub‑vectors, 8 bits/sub‑vector | 8 | Faiss `IndexPQ`. |
| **PQ‑16** | 16 sub‑vectors | 16 | |
| **PQ‑32** | 32 sub‑vectors | 32 | |
| **PQ‑64** | 64 sub‑vectors | 64 | |
| **Binary‑64** | Phase‑1, 64 bits | 8 | |
| **Binary‑128** | Phase‑1, 128 bits | 16 | |
| **Binary‑256** | Phase‑1, 256 bits | 32 | |
| **Binary‑512** | Phase‑1, 512 bits | 64 | |

### 4.5 Analysis Outputs

1. **Trade‑off curves:** Recall@10 vs memory (bytes/vector) for each (model, dataset), overlaying binary and PQ.
2. **Bit‑length scaling curves:** Recall@{1,5,10,50,100} vs bit‑length for each (model, dataset).
3. **Geometry correlation table:** Spearman ρ between each geometric diagnostic and Recall@10 at 128 bits.
4. **Failure case studies:** Qualitative analysis of (model, dataset) pairs where binary fails.
5. **Throughput table:** QPS for each method on a reference hardware configuration.
6. **Minimum acceptable bit‑length table:** Smallest bit‑length achieving Recall@10 ≥ 0.70 per benchmark.

---

## 5. Theoretical Notes

### 5.1 Classical LSH / Binary Hashing Theory

Phase‑1 uses **random hyperplane hashing** (SimHash / LSH for cosine similarity):

\[
h_r(x) = \text{sign}(r^\top x), \quad r \sim \mathcal{N}(0, I)
\]

For unit vectors \(x, y\), the probability of hash collision is:

\[
\Pr[h_r(x) = h_r(y)] = 1 - \frac{\theta(x, y)}{\pi}
\]

where \(\theta(x, y) = \arccos(\langle x, y \rangle)\).

**Implications:**

- Hamming distance \(d_H\) between \(b\)‑bit codes approximates angular distance:  
  \(\mathbb{E}[d_H] = b \cdot \frac{\theta}{\pi}\).
- For small angles (high similarity), the approximation is accurate; for large angles, variance increases.
- **Limitation:** Theory assumes **uniform angular distribution**; real LLM embeddings are **not uniform**.

### 5.2 LLM Embedding Pathologies

Prior work (e.g., Ethayarajh 2019, Su et al. 2021) documents:

- **Anisotropy:** LLM embeddings cluster in a narrow cone; most pairwise cosine similarities are high.
- **Outlier dimensions:** A few dimensions carry disproportionate variance (related to frequency effects).
- **Hubness:** Some embeddings are nearest neighbors to many others, distorting retrieval.

**Expected impact on binary hashing:**

- High anisotropy → small angular gaps → binary codes struggle to separate neighbors from non‑neighbors.
- Outlier dimensions → sign binarization is dominated by outlier signs, losing information from other dimensions.
- Hubness → binary hashing may amplify hubness (hubs have generic binary codes).

### 5.3 Theoretical Expectations

Given the above, we **expect**:

- **Lower recall for binary vs PQ at matched memory**, especially on high‑anisotropy embeddings.
- **Diminishing returns beyond 256–512 bits**, as the information bottleneck shifts from bit‑length to embedding geometry.
- **Failure cases** on datasets with very high mean pairwise similarity (e.g., specialized domains with homogeneous text).

We do **not** claim provable guarantees; the study is empirical.

---

## 6. Minimum Experiment Set

### 6.1 Core Experiments (Mandatory)

| ID | Experiment | Inputs | Outputs | Est. GPU‑h | Est. CPU‑h |
|----|------------|--------|---------|-----------|-----------|
| **E1** | Embedding generation | 3 models × 4 datasets | Cached float embeddings | 15–25 | 0 |
| **E2** | Binarization sweep | Embeddings × {64,128,256,512} bits | Binary codes | 0 | 5–10 |
| **E3** | PQ index building | Embeddings × {8,16,32,64} bytes | Faiss PQ indices | 5–10 | 5 |
| **E4** | Brute‑force retrieval (float) | Embeddings | Ground‑truth top‑k | 0 | 20–40 |
| **E5** | Binary retrieval evaluation | Binary codes | Recall/nDCG/Overlap | 0 | 10–20 |
| **E6** | PQ retrieval evaluation | PQ indices | Recall/nDCG | 0 | 10–20 |
| **E7** | Geometric diagnostics | Embeddings | Anisotropy, outliers, etc. | 0 | 5 |
| **E8** | Correlation analysis | E5 + E7 results | Correlation table | 0 | 1 |
| **E9** | Throughput benchmarking | Indices | QPS table | 0 | 5 |

**Total estimated GPU time:** 20–35 A100‑hours.  
**Total estimated CPU time:** 60–100 hours (parallelizable).

### 6.2 Optional Experiments (Nice‑to‑Have)

| ID | Experiment | Purpose | Est. GPU‑h |
|----|------------|---------|-----------|
| **E10** | GPU Hamming kernel prototype | Measure GPU speedup potential | 10–20 |
| **E11** | Re‑ranking evaluation | Quantify two‑stage retrieval benefit | 0 (CPU) |
| **E12** | Additional embedding models | Broader coverage | 10–20 |
| **E13** | HNSW + binary codes | Graph‑based ANN with binary | 5–10 |

### 6.3 Compute Budget Summary

| Category | Planned | Upper Bound |
|----------|---------|-------------|
| Embedding generation | 20 A100‑h | 30 A100‑h |
| Index building (PQ) | 8 A100‑h | 15 A100‑h |
| Optional GPU kernels | 0–15 A100‑h | 20 A100‑h |
| **Total** | **28–43 A100‑h** | **65 A100‑h** |

Well under the 150–200 hour cap, leaving headroom for iteration and unexpected needs.

### 6.4 Storage Estimate

| Artifact | Size (approx) |
|----------|---------------|
| Float embeddings (3 models × 4 datasets × 500K–1M vectors × 384–1024 dim × 4 bytes) | ~50–150 GB |
| Binary codes (same × 64–512 bits) | ~2–10 GB |
| PQ indices | ~5–20 GB |
| Results JSON/CSV | <1 GB |
| **Total** | ~60–180 GB |

Manageable on a standard workstation or cloud instance.

---

## 7. Kill‑Switch Criteria

To prevent sunk‑cost fallacy and ensure scientific honesty, we define explicit **stop conditions**:

### 7.1 Failure Criteria (Kill Phase‑2 Direction)

Phase‑2 is considered a **failure** if:

- **F1:** Recall@10 at 512 bits **never exceeds 0.55** on **any** of the 4 selected datasets, for **any** of the 3 embedding models.
- **F2:** Binary codes at matched memory (e.g., 64 bits vs PQ‑8) achieve **<0.5×** the Recall@10 of PQ on **all** benchmarks.
- **F3:** Geometric diagnostics show **no statistically significant correlation** (|ρ| < 0.3) with binarization quality across (model, dataset) pairs.

**If any of F1–F3 is met:**

- Document the negative result thoroughly.
- Publish as a "negative results" or "lessons learned" artifact.
- Do **not** proceed to Phase‑3 with binary retrieval as the core direction.

### 7.2 Success Criteria (Justify Phase‑3)

Phase‑2 is considered **successful** and justifies Phase‑3 exploration if:

- **S1:** On at least **2 of 4 datasets**, binary codes at 256–512 bits achieve **Recall@10 ≥ 0.70**.
- **S2:** On at least **1 dataset**, binary codes at matched memory achieve **≥0.80×** the Recall@10 of PQ.
- **S3:** Geometric diagnostics show **at least one significant predictor** (|ρ| ≥ 0.5) of binarization quality.
- **S4:** The study produces **≥3 publishable‑quality plots** (trade‑off curves, correlation scatter, failure case study).

**If S1–S4 are met:**

- Phase‑2 is a success as a research artifact.
- Phase‑3 options may include: learned projections, binary‑friendly encoder fine‑tuning, or hybrid retrieval systems.

### 7.3 Early Termination Checkpoints

| Checkpoint | Timing | Condition for Early Stop |
|------------|--------|--------------------------|
| **CP1** | After E1 + E2 + E4 + E5 on 1 model × 1 dataset | If Recall@10 at 512 bits < 0.40, investigate immediately; consider dataset/model swap. |
| **CP2** | After E5 on all 3 models × 2 datasets | If F1 is trending toward failure, halt expansion and focus on diagnostics. |
| **CP3** | After E7 + E8 | If F3 is met, document and consider pivoting to alternative analysis (e.g., per‑dimension binarization). |

---

## 8. Experiment Execution Plan

### 8.1 Phase‑2 Directory Structure

```
phase_2/
├── configs/                    # Experiment configs (YAML)
│   ├── embed_gen/              # Embedding generation configs
│   ├── binary_eval/            # Binary retrieval evaluation configs
│   └── pq_eval/                # PQ baseline evaluation configs
├── scripts/                    # Execution scripts
│   ├── generate_embeddings.py
│   ├── run_binarization.py
│   ├── run_pq_baseline.py
│   ├── run_retrieval_eval.py
│   ├── compute_geometry.py
│   └── analyze_results.py
├── data/                       # Cached embeddings and indices (gitignored)
├── results/                    # Experiment results (JSON/CSV)
├── notebooks/                  # Analysis and plotting notebooks
├── docs/                       # Phase‑2 documentation
│   ├── phase2_literature_review.md
│   ├── phase2_architecture_overview.md
│   └── phase2_final_report.md
└── iteration_*.md              # Research logs (this file series)
```

### 8.2 Logging Schema (Phase‑2 Extension)

Extend Phase‑1 result schema with Phase‑2 fields:

```json
{
  "version": "2.0",
  "phase": "phase2",
  "experiment_id": "E5_binary_eval_msmarco_e5base_256bits",
  "timestamp": "2025-...",
  "status": "success",
  
  "dataset": {
    "name": "msmarco_dev_small",
    "corpus_size": 500000,
    "query_count": 6980,
    "subsample_seed": 42
  },
  
  "embedding_model": {
    "name": "e5-base-v2",
    "dim": 768,
    "normalized": true
  },
  
  "method": {
    "type": "binary",
    "code_bits": 256,
    "projection_type": "gaussian",
    "projection_seed": 12345
  },
  
  "metrics": {
    "recall_at_1": 0.45,
    "recall_at_5": 0.62,
    "recall_at_10": 0.71,
    "recall_at_50": 0.85,
    "recall_at_100": 0.91,
    "ndcg_at_10": 0.68,
    "overlap_at_10": 0.71,
    "mrr": 0.52
  },
  
  "geometry": {
    "anisotropy": 12.3,
    "outlier_fraction": 0.02,
    "intrinsic_dim_90": 45,
    "mean_pairwise_cosine": 0.35
  },
  
  "throughput": {
    "qps_cpu": 1250,
    "latency_p50_ms": 0.8,
    "latency_p99_ms": 2.1
  },
  
  "system": {
    "cpu": "AMD EPYC 7742",
    "gpu": "NVIDIA A100 80GB",
    "ram_gb": 512
  }
}
```

### 8.3 Reproducibility Requirements

- **All random seeds** (projection, subsampling, train/test splits) are fixed and logged.
- **All embeddings** are cached to disk with checksums.
- **All configs** are version‑controlled in `phase_2/configs/`.
- **All results** are logged in machine‑readable JSON with full provenance.
- **Phase‑1 code is never modified**; Phase‑2 imports Phase‑1 as a library.

---

## 9. Timeline

| Week | Activities | Deliverables |
|------|------------|--------------|
| **1–2** | Dataset download, subsampling, embedding generation (E1) | Cached embeddings for 1 model × 2 datasets |
| **3–4** | Complete E1 for all models/datasets; run E2, E4, E5 on first batch | First Recall vs bit‑length curves |
| **5–6** | Run E3, E6 (PQ baselines); expand E5 to all datasets | Binary vs PQ comparison tables |
| **7–8** | Run E7, E8 (geometry diagnostics); checkpoint CP2/CP3 | Correlation analysis draft |
| **9–10** | Optional experiments (E10–E13 if time permits); polish plots | Near‑final trade‑off curves |
| **11–12** | Write Phase‑2 final report; documentation | Draft report |
| **13–14** | Review, iterate, finalize | `phase2_final_report.md`, notebooks, code cleanup |

---

## 10. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Binary recall universally poor** | Medium | High | Embrace as negative result; document thoroughly; pivot analysis to "why it fails." |
| **Embedding generation bottleneck** | Low | Medium | Use pre‑computed embeddings from Hugging Face Hub where available; prioritize smaller models first. |
| **PQ baseline misconfiguration** | Medium | High | Use Faiss reference configs; sanity‑check against published benchmarks (e.g., BEIR leaderboard). |
| **Scope creep into engine building** | Medium | Medium | Strict checklist of non‑goals; no distributed systems, no production APIs. |
| **Time overrun** | Medium | Medium | Early checkpoints (CP1–CP3); cut optional experiments if behind schedule. |

---

## 11. Conclusion

This scientific experiment plan operationalizes the Strategist's Phase‑2 choice into a rigorous, falsifiable, and executable research program. The study is:

- **Scoped** to fit within 65 A100‑hours (well under budget) and 14 weeks.
- **Honest** about expected limitations of binary retrieval.
- **Structured** with explicit hypotheses, metrics, baselines, and kill‑switch criteria.
- **Aligned** with Phase‑1's frozen, deterministic infrastructure.

The primary deliverable is a **comprehensive empirical map** of binary retrieval trade‑offs for LLM embeddings—a publishable, portfolio‑defining artifact that serves both researchers and practitioners.

---

*End of Phase‑2 Scientific Experiment Plan (Iteration 3)*


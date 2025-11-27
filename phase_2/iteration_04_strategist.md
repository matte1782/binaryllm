## Executive Summary

- **Core Track stays: binary retrieval empirical study, slightly pruned in execution but fully aligned with Hostile v3.**  
  Phase‑2 remains an honest, LLM‑centric **quality–memory characterization of Phase‑1 binary codes vs PQ/INT8** on three locked datasets and three locked embedding models, with E1–E8 as the core experiment set and no production‑engine or fantasy speedup claims.
- **High‑upside main track for Phase‑3/4: quantization‑friendly / binary‑friendly embeddings (Arc A).**  
  The long‑term arc is to **train or adapt encoders whose geometry is inherently more binarization‑friendly**, enabling cheaper retrieval, on‑device search, and specialized embedding services; this is scientifically meaningful and product‑relevant while remaining feasible to prototype on modest compute.
- **Phase‑2 Exploratory Track: Binary Friendliness Index + light adaptation experiments.**  
  On top of the core benchmarks, Phase‑2 adds a small **Exploratory Track** that (i) defines a **Binary Friendliness Index (BFI)** for each (model, dataset), (ii) learns simple predictors from geometry to binary quality, and (iii) runs **tiny, embedding‑level adapters** on 1–2 dataset/model pairs to probe how easily binary performance can be improved without heavy training.
- **Core scope is execution‑pruned, not conceptually reduced.**  
  Datasets, models, bit‑lengths, and PQ configs remain as in Hostile v3, but we adopt a **tiered execution plan**: full sweeps on a 2×3 “Tier‑1” matrix (2 models × 3 datasets), lighter coverage for the largest model (BGE‑large), and only if there is slack do we expand to the full 3×3 matrix and optional extras.
- **Compute & risk: still well within bounds; exploratory is strictly secondary.**  
  Updated estimate is **≈30–40 A100‑hours for Core** plus **≈5–10 A100‑hours for Exploratory**, over **12–14 weeks part‑time**. Exploratory has explicit kill‑switches and is treated as optional if Core falls behind, ensuring Phase‑2 remains feasible for a single student while seeding a serious Phase‑3/4 main track.

---

## Iteration‑3 Baseline (Where We Are)

### Approved Phase‑2 Core Plan

From Iteration‑3 Strategist + Scientist:

- **Direction:**  
  > **Binary Retrieval with LLM Embeddings: An Empirical Study of Quality–Memory Trade‑offs**
- **Role of Phase‑1:**  
  Phase‑1 is a **frozen, deterministic binary embedding + evaluation engine**. Phase‑2 treats it purely as a **calibrated measurement instrument** (black‑box library) for generating binary codes and computing retrieval/similarity metrics.
- **Scientific goals:**  
  - Produce the first **systematic, reproducible study** of how 1‑bit (and few‑bit) binary embeddings from Phase‑1 compare against **PQ and INT8** compression for modern LLM embeddings.  
  - Map **quality–memory trade‑offs** across multiple bit‑lengths, datasets, and embedding models.  
  - Analyze how **embedding geometry (anisotropy, outliers, intrinsic dim, angular spread)** correlates with binarization quality.  
  - Document **failure modes** where binary retrieval collapses.
- **Core experimental ingredients (Iteration‑3 Scientist):**  
  - **Datasets (pre‑Hostile‑prune):** MS MARCO (subsampled), Natural Questions (subsampled), SciFact, STS Benchmark.  
  - **Embedding models:** `all‑MiniLM‑L6‑v2`, `e5‑base‑v2`, `bge‑large‑en‑v1.5`.  
  - **Bit‑lengths:** 64, 128, 256, 512 bits.  
  - **Baselines:** float32 brute‑force, INT8, PQ‑8/16/32/64 bytes per vector, potentially OPQ/ScaNN.  
  - **Metrics:** Recall@k, nDCG@k, Overlap@k, MRR, memory per vector; secondary QPS/latency (later demoted).  
  - **Experiments E1–E8:** embedding generation, binarization sweeps, PQ index building, retrieval evaluations, geometry diagnostics, correlation analysis.
- **Framing:**  
  - No claim that binary codes will dominate PQ/ScaNN.  
  - No production engine promises; only **benchmarking code + small demos**.  
  - Negative results are acceptable and even valuable.

### Hostile v3 Constraints and Locked Components

Hostile Iteration‑3 delivers an **“APPROVE WITH CONSTRAINTS”** verdict and tightens the plan:

- **Scope and novelty constraints**
  - Acknowledge that **scientific novelty is modest**: this is a **useful empirical characterization**, not a NeurIPS‑level breakthrough.  
  - Soften language from “first modern, LLM‑centric map” to “a systematic, reproducible study of binary retrieval for LLM embeddings.”
- **Dataset & model constraints (locked after edits)**  
  - **Datasets (3):**  
    - MS MARCO (≈500K passage subsample).  
    - Natural Questions (≈500K passage subsample).  
    - SciFact.  
  - **Models (3):**  
    - `all‑MiniLM‑L6‑v2`.  
    - `e5‑base‑v2`.  
    - `bge‑large‑en‑v1.5`.
- **Method/config constraints (locked)**
  - **Bit‑lengths:** 64, 128, 256, 512 bits.  
  - **PQ configs:** 8, 16, 32, 64 bytes/vector.  
  - **Core experiments:** E1–E8 remain; **E9 (throughput/QPS) is demoted to optional/appendix**.  
  - **Geometry correlation:** downgraded from hard hypothesis H2.3 to **“Exploratory analysis: Geometry–quality correlation”** with softer success criteria.  
  - **Float retrieval:** must use **Faiss `IndexFlatIP`** (or equivalent) for tractable brute‑force, not naive O(N²) loops.
- **Non‑negotiable global constraints**
  - **Phase‑1 freeze:** no changes to Phase‑1 code, tests, golden artifacts.  
  - **No KV‑cache, no binary latent blocks, no production engine, no fantasy speedups.** Those ideas are explicitly **dead** for Phase‑2.  
  - **Compute:** ≤65 A100‑hours planned for Phase‑2 core, under an overall ≤150–200 A100‑hour envelope; single student, part‑time.  
  - **Timeline:** ≈14 weeks part‑time with explicit kill‑switch and success criteria.

### Author’s New Desire in Iteration 4

The author now wants to **upgrade the role of Phase‑2**:

- Phase‑2 must **remain feasible** for a single student (no drift into LLM‑scale training, no binary KV‑cache, no production engine).  
- But it should also act as a **launchpad toward a much higher‑upside main track** for Phase‑3/4—something like:  
  - Quantization‑aware / binary‑friendly embeddings.  
  - Multi‑precision memory stacks (2–4 bit KV‑cache, hybrid attention pathways).  
  - Binary + PQ hybrid indexing toolkits that could one day be productized.  
- The key change: **Phase‑2 should not be “just” a characterization**; it must produce **data, tools, and signals** that clearly point to **one or more serious, monetizable, research‑worthy Phase‑3/4 arcs**, without exploding compute or complexity.

---

## Phase‑3/4 Main‑Track Arcs

This section proposes **three high‑upside arcs** that Phase‑2 can feed, then will be narrowed to a primary arc for the Exploratory Track.

### Arc A — Quantization‑Friendly / Binary‑Friendly Embedding Training

#### Long‑Term Vision (Phase‑3/4)

- **Goal:** Design and train **embedding models whose geometry is intrinsically compatible with extreme compression** (binary or ultra‑low‑bit) while preserving downstream retrieval quality.  
- **Phase‑3/4 activities could include:**
  - **Quantization‑aware training** (QAT) of sentence/retrieval encoders with losses that directly reflect **Hamming‑space retrieval objectives**.  
  - Learning **projections or adapters** jointly with encoders so that the final embedding space has:  
    - Controlled anisotropy (less cone collapse).  
    - Stable sign patterns under noise.  
    - Good separation between near and far neighbors in Hamming space.  
  - Exploring **multi‑head / multi‑space encoders** where one branch is optimized for float/PQ search and another for binary search.  
  - Training **small, on‑device‑friendly embedding models** that assume binary storage from day zero.

#### Why It Is Scientifically Meaningful and Monetizable

- **Scientific value**
  - Connects classic LSH / binary hashing theory with **modern LLM embedding pathologies** (anisotropy, hubness) and shows how to **shape geometry** for binary codes.  
  - Naturally builds on Phase‑2’s diagnostics: moving from “which embeddings are good/bad?” to “how can we train embeddings to be good?”  
  - Fits in a rich but still evolving literature on **quantization‑aware representations**, with room for publishable contributions (e.g., new loss functions, new diagnostics, new training recipes).
- **Monetization / product potential**
  - A **binary‑friendly embedding service** that halves or quarters memory vs mainstream embeddings with acceptable quality is directly monetizable in:  
    - Cost‑sensitive retrieval/RAG backends.  
    - On‑device or edge search where RAM is tight.  
  - Easier to productize than core infrastructure: can be shipped as **“drop‑in embeddings”** for existing vector DBs.  
  - Strong fit for consulting or internal infra roles: “we can adapt your retrieval models to be binary‑friendly.”

#### How Phase‑2 Can Feed This Arc

- Phase‑2 core already:
  - Evaluates multiple **pre‑trained embeddings** under binary compression vs PQ/INT8, yielding **rich performance matrices**.  
  - Computes **geometry diagnostics** (anisotropy, outliers, intrinsic dim, angular spread).  
  - Logs **full trade‑off curves** for each (model, dataset, bit‑length).
- With a small extra layer (Exploratory Track), Phase‑2 can:
  - Define a **Binary Friendliness Index (BFI)** per embedding/dataset, giving a scalar target for Phase‑3 training.  
  - Learn **predictors from geometry → binary performance**, informing Phase‑3 objectives (e.g., “reduce anisotropy” or “control outlier fraction”).  
  - Run small **adapter experiments** that test whether binary performance is easily recoverable with modest changes—an important feasibility signal for Phase‑3.

### Arc B — Multi‑Precision LLM Memory Stack (2–4 Bit KV‑Cache / Attention Paths)

#### Long‑Term Vision (Phase‑3/4)

- **Goal:** Design a **multi‑precision memory stack** for LLM inference, where:  
  - KV‑cache entries, attention subpaths, or FFN activations are stored in **2–4 bits** (not 1‑bit) when possible.  
  - Outlier channels or sensitive layers remain at higher precision (8–16 bits).  
  - The system dynamically selects precision per layer or token based on error tolerances.
- **Phase‑3/4 activities could include:**
  - Training or fine‑tuning 100M–1B parameter models with **mixed‑precision KV‑cache** and **smooth quantization** schemes.  
  - Designing **error models** for attention and FFNs that predict which components tolerate low‑bit storage.  
  - Implementing **runtime policies** for when to store/reconstruct KV entries at different precisions.

#### Why It Is Scientifically Meaningful and Monetizable

- **Scientific value**
  - Extends existing work (LLM.int8, SmoothQuant, AWQ) into the **KV‑cache / activation storage** regime with more aggressive quantization.  
  - Directly addresses GPU‑level constraints: memory footprint, cache behavior, throughput vs precision trade‑offs.  
  - Could produce new algorithms for **layer‑wise or head‑wise precision allocation**.
- **Monetization / product potential**
  - A **KV‑cache optimizer** that halves or quarters memory for inference is immediately valuable to any company running large LLMs.  
  - Could be integrated into serving stacks (vLLM, TensorRT‑LLM, Triton) as a feature: “multi‑precision cache with guaranteed quality bounds.”

#### How Phase‑2 Can Feed This Arc

- Indirectly, Phase‑2:
  - Builds expertise in **binary vs low‑bit trade‑offs** and **geometry‑driven error behavior**.  
  - Provides measurement and logging patterns that would be re‑used for KV‑cache experiments.  
  - Establishes the author as credible in **binary quantization** and **evaluation rigor**.
- However:
  - Phase‑2 does not manipulate **internal transformer states**.  
  - The leap from **post‑hoc binary embeddings** to **KV‑cache quantization** is substantial.  
  - A Phase‑2 exploratory track in this direction would risk violating compute constraints or re‑entering the territory Hostile v2 killed.

### Arc C — Binary + PQ Hybrid Indexing & “Compression Studio” for Mid‑Scale RAG

#### Long‑Term Vision (Phase‑3/4)

- **Goal:** Build a **hybrid indexing toolkit** where binary codes, PQ codes, and possibly scalar quantization are combined to:  
  - Serve **mid‑scale RAG use cases** (10M–1B vectors) with flexible memory/latency/quality trade‑offs.  
  - Provide an interactive **“compression studio”** that lets practitioners explore configurations (bits, PQ params, re‑ranking depth) and see cost vs quality curves.
- **Phase‑3/4 activities could include:**
  - Efficient **two‑stage or multi‑stage indexes** that use binary codes for coarse search and PQ/float for refinement.  
  - Advanced **OPQ/ScaNN‑style baselines** plus binary hybrid variants.  
  - Possibly shipping a **SaaS evaluation and tuning platform** for vector DB configurations.

#### Why It Is Scientifically Meaningful and Monetizable

- **Scientific value**
  - Stronger on **systems/engineering** than pure ML: careful exploration of hybrid indexes, index composition, and tuning strategies.  
  - Could formalize **configuration search spaces** and propose automatic optimizers.  
  - Leverages Phase‑2’s benchmarking infrastructure directly.
- **Monetization / product potential**
  - A **Compression Studio** or “vector DB tuning copilot” is extremely legible commercially.  
  - Could integrate with existing engines (Faiss, Milvus, Pinecone, Weaviate, etc.) as an external optimizer or plugin.

#### How Phase‑2 Can Feed This Arc

- Phase‑2’s core:
  - Already builds a **benchmark harness** around binary vs PQ.  
  - Produces the **trade‑off curves and hero figures** needed to motivate a Compression Studio.  
  - Can easily be extended in Phase‑3/4 to include more baselines (OPQ, ScaNN) and additional index configurations.
- However:
  - This arc pulls the project back toward **“engine territory”**, which Hostile v2 correctly flagged as dangerous at student scale.  
  - It is more about **glue + systems** than novel ML, which may not fully leverage the research potential of Phase‑2’s geometric analysis.

---

## Chosen Arc & Phase‑2 Exploratory Track

### Chosen Main Track: Arc A — Quantization‑Friendly / Binary‑Friendly Embeddings

Given the constraints and goals:

- **Arc A** is the best fit for the author:
  - **Aligned with Phase‑2 core:** It is a **direct continuation** of the binary retrieval study—same embeddings, same metrics, same geometry diagnostics.  
  - **Feasible at student scale:** Early experiments can be done at the **embedding level** (no backprop through large encoders) using cached embeddings from Phase‑2.  
  - **High‑upside:** Successful techniques for binary‑friendly embeddings translate naturally into monetizable embedding services and productized models.  
  - **Scientifically grounded:** Builds on solid theory (LSH, anisotropy) and empirical observations from Phase‑2, offering multiple avenues for publishable work.
- **Arc B** is too far from Phase‑2’s scope and too compute‑intensive for a single student, given Hostile v2’s strong criticisms of binary KV‑cache.  
- **Arc C** is attractive for systems/product, but risks pulling Phase‑3/4 back into engine building; better as a **secondary direction** that builds on Arc A’s improved embeddings.

Therefore, **Arc A** is selected as the **primary long‑term direction** that Phase‑2 should seed.

### Phase‑2 Exploratory Track: Binary Friendliness Index & Light Adaptation

#### Objectives

Design a small, bounded Exploratory Track that:

- **Reuses Phase‑2 Core artifacts** (embeddings, binary codes, geometry diagnostics, retrieval metrics) with minimal extra compute.  
- Produces a **Binary Friendliness Index (BFI)** and related tools that:  
  - Score (model, dataset) pairs by how suitable they are for binary encoding.  
  - Provide **targets and diagnostics** for Phase‑3 training.  
- Runs **tiny adaptation experiments** on top of frozen embeddings to probe whether binary performance is **easily recoverable** with simple projections/adapters.  
- Stays within **≈5–10 extra A100‑hours** and modest CPU time.

#### Leveraging the Core Track

The Exploratory Track is intentionally parasitic on the Core:

- Uses **the same embeddings** generated in E1.  
- Uses **the same binary codes** and retrieval metrics from E2/E5.  
- Uses **the same geometry diagnostics** from E7.  
- Adds only:
  - Lightweight analysis scripts.  
  - One small training loop on cached embeddings.  
  - A few additional evaluation runs on already‑existing codes.

No changes to Phase‑1 are required; all new work lives in **Phase‑2 notebooks/scripts**.

#### Exploratory Components

We define four exploratory experiments (X1–X4).

##### X1 — Define a Binary Friendliness Index (BFI)

- **Idea:** Collapse multiple performance and geometry signals into a single **scalar index** per (model, dataset) capturing “how naturally this embedding binarizes.”  
- **Inputs (from Core):**
  - Recall@10 at **256 and 512 bits** for each (model, dataset).  
  - Ratio of binary Recall@10 vs PQ Recall@10 at iso‑memory (e.g., 256 bits vs PQ‑32).  
  - Geometry diagnostics: anisotropy, intrinsic dim, outlier fraction, mean pairwise cosine.  
- **BFI design (example, tunable):**
  - Start with a **normalized recall score** at 256/512 bits and a **normalized gap to PQ**.  
  - Optionally incorporate geometry as penalties (e.g., high anisotropy reduces BFI).  
  - Ensure the formula is **transparent and documented**, not a black‑box ML model.
- **Outputs:**
  - A **BFI table** for all (model, dataset) combos in the Core matrix.  
  - Plots of BFI vs geometry metrics.  
  - This becomes a **Phase‑3 target metric** (“we want new embeddings with BFI ≥ X on Y dataset”).
- **Compute cost:** negligible (pure analysis on existing results).

##### X2 — Geometry → Binary Quality Predictors (Exploratory Regression)

- **Idea:** Test whether simple models can **predict binary recall from geometry alone**, across (model, dataset, bit‑length) points.  
- **Inputs:**
  - Geometry features per (model, dataset) (E7).  
  - Binary Recall@10 at various bit‑lengths (E5).  
  - Derived features like slopes of recall vs bits.  
- **Method:**
  - Fit **very small models** (ridge regression, Lasso, maybe a tiny MLP) on the point set:  
    - Each data point: (geometry features, bit‑length) → (Recall@10 or normalized BFI).  
  - Use **cross‑validation** within this small dataset to avoid overfitting illusions.  
  - Check whether any feature combination provides meaningful predictive power.
- **Outputs:**
  - Reported R² / correlation scores, with **clear caveats** about small‑N.  
  - If any robust patterns appear (e.g., high anisotropy predicts poor binary performance), they become **Phase‑3 design constraints**.  
  - If not, we record this as a negative result and treat geometry as a **weak prior**, not a strong predictor.
- **Compute cost:** negligible (CPU‑only, small datasets).

##### X3 — Light Adaptation: Binary‑Friendly Projection on Cached Embeddings

- **Idea:** On **one dataset (MS MARCO subsample)** and **one or two models (MiniLM, e5‑base)**, train a **small projection or adapter** on cached float embeddings to see whether a modest, embedding‑level change can significantly improve binary retrieval.
- **Setup:**
  - Freeze the base encoder; work entirely on **cached float embeddings** (no backprop through the transformer).  
  - Define a projection \( W \) (e.g., linear or shallow MLP) mapping \( \mathbb{R}^d \to \mathbb{R}^{d'} \) or \( \mathbb{R}^d \to \mathbb{R}^d \).  
  - Binarize \( W x \) using Phase‑1’s pipeline (L2‑normalize, random Gaussian projections, sign).  
  - Train \( W \) using a **proxy loss** that encourages Hamming‑neighbor structure to match float‑cosine neighbors, e.g.:  
    - Contrastive loss where positives are float‑top‑k neighbors, negatives are random non‑neighbors.  
    - Or directly optimize Overlap@k / Recall@k using a differentiable surrogate.
- **Constraints:**
  - Work on a **small subset** (e.g., 50K–100K documents and a few thousand queries) to keep training and evaluation cheap.  
  - Keep the projection size small (e.g., one or two linear layers with ≤1–2M parameters).  
  - Limit hyperparameter search to a few runs with fixed seeds; no broad sweeps.
- **Outputs:**
  - Comparison of:  
    - Baseline binary retrieval (no adapter) vs. adapted version at 128/256 bits.  
    - PQ baselines at comparable memory.  
  - Insight into **how much headroom exists** for learning‑based improvements **without touching the encoder**.  
  - Ground truth for Arc A: if even simple adapters yield noticeable gains, it suggests Phase‑3/4 could do much better with modest compute.
- **Compute cost (rough):**
  - Training on cached embeddings is cheap: likely **1–3 A100‑hours** (or even CPU‑only with patience) per configuration.  
  - With a tightly scoped experiment (1 dataset × 1–2 models), the Exploratory Track can stay within **≈5 A100‑hours** total.

##### X4 — Design Guidelines and Metrics for Phase‑3

- **Idea:** Synthesize X1–X3 into **concrete design rules** and metrics for Phase‑3/4 embedding work.  
- **Outputs:**
  - A **short design memo** (could live in `phase2_master_plan.md` or a dedicated `phase2_arcA_guidelines.md`) specifying:  
    - Target ranges for anisotropy, outlier fraction, etc., if they correlate with BFI.  
    - A working definition of **BFI** and recommended thresholds for “good” vs “bad” embeddings.  
    - Lessons from light adaptation: e.g., “for MS MARCO, a simple adapter recovers X% of binary vs PQ gap.”  
  - This memo becomes the **bridge artifact**: when Phase‑3 begins, it is the starting spec.
- **Compute cost:** none beyond previous X1–X3.

#### Realism Check (Single Student, Limited Compute)

- **Most of the Exploratory Track is analysis‑only.**  
  X1, X2, X4 are essentially free once core results exist.  
- **The only non‑trivial compute is X3,** which can be tightly budgeted to **≈5 A100‑hours** or less and can even fall back to CPU for small‑scale prototypes.  
- All work is **modular**: if Core slips, the Exploratory Track can be partially executed (e.g., only X1+X2) or even paused entirely without invalidating Core results.

---

## Updated Core vs Exploratory Plan

### Re‑scoping the Core Track (Execution‑Level, Not Conceptual)

To make room for the Exploratory Track while staying sane:

- We **do not change the locked conceptual scope** (3 datasets, 3 models, 4 bit‑lengths, 4 PQ configs, E1–E8).  
- Instead, we introduce a **tiered execution plan**:
  - **Tier‑1 (must‑have coverage):**  
    - Models: `all‑MiniLM‑L6‑v2`, `e5‑base‑v2`.  
    - Datasets: MS MARCO (500K), NQ (500K), SciFact.  
    - Full matrix: all 4 bit‑lengths × all 4 PQ configs, plus geometry diagnostics and core metrics.  
  - **Tier‑2 (nice‑to‑have coverage):**  
    - Model: `bge‑large‑en‑v1.5`.  
    - Datasets: MS MARCO + SciFact (NQ with BGE‑large only if time permits).  
    - Possibly reduced sweep (e.g., prioritize 128/256/512 bits and PQ‑16/32 for iso‑memory story, with geometry diagnostics).  
  - **Tier‑3 (stretch extensions):**  
    - Additional OPQ baseline, re‑ranking study, extended BGE‑large coverage, or more datasets/models.
- This keeps the **core story intact** (binary vs PQ across multiple models/datasets) while acknowledging realistic **time and compute limits**.

### Experiment Table: Core vs Exploratory

**Core Experiments (post‑Hostile, execution‑pruned):**

| ID | Type | Description | Scope (min) | Est. GPU‑h | Est. CPU‑h |
|----|------|-------------|-------------|------------|------------|
| E1 | Core | Embedding generation | Tier‑1 (2 models × 3 datasets) + Tier‑2 (BGE‑large on MS MARCO + SciFact) | 18–25 | 0 |
| E2 | Core | Binarization sweeps (64/128/256/512 bits) | Same as E1 | 0 | 5–10 |
| E3 | Core | PQ index building (8/16/32/64 bytes) | Same as E1 | 5–8 | 5–8 |
| E4 | Core | Float brute‑force retrieval via Faiss | Same as E1 | 0 | 30–50 |
| E5 | Core | Binary retrieval evaluation | Same as E1 | 0 | 15–25 |
| E6 | Core | PQ retrieval evaluation | Same as E1 | 0 | 15–25 |
| E7 | Core | Geometry diagnostics | Same as E1 | 0 | 5–8 |
| E8 | Core | Correlation / analysis (exploratory) | Same as E1 | 0 | 3–5 |

**Exploratory Experiments (new, high‑upside, Arc A‑aligned):**

| ID | Type | Description | Scope | Est. GPU‑h | Est. CPU‑h |
|----|------|-------------|-------|------------|------------|
| X1 | Exploratory | Define Binary Friendliness Index (BFI) | All Tier‑1 (and Tier‑2 if available) | 0 | 3–5 |
| X2 | Exploratory | Geometry → binary quality predictors | All Tier‑1 (and Tier‑2 if available) | 0 | 3–5 |
| X3 | Exploratory | Light adaptation on cached embeddings | MS MARCO × (MiniLM, e5‑base) | 3–5 | 5–10 |
| X4 | Exploratory | Design guidelines & Phase‑3 spec | Synthesis of Core + X1–X3 | 0 | 5–8 |

### Success Criteria: Core vs Exploratory

#### Core Track Success (unchanged in spirit, slightly relaxed in execution)

Core is successful if:

- **Coverage:**  
  - Full E1–E8 completed on **Tier‑1 matrix** (2 models × 3 datasets).  
  - At least partial coverage for **BGE‑large** on MS MARCO + SciFact (Tier‑2).
- **Trade‑off curves & tables:**  
  - Clear, reproducible plots for Recall@10 vs memory, binary vs PQ, for all Tier‑1 combos.  
  - At least **2–3 hero figures** that summarize the story.  
- **Honest conclusions:**  
  - Explicit, quantitative identification of regimes where binary is: clearly worse, competitive, or potentially preferable.  
  - Negative results are embraced and documented.  
- **Reproducibility:**  
  - A new user can reproduce a subset of key plots on at least one dataset/model pair using documented scripts.

#### Exploratory Track Success

Exploratory is successful if:

- **BFI & diagnostics (X1/X2):**  
  - A well‑defined **Binary Friendliness Index** is computed for all Tier‑1 pairs.  
  - At least **some interpretable relationships** between geometry and BFI are identified (even if modest), or a clear negative result is documented.  
- **Light adaptation (X3):**  
  - At least one adapter configuration on MS MARCO × (MiniLM or e5‑base) is trained and evaluated.  
  - We obtain **quantitative evidence** of whether simple adapters can close a meaningful fraction of the binary vs PQ gap at fixed memory.  
- **Bridge document (X4):**  
  - A short, concrete design memo exists that **translates Phase‑2 findings into Phase‑3 targets**, including BFI thresholds, geometry heuristics, and adapter lessons.

If Core succeeds but Exploratory fails or is dropped, **Phase‑2 is still a success** as a high‑quality empirical study; Exploratory is explicitly treated as **bonus upside**.

---

## Compute & Risk Update

### Updated Compute & Time Estimates

**Core Track (Tier‑1 + minimal Tier‑2):**

- **GPU:**  
  - E1 (embeddings): 18–25 A100‑hours.  
  - E3 (PQ training/building): 5–8 A100‑hours.  
  - **Total Core GPU:** ≈23–33 A100‑hours.  
- **CPU:**  
  - Retrieval + geometry + analysis: ≈80–120 CPU‑hours (parallelizable across days; can run overnight).  
- **Time (single student, part‑time):**  
  - ≈10–12 weeks for Core, within the 14‑week envelope, leaving slack for reporting and minor extensions.

**Exploratory Track:**

- **GPU:**  
  - X3 (light adaptation): 3–5 A100‑hours (upper bound, can be lower if CPU‑only).  
  - X1/X2/X4: 0 GPU.  
  - **Total Exploratory GPU:** ≈3–5 A100‑hours.  
- **CPU:**  
  - Analysis and small models: ≈20–30 CPU‑hours.  
- **Time:**  
  - ≈2–3 weeks, interleaved with late Core analysis/reporting.

**Grand Total (Core + Exploratory):**

- **GPU:** ≈26–38 A100‑hours, well under the 65 A100‑hour Phase‑2 plan and far below the global 150–200 A100‑hour bound.  
- **Timeline:** ≈12–14 weeks part‑time, with Exploratory explicitly **dropped or frozen** if Core falls behind.

### Top 3 New Risks from the Exploratory Track & Mitigations

1. **Risk: Exploratory results are noisy or inconclusive.**  
   - *Description:* Geometry → quality predictors may be weak; BFI may not correlate cleanly with any simple feature; light adapters may yield only marginal gains.  
   - *Mitigation:*  
     - Treat all Exploratory claims as **exploratory by design**, not central hypotheses.  
     - If X1/X2 produce no strong patterns, document this explicitly and narrow Phase‑3 scope to **data‑driven, trial‑and‑error training** rather than theory‑driven geometry shaping.  
     - If X3 yields small gains, frame them as “upper bounds” on what adapter‑only methods can do.

2. **Risk: Adapter experiments (X3) silently explode in scope.**  
   - *Description:* It is easy to lose time on hyperparameter sweeps, architecture variants, and dataset scaling, turning a small experiment into a pseudo‑Phase‑3 project.  
   - *Mitigation:*  
     - Hard cap: **≤5 A100‑hours and ≤2–3 distinct adapter configs**.  
     - Fix all random seeds and logging from the start; treat the first working setup as the “main result”.  
     - If early runs show no promising signal, **stop X3** and keep Exploratory focused on X1/X2/X4.

3. **Risk: Exploratory delays or distracts from Core.**  
   - *Description:* Enthusiasm for the high‑upside arc could cause the author to start X3 before Core plots are stable, risking incomplete core results.  
   - *Mitigation:*  
     - Enforce a simple rule: **do not start X3 until Tier‑1 Core experiments (E1–E6) are at least 80% complete and basic plots exist**.  
     - Treat Exploratory as an **optional “sprint” in the last 3–4 weeks**; if behind schedule, cancel X3 entirely and execute only X1/X2/X4 (or even just X1).

---

## Unified Phase‑2 Plan (Core + Exploratory)

Bringing everything together, **Phase‑2 Master Plan v2** is:

- **Core Track (locked, honest, empirically focused):**  
  - Deliver a **systematic, reproducible study** of binary retrieval with LLM embeddings across 3 datasets and 3 models, with binary codes vs PQ/INT8 under matched memory budgets.  
  - Produce high‑quality trade‑off curves, failure analyses, and a practitioner‑oriented **decision guide** explaining when binary embeddings are viable.  
  - Stay within ≈23–33 A100‑hours and 10–12 weeks of focused work, using Phase‑1 as a frozen, deterministic backbone.
- **Exploratory Track (small, high‑upside, arc‑aligned):**  
  - Layer on a **Binary Friendliness Index**, geometry‑based diagnostics, and **light adaptation experiments** on cached embeddings.  
  - Use these to generate a clear **Phase‑3/4 spec for quantization‑friendly embeddings (Arc A)**—including target metrics (BFI), geometry heuristics, and empirical evidence of adaptability.  
  - Budget ≈3–5 A100‑hours and 2–3 weeks, with explicit kill‑switches and no impact on the integrity of Core.

**What success looks like overall:**

- The **Core Track** yields a **portfolio‑grade empirical artifact**: plots, tables, code, and a narrative that any NVIDIA/FAANG‑level team can trust to guide binary vs PQ decisions.  
- The **Exploratory Track** yields a **compact, actionable blueprint** for Phase‑3/4: a BFI definition, early geometry insights, and evidence about the ease or difficulty of making embeddings more binary‑friendly.  
- Together, they position the author with:  
  - Demonstrated **research rigor and systems realism** (Core).  
  - A **credible, high‑upside roadmap** toward binary‑aware embedding training and potential embedding products (Exploratory → Arc A) without violating single‑student, limited‑compute constraints.







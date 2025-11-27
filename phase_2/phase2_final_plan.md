## Overview

Phase‑1 is a **frozen, fully‑verified binary embedding and evaluation engine**. Phase‑2 treats it strictly as a **black‑box measurement instrument**: we generate embeddings and binary codes with Phase‑1, then run additional benchmarking, analysis, and light learning code **only under `phase_2/`**, without touching Phase‑1 APIs, tests, or artifacts.

Phase‑2 is now **fully locked** as:

- **Core Track (mandatory):**  
  > **Binary Retrieval with LLM Embeddings: An Empirical Study of Quality–Memory Trade‑offs**  
  A rigorous, LLM‑centric empirical study comparing Phase‑1 binary codes against **PQ/INT8** under matched memory budgets on **3 datasets × 3 models × 4 bit‑lengths**, with geometry diagnostics and explicit failure/success criteria.

- **Exploratory Track (small, secondary):**  
  > **Binary Friendliness Index + Light Adaptation for Arc A (Quantization‑Friendly Embeddings)**  
  A bounded, analysis‑heavy track (X1–X4) that reuses Core artifacts to (i) define a **Binary Friendliness Index (BFI)**, (ii) test whether geometry predicts binary quality, and (iii) run one small **adapter experiment on cached embeddings** to probe feasibility for Phase‑3/4.

**Budget and scope (locked):**

- **Datasets:** MS MARCO (≈500K passages), Natural Questions (≈500K passages), SciFact.  
- **Embedding models:** `all‑MiniLM‑L6‑v2`, `e5‑base‑v2`, `bge‑large‑en‑v1.5`.  
- **Bit‑lengths:** 64, 128, 256, 512 bits (Phase‑1 binary pipeline).  
- **PQ configs:** 8, 16, 32, 64 bytes/vector (Faiss PQ; OPQ only if time permits as stretch).  
- **GPU budget:** **Target 30–40 A100‑hours**, hard cap **≤65 A100‑hours** for Phase‑2.  
- **Timeline:** **≤14 weeks part‑time** (Core ≈10–12 weeks, Exploratory ≈2–3 weeks).  
- **Determinism:** All non‑deterministic GPU work is confined to Phase‑2; Phase‑1 tests and golden artifacts remain untouched.

Phase‑2 **does not** promise a production‑grade search engine, revolutionary speedups, or “first‑ever” novelty. It aims for a **portfolio‑grade empirical artifact + a concrete Phase‑3/4 launchpad**, under honest, NVIDIA‑grade realism.

---

## Core Track (E1–E8)

Locked scope: empirical binary retrieval study across Tiered coverage (see “Execution Tiers and Kill‑Switches”).

### E1 — Embedding Generation (Tier‑1)

- **What it does:**  
  Generates and caches float32 L2‑normalized embeddings for all (dataset, model) pairs using frozen public encoders (MiniLM, e5‑base, BGE‑large).
- **Question:**  
  Can we obtain a **clean, reusable embedding corpus** for all planned Phase‑2 experiments within **≤25 A100‑hours** while respecting data and Phase‑1 constraints?
- **Metrics:**  
  GPU hours used, number of vectors per (dataset, model), checksum/shape validation, wall‑clock latency per 1M vectors.
- **Target bands (execution‑level):**
  - **Hope:** All Tier‑1 and Tier‑2 embeddings cached with **≤20 A100‑hours**, no major re‑runs.  
  - **Still useful:** Tier‑1 fully cached, Tier‑2 partially covered, **≤30 A100‑hours** total.  
  - **Negative:** Embedding generation repeatedly fails or exceeds **35–40 A100‑hours**, forcing reductions in model/dataset coverage.

### E2 — Binarization Sweep (Tier‑1)

- **What it does:**  
  Applies Phase‑1’s deterministic binarization (random Gaussian projections + sign) to all cached embeddings at **64/128/256/512 bits**, producing packed codes and code‑length metadata.
- **Question:**  
  How do **code length and memory per vector** interact, and can we generate stable, deterministic binary codes across all planned bit‑lengths without numerical or storage issues?
- **Metrics:**  
  Memory per vector (bytes), total storage, generation throughput, sanity checks (Hamming self‑distance = 0, reproducibility under fixed seeds).
- **Target bands (quality‑adjacent):**
  - **Hope:** Codes stable and reproducible; no pathological failures; Hamming distance distributions look reasonable for all bit‑lengths.  
  - **Still useful:** Minor quirks (e.g., slightly skewed distributions) but no blocking issues; codes usable for retrieval.  
  - **Negative:** Systematic failures (e.g., frequent collisions, non‑reproducible codes) that invalidate downstream evaluation.

### E3 — PQ Index Building (Tier‑1)

- **What it does:**  
  Builds Faiss PQ/OPQ indices at **8/16/32/64 bytes/vector** for each Tier‑1 (dataset, model), using subsampled training sets and documented configurations.
- **Question:**  
  Can we create **strong, fair compression baselines** at matched memory budgets without over‑tuning, and within the GPU budget?
- **Metrics:**  
  PQ training time, index build time, memory per vector, convergence diagnostics (Faiss training loss), sanity checks vs float32 brute‑force on small subsets.
- **Target bands:**
  - **Hope:** PQ indices train reliably with small subsamples and produce **competitive recall (≥0.8 of float) on at least 2/3 datasets** at 32–64 bytes.  
  - **Still useful:** PQ indices are stable but underperform slightly vs expectations; we still get clean recall vs memory curves.  
  - **Negative:** PQ indices are unstable or extremely weak (e.g., Recall@10 < 0.6 at 64 bytes across the board), making binary vs PQ comparisons meaningless.

### E4 — Float Brute‑Force Retrieval via Faiss (Tier‑1)

- **What it does:**  
  Uses `faiss.IndexFlatIP` (or equivalent) to compute **float32 brute‑force nearest neighbors** for each (dataset, model), serving as the ground‑truth ranking under cosine similarity.
- **Question:**  
  Can we obtain **trusted float baselines** (top‑k neighbors per query) at realistic compute cost that anchor all Phase‑2 quality metrics?
- **Metrics:**  
  Recall@k / nDCG@k of PQ and binary vs float, Q×N FLOP estimates, CPU time used, verification on held‑out subsets vs existing metrics in Phase‑1 where applicable.
- **Target bands:**
  - **Hope:** Full top‑k ground truth for Tier‑1 within **30–50 CPU‑hours**, no correctness doubts (sanity checks all pass).  
  - **Still useful:** Some minor compromises (e.g., fewer queries for the largest model) but float baselines are clearly trustworthy.  
  - **Negative:** Float baselines are too expensive to compute or show inconsistencies, undermining all downstream comparisons.

### E5 — Binary Retrieval Evaluation (Tier‑1)

- **What it does:**  
  Runs Hamming‑based top‑k search on Phase‑1 binary codes (64–512 bits) using CPU and/or simple GPU kernels, logging Recall@k, nDCG@k, Overlap@k vs float baselines.
- **Question:**  
  For each (dataset, model, bit‑length), **how much quality is lost** when moving from float/PQ to pure binary retrieval under matched memory?
- **Metrics:**  
  Recall@{1,5,10,50,100}, nDCG@10, Overlap@10, memory per vector, evaluation time (for context only; no speedup promises).
- **Target bands (quality):**
  - **Hope:**  
    - On **≥2 of 3 datasets**, for **≥2 of 3 models**, binary‑256/512 achieves **Recall@10 ≥ 0.70**.  
    - At iso‑memory vs PQ (e.g., 256 bits vs PQ‑32), binary achieves **≥0.80×** PQ Recall@10 on **≥1 dataset**.  
  - **Still useful:**  
    - Binary‑256/512 Recall@10 mostly in **0.60–0.70** range, or strong on some datasets and weak on others.  
    - Iso‑memory ratios in **0.60–0.80×** PQ range, clearly quantifying where binary is acceptable but not exciting.  
  - **Negative (Core kill‑switch F1/F2):**  
    - **F1:** Even at 512 bits, Recall@10 **never exceeds 0.55** on **any** dataset/model.  
    - **F2:** At matched memory, binary Recall@10 is **<0.5× PQ** on **all** benchmarks.

### E6 — PQ Retrieval Evaluation (Tier‑1)

- **What it does:**  
  Evaluates PQ indices on all Tier‑1 combinations, computing Recall@k / nDCG@k vs float ground truth and logging memory/latency for context.
- **Question:**  
  Where does PQ sit on the **quality–memory frontier** for modern LLM embeddings, and how large is the **binary vs PQ quality gap** at each memory level?
- **Metrics:**  
  Same metrics as E5 (Recall@k, nDCG@k, Overlap@k), plus PQ training/eval time and memory per vector.
- **Target bands:**
  - **Hope:** PQ‑32/64 reaches **Recall@10 ≥ 0.80** on most Tier‑1 pairs; gap vs binary is visible but not catastrophic.  
  - **Still useful:** PQ performs strongly on at least **2 of 3 datasets**; gaps vs binary are clearly measurable even if large.  
  - **Negative:** PQ itself underperforms (e.g., Recall@10 < 0.7 at 64 bytes for most pairs), making the study less informative about realistic trade‑offs.

### E7 — Geometry Diagnostics (Tier‑1)

- **What it does:**  
  Computes **anisotropy, outlier fraction, intrinsic dimensionality, and angular spread** for each (dataset, model) embedding set, using cached float vectors.
- **Question:**  
  What are the **geometric pathologies** of LLM embeddings in practice, and do they qualitatively align with known binary hashing theory (e.g., cone collapse, hubness)?
- **Metrics:**  
  Anisotropy (\(\sigma_1 / \sigma_{\text{median}}\)), outlier fraction, intrinsic dim @ 90% variance, mean pairwise cosine, simple sanity plots.
- **Target bands:**
  - **Hope:** Diagnostics confirm known pathologies (high anisotropy, non‑trivial outlier fraction), and correlate qualitatively with observed binary failures.  
  - **Still useful:** Diagnostics show some variation across models/datasets, even if patterns are noisy.  
  - **Negative:** Metrics are numerically unstable or provide no usable variation, making them unhelpful for analysis.

### E8 — Correlation / Analysis (Tier‑1 but Exploratory in claims)

- **What it does:**  
  Runs **simple regressions** (ridge/Lasso/tiny MLP) from geometry + bit‑length features to Recall@10 / BFI, using LOOCV to avoid overfitting on the small sample.
- **Question:**  
  Do geometry diagnostics meaningfully **predict binary retrieval quality**, or are they weak signals that should only be used as soft priors?
- **Metrics:**  
  LOOCV R², RMSE, Spearman ρ, feature importances, bootstrap confidence intervals.
- **Target bands (exploratory, not success criteria):**
  - **Hope:** LOOCV **R² > 0.3** and **ρ > 0.5**, indicating geometry explains a non‑trivial fraction of variance.  
  - **Still useful:** R² in **0.1–0.3** range; geometry is a weak prior but still informative for Phase‑3 heuristics.  
  - **Negative (Core F3 condition):** |ρ| < 0.3 and R² < 0.1 → geometry provides **no predictive power**; document as a negative result and avoid overclaiming in Phase‑3.

---

## Exploratory Track (X1–X4)

Locked scope: small, parasitic on Core artifacts, strictly secondary to E1–E8.

### X1 — Binary Friendliness Index (BFI) (Tier‑2)

- **What it does:**  
  Defines a **scalar BFI** per (dataset, model) that combines normalized Recall@10 at 256/512 bits, binary/PQ ratios at iso‑memory, and optionally anisotropy penalties.
- **Question:**  
  Can we build a **transparent, stable index** that meaningfully ranks embeddings by how naturally they tolerate binarization?
- **Metrics:**  
  BFI formula, BFI values per Tier‑1 pair, variance of BFI across pairs, monotonicity vs raw Recall@10.
- **Target bands:**
  - **Hope:** BFI spans a **wide range (≥0.3 spread)** across Tier‑1 pairs, and its ranking is almost identical to Recall@10@256/512.  
  - **Still useful:** BFI provides mild separation (spread ≈0.1–0.3) and is at least monotonic with Recall@10@256.  
  - **Negative:** BFI collapses to an almost constant value (spread <0.05) or gives rankings that contradict Recall@10; then we treat it as a cosmetic convenience only.

### X2 — Geometry → Binary Quality Predictors (Tier‑2)

- **What it does:**  
  Fits small predictive models from geometry metrics + bit‑length to Recall@10 / BFI across Tier‑1 (and optionally Tier‑2) points.
- **Question:**  
  Are **simple geometry features** (anisotropy, outliers, intrinsic dim, angular spread) actually useful for predicting binary performance, or are they too noisy?
- **Metrics:**  
  LOOCV R², Spearman ρ, feature coefficients, bootstrapped confidence intervals.
- **Target bands (explicitly exploratory):**
  - **Hope:** LOOCV **R² > 0.3**, **ρ > 0.5**, with interpretable coefficients (e.g., high anisotropy strongly negative).  
  - **Still useful:** Some weak but consistent trends (R² ~0.1–0.3) that justify light geometry regularization in Phase‑3.  
  - **Negative:** No meaningful signal (R² < 0.1, |ρ| < 0.3); we record this and treat geometry as a weak prior only.

### X3 — Light Adaptation on Cached Embeddings (Tier‑3, high‑upside)

- **What it does:**  
  Trains a **small projection/adapter** on cached embeddings (MS MARCO × MiniLM / e5‑base) to see whether a simple learned transform can improve binary Recall@10 at 128/256 bits without touching the encoder.
- **Question:**  
  Is binary performance **easily recoverable** with low‑compute adapters, or does closing the binary vs PQ gap require full encoder retraining?
- **Metrics:**  
  Recall@10 at 128/256 bits (baseline vs adapted vs PQ), fraction of PQ gap closed, changes in geometry (anisotropy, angular spread), GPU hours used.
- **Target bands (Arc‑A feasibility):**
  - **Hope:** For at least one (dataset, model), adapted binary‑256 improves Recall@10 by **≥5 percentage points** vs baseline and closes **≥20–30%** of the binary vs PQ gap at iso‑memory.  
  - **Still useful:** Smaller but non‑trivial gains (≈3–5 pp or ≈10–20% gap closure) that show adapters have *some* leverage but not enough alone.  
  - **Negative (Exploratory kill for X3):** After **1–2 configs**, improvements are **<2 pp** and **<10%** gap closure → simple adapters are likely insufficient; we stop X3 and treat Arc A as requiring deeper changes.

### X4 — Phase‑3/4 Design Guidelines (Tier‑2)

- **What it does:**  
  Synthesizes X1–X3 findings into a **short, actionable design memo** (BFI thresholds, geometry heuristics if any, adapter baselines) that defines Phase‑3 targets for quantization‑friendly embeddings.
- **Question:**  
  Can we turn Phase‑2 results into **concrete, numeric success criteria and baselines** for Phase‑3/4, rather than vague “future work”?
- **Metrics:**  
  Presence and clarity of: BFI definition, recommended thresholds, geometry rules (if any), adapter baseline numbers, prioritized Phase‑3 experiment list.
- **Target bands:**
  - **Hope:** Memo contains **clear thresholds** (e.g., “BFI ≥ 0.65, anisotropy < 15”) and adapter baselines, enabling a sharp Phase‑3 spec.  
  - **Still useful:** Memo mostly summarizes negative results (“geometry not predictive, adapters weak”) but still states this explicitly, preventing wishful thinking.  
  - **Negative:** Memo remains vague (“do more experiments”) without concrete metrics or baselines.

---

## Execution Tiers and Kill‑Switches

### Tiers (Priority and Coverage)

- **Tier‑1 (mandatory Core):**
  - Experiments: **E1–E8**.  
  - Coverage: **MiniLM + e5‑base × {MS MARCO, NQ, SciFact}** (2 models × 3 datasets).  
  - Requirement: Full binary and PQ sweeps at 64–512 bits / 8–64 bytes, geometry diagnostics, and core analysis plots for Tier‑1.

- **Tier‑2 (nice‑to‑have Core + Exploratory):**
  - Core: BGE‑large on **MS MARCO + SciFact** with reduced sweeps if necessary (prioritize 128/256/512 bits, PQ‑16/32).  
  - Exploratory: **X1, X2, X4** (analysis‑heavy, 0 GPU).  
  - Executed only after Tier‑1 is stable and the remaining GPU/time budget is clear.

- **Tier‑3 (stretch):**
  - Exploratory: **X3** (light adaptation) and any optional extras (e.g., OPQ baseline, re‑ranking, extended BGE coverage) if time permits.  
  - Hard cap: **≤5 A100‑hours** and **≤3 adapter configs** for X3.

### Core Kill‑Switches (from Iteration‑3 Scientist, preserved)

- **F1 — Binary quality collapse:**  
  On **all** datasets and models, Recall@10 at **512 bits** is **<0.55** → binary retrieval is too weak even with many bits.  
  **Action:** Stop expansion; document negative result and do not position binary retrieval as a viable alternative in Phase‑3.

- **F2 — Iso‑memory dominance by PQ:**  
  At matched memory (e.g., 64 bits vs PQ‑8), binary Recall@10 is **<0.5× PQ** on **all** benchmarks.  
  **Action:** Same as F1; treat binary codes purely as a reference point, not a candidate for future optimization.

- **F3 — Geometry non‑signal:**  
  E8 shows **no statistically meaningful correlation** (|ρ| < 0.3, R² < 0.1) between geometry features and binary quality.  
  **Action:** Still complete Core, but explicitly label geometry‑based ideas as **weak hypotheses**; Phase‑3 should not rely on them as main levers.

### Exploratory Kill‑Switches (X‑Track)

- **K1 — Core behind schedule:**  
  By week 10, Tier‑1 Core (E1–E6) is **<80% complete**.  
  **Action:** Pause all Exploratory; complete Core; Exploratory becomes optional or dropped.

- **K2 — X3 shows no signal:**  
  After the first adapter configuration on MS MARCO × MiniLM, Recall@10 improvement is **<1 pp** at 256 bits.  
  **Action:** Stop X3 entirely; do not burn more GPU; Exploratory finishes with X1/X2/X4 only.

- **K3 — Geometry predictor failure:**  
  After initial X2 regression, LOOCV **R² < 0.05** and |ρ| < 0.2.  
  **Action:** Document the negative result; avoid spending extra time on more complex models or feature engineering.

These kill‑switches are **binding**: if triggered, we document, stop the relevant work, and do **not** silently expand scope.

---

## GREEN / YELLOW / RED Summary

- **GREEN (Strong success):**
  - Tier‑1 Core completed with clean trade‑off curves; at least some regimes where binary‑256/512 is **≥0.70 Recall@10** and **≥0.8× PQ** on at least one dataset.  
  - Tier‑2 Core mostly done; Exploratory X1/X2/X4 completed; X3 demonstrates **meaningful gains (≥5 pp or ≥20–30% gap closure)** on at least one setting.  
  - Clear, concrete Phase‑3 spec (Arc A) with BFI thresholds and adapter baselines.

- **YELLOW (Useful but limited):**
  - Tier‑1 Core completed; binary often weak but results are honest and reproducible.  
  - PQ dominates in most regimes; binary has niche value only under extreme memory constraints.  
  - Exploratory yields weak or negative signals (e.g., geometry not predictive, adapters marginal), but these are documented and still inform Phase‑3.

- **RED (Kill or archive):**
  - Core hit F1 or F2 (binary clearly unusable) **and** results are too noisy or incomplete to quantify the failure well.  
  - Logging and reproducibility are poor; plots are missing or inconsistent; no clear Phase‑3 pathway is articulated.  
  - In this case, Phase‑2 is treated as a **post‑mortem artifact**, not a foundation for Phase‑3.

---

## Non‑Goals (What Phase‑2 Is NOT)

- **Not a production engine:**  
  No distributed vector DB, no serving layer, no SLAs, no integration into RAG stacks as a product. Any demo/CLI is strictly didactic.

- **Not a KV‑cache or latent‑block project:**  
  No binary KV‑cache, no binary latent transformer paths, no binary adapters inside LLMs, no MoE routing work—these were explicitly killed by Hostile review.

- **Not a fantasy speedup story:**  
  No “48×” or “10×” speedup claims; any throughput/QPS numbers, if reported at all, are **appendix‑level context**, not the headline.

- **Not a NeurIPS‑main‑track novelty play:**  
  Scientific novelty is modest: **systematic empirical characterization + clean tooling**. This is a **tech‑report / workshop / portfolio artifact**, not a breakthrough.

- **Not a Phase‑1 refactor:**  
  No changes to Phase‑1 code, tests, packing layout, or artifacts. Any GPU kernels or additional evaluation logic live in Phase‑2 only.

---

### Self‑Hostile Sanity Check

- **Assumption 1 — Binary can reach Recall@10 ≥ 0.70 on some benchmarks.**  
  - *Attack:* Prior literature and Hostile logs show binary often lags PQ significantly; on challenging datasets, even 512‑bit codes may struggle to cross 0.7 Recall@10. Anchoring expectations around 0.7 risks quiet disappointment.  
  - *Adjustment:* The plan already treats 0.7 as a **“hope” band, not a success requirement**, and YELLOW is explicitly acceptable (0.6–0.7). We will frame any failure to reach 0.7 as an expected outcome, not a failure of execution.

- **Assumption 2 — Geometry will show at least weak predictive power (E8/X2).**  
  - *Attack:* With 24–36 points, even real underlying structure can be drowned in noise; small‑N regressions are notorious for spurious correlations. Expecting R² > 0.3 is optimistic.  
  - *Adjustment:* The plan labels E8/X2 as **exploratory** and allows a fully negative result (F3/K3) without jeopardizing Phase‑2. Any positive signal will be treated as **hypothesis‑generating only**, never as proof.

- **Assumption 3 — Simple adapters (X3) can close ≥20–30% of the binary vs PQ gap.**  
  - *Attack:* Post‑hoc projections operate on embeddings that may already be fundamentally ill‑shaped for binarization; a small linear/MLP layer may have very limited leverage, especially if binarization is outside the training loop. Gains of 5–10 pp could be unrealistic.  
  - *Adjustment:* The target band for X3 is explicitly framed as **hopeful**, with clear negative thresholds (<2 pp, <10% gap). X3 is Tier‑3, capped in compute, and can be dropped entirely without harming Core.

- **Assumption 4 — PQ baselines will be “strong enough” out‑of‑the‑box.**  
  - *Attack:* Poor PQ configurations or too‑small training sets can significantly understate PQ’s real potential, making binary look better than it should—this is a subtle but serious bias risk.  
  - *Adjustment:* E3/E6 include sanity checks vs float and the plan encourages starting from **Faiss reference configs** with reasonable training subsamples. Any suspiciously weak PQ results must trigger re‑checks before being used in headline plots.

- **Assumption 5 — Time and compute estimates are realistic for a single student.**  
  - *Attack:* Real life adds friction: debugging, environment issues, I/O bottlenecks, and personal schedule slips. Even modest plans tend to slip by 1.5×.  
  - *Adjustment:* The plan includes **tiered execution** and strict kill‑switches (K1) to protect Core. Exploratory is explicitly optional, with X3 easiest to cut. Success is defined by Tier‑1 completion, not by hitting every stretch goal.

Overall, this plan is **deliberately conservative**: it expects that binary often loses to PQ, that geometry and adapters may provide only weak signals, and that many “hope” bands will not be met. The value of Phase‑2 is not in forcing positive results, but in producing **clean, reproducible evidence and clear Phase‑3 guidance** regardless of whether outcomes are flattering to binary retrieval.







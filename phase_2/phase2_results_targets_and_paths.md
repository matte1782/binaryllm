# BinaryLLM Phase‑2 — Results, Targets, and Future Paths

**Author:** Futures Architect  
**Date:** 2025-11-26  
**Revision:** Post-Hostile v1  
**Purpose:** Interpretation document—what we are looking for, what each outcome means, and where it leads.

---

## 1. Why Phase‑2 Exists

Phase‑2 is **not** an attempt to build a production search engine, claim record-breaking speedups, or prove that binary retrieval is universally superior. It is a **calibrated measurement campaign** that uses Phase‑1's frozen binary embedding pipeline to answer one practical question:

> **Under what conditions—if any—are 1‑bit binary embeddings a reasonable trade‑off against Product Quantization (PQ) and INT8 compression for modern LLM text embeddings?**

The study is deliberately modest in scope because:

1. **Prior work suggests binary often loses.** The goal is to *quantify* the gap, not to close it.
2. **A clean negative result is valuable.** Knowing exactly where binary fails prevents wasted effort later.
3. **Phase‑2 seeds Phase‑3/4.** A small Exploratory Track probes whether simple adapters or geometry shaping could improve binary quality—signals that guide the next phase, not products of this one.

Phase‑2 matters more for **learning and positioning** than for immediate product or publication. Its success is measured by the clarity and reproducibility of its evidence, not by flattering numbers.

---

## 2. Experiments and Their Purpose

### Core Track (E1–E8)

| ID | Name | What It Measures | Why We Run It |
|----|------|------------------|---------------|
| **E1** | Embedding Generation | GPU time and storage to cache float embeddings for 3 datasets × 3 models. | Confirms infrastructure works; provides raw material for everything else. |
| **E2** | Binarization Sweep | Memory per vector and Hamming self‑distance sanity at 64/128/256/512 bits. | Validates Phase‑1 pipeline stability and sets up binary codes for retrieval. |
| **E3** | PQ Index Building | Faiss PQ training time and sanity vs float (≥0.8× Recall) at 8–64 bytes. | Produces the **strong baseline** that binary must compete against. |
| **E4** | Float Brute‑Force | Ground‑truth top‑k neighbors under cosine similarity. | Anchors all quality metrics—without this, comparisons are meaningless. |
| **E5** | Binary Retrieval Eval | Recall@k, nDCG@10, Overlap@10 for binary codes vs float ground truth. | The **core output**: how much quality is lost at each bit‑length? |
| **E6** | PQ Retrieval Eval | Same metrics for PQ indices. | Quantifies the **binary vs PQ gap** at matched memory. |
| **E7** | Geometry Diagnostics | Anisotropy, outlier fraction, intrinsic dim, angular spread per (model, dataset). | Tests whether embedding shape correlates with binarization difficulty. |
| **E8** | Correlation Analysis | LOOCV R² and Spearman ρ from geometry → Recall@10. | Determines if geometry can **predict** binary quality (exploratory). |

---

#### E1 — Embedding Generation

| Band | Criterion |
|------|-----------|
| **Green** | All Tier‑1 + Tier‑2 embeddings cached in **≤20 A100‑h**, no major re‑runs. |
| **Yellow** | Tier‑1 complete, Tier‑2 partial; **≤30 A100‑h** total. |
| **Red** | Repeated failures or **>35 A100‑h** before coverage is adequate. |

---

#### E2 — Binarization Sweep

| Band | Criterion |
|------|-----------|
| **Green** | Codes stable, Hamming self‑distance = 0, distributions look healthy (no bit >60% prevalence). |
| **Yellow** | Skewed bit distributions (e.g., >60% of bits = 1) but Hamming distances still discriminative. |
| **Red** | Frequent collisions, non‑reproducible codes, or storage failures. |

---

#### E3 — PQ Index Building

| Band | Criterion |
|------|-----------|
| **Green** | PQ‑32/64 achieves **Recall@10 ≥ 0.80 of float** on ≥2/3 datasets. |
| **Yellow** | PQ‑64 achieves **0.70–0.80× float Recall@10** on ≥2/3 datasets; still yields clean recall vs memory curves. |
| **Red** | PQ Recall@10 < 0.60 at 64 bytes across the board—baseline is broken. |

---

#### E4 — Float Brute‑Force

| Band | Criterion |
|------|-----------|
| **Green** | Full top‑k ground truth for Tier‑1 in **≤50 CPU‑h**, sanity checks pass. |
| **Yellow** | Full top‑k for 2 models × 3 datasets; BGE‑large limited to 1K queries; baselines trustworthy. |
| **Red** | Baselines too expensive or show inconsistencies—downstream metrics invalid. |

---

#### E5 — Binary Retrieval Evaluation

| Band | Criterion |
|------|-----------|
| **Green (Optimistic Hope)** | On **≥2 datasets / ≥2 models**, binary‑256/512 reaches **Recall@10 ≥ 0.65**; at iso‑memory, binary is **≥0.80× PQ** on ≥1 dataset. *(Note: 0.70 would be exceptional; 0.65 is the realistic upper bound.)* |
| **Yellow (Realistic Success)** | Recall@10 mostly **0.55–0.65**; iso‑memory ratio **0.60–0.80× PQ**. Still useful quantification. |
| **Red (F1/F2)** | Recall@10 at 512 bits **never exceeds 0.55** on any dataset (F1), **or** binary < 0.5× PQ at matched memory on **all** benchmarks (F2). |

---

#### E6 — PQ Retrieval Evaluation

| Band | Criterion |
|------|-----------|
| **Green** | PQ‑32/64 reaches **Recall@10 ≥ 0.80** on most Tier‑1 pairs. |
| **Yellow** | PQ strong on ≥2/3 datasets; gaps vs binary measurable. |
| **Red** | PQ itself underperforms (Recall@10 < 0.70 at 64 bytes everywhere)—study less informative. |

---

#### E7 — Geometry Diagnostics

| Band | Criterion |
|------|-----------|
| **Green** | Anisotropy varies by **≥2×** across (model, dataset) pairs; at least one diagnostic correlates visually with Recall@10. |
| **Yellow** | Diagnostics show variation but no clear visual correlation with Recall@10. |
| **Red** | Metrics numerically unstable or show no usable variation. |

---

#### E8 — Correlation / Analysis

| Band | Criterion |
|------|-----------|
| **Green (Unlikely)** | LOOCV **R² > 0.3**, **ρ > 0.5**—geometry explains non‑trivial variance. *(Note: With N=24–36, this is optimistic; Yellow is the realistic positive outcome.)* |
| **Yellow (Realistic Positive)** | R² ∈ 0.1–0.3; geometry is a weak but usable prior. |
| **Red (F3)** | |ρ| < 0.3 and R² < 0.1—geometry provides no predictive power. |

---

### Exploratory Track (X1–X4)

| ID | Name | What It Measures | Why We Run It |
|----|------|------------------|---------------|
| **X1** | Binary Friendliness Index (BFI) | A scalar combining Recall@256/512, PQ ratio, and optional geometry penalty. | Gives Phase‑3 a **target metric** for embedding training. |
| **X2** | Geometry → Quality Predictors | LOOCV R²/ρ from geometry features to Recall@10. | Tests whether geometry‑based regularization is worth pursuing. |
| **X3** | Light Adaptation | Recall@10 improvement from a simple projection on cached embeddings. | **Feasibility signal** for Arc A: can adapters help without encoder retraining? |
| **X4** | Design Guidelines Memo | Synthesis of X1–X3 into Phase‑3 spec. | Converts numbers into **actionable recommendations**. |

---

#### X1 — BFI Definition

| Band | Criterion |
|------|-----------|
| **Green** | BFI ranking matches Recall@10 ranking on **≥5/6 Tier‑1 pairs** (Kendall τ ≥ 0.8). |
| **Yellow** | BFI ranking matches Recall@10 on **4/6 pairs** (Kendall τ ∈ 0.5–0.8); BFI is monotonic but not highly discriminative. |
| **Red** | BFI ranking contradicts Recall@10 on ≥3/6 pairs (Kendall τ < 0.5)—treat as cosmetic only. |

---

#### X2 — Geometry Predictors

| Band | Criterion |
|------|-----------|
| **Green (Unlikely)** | LOOCV **R² > 0.3**, **ρ > 0.5**, interpretable coefficients. *(Note: With N=24–36, Yellow is the realistic positive outcome.)* |
| **Yellow (Realistic Positive)** | R² ∈ 0.1–0.3; weak but consistent trends. |
| **Red (K3)** | R² < 0.05, |ρ| < 0.2—no signal; document and stop. |

---

#### X3 — Light Adaptation

| Band | Criterion |
|------|-----------|
| **Green (Optimistic Hope)** | Adapter improves Recall@10 by **≥3 pp** at 256 bits, **or** closes **≥15%** of binary vs PQ gap. *(Note: ≥5 pp / ≥20% would be exceptional but is unlikely given training loss doesn't see binarization.)* |
| **Yellow (Realistic Positive)** | Improvement **2–3 pp** or **10–15%** gap closure—adapters have *some* leverage. |
| **Red (K2)** | Improvement **< 2 pp** and **< 10%** gap closure after first config—stop X3. |

---

#### X4 — Design Guidelines

| Band | Criterion |
|------|-----------|
| **Green** | Memo contains clear BFI thresholds, geometry heuristics (if X2 positive), adapter baseline, and prioritized Phase‑3 experiments. |
| **Yellow** | Memo mostly summarizes negative results but states them explicitly. |
| **Red** | Memo remains vague ("do more experiments") with no concrete numbers. |

---

## 3. Scenario Map — GREEN / YELLOW / RED

### GREEN — Arc A GO (Optimistic Best-Case)

**Pattern:**

- Core E5 shows Recall@10 ≥ 0.65 at 256/512 bits on ≥2 datasets and ≥0.80× PQ at iso‑memory on ≥1.
- Exploratory X2 shows R² > 0.3 (geometry matters).
- Exploratory X3 shows ≥3 pp improvement or ≥15% gap closure.

**Meaning:**

| Dimension | Implication |
|-----------|-------------|
| **Phase‑3** | Full Arc A program: train or adapt encoders with geometry regularization and Hamming‑space losses; use BFI as success criterion. |
| **Phase‑4 / Monetization** | **Low–Medium (speculative).** A binary‑friendly embedding service is a distant possibility, not validated by Phase‑2. Any product direction requires Phase‑3/4 success first. |
| **Risk** | **High.** Even with positive Phase‑2 signals, Phase‑3 encoder fine‑tuning is expensive (100+ A100‑h) and may fail. |

---

### YELLOW — Arc A MAYBE (Realistic Success)

**Pattern:**

- Core E5 shows Recall@10 mostly 0.55–0.65; binary is 0.60–0.80× PQ.
- Exploratory X2 shows weak signal (R² 0.1–0.3) or X3 shows marginal gains (2–3 pp).
- No catastrophic failures (F1/F2 not triggered).

**Meaning:**

| Dimension | Implication |
|-----------|-------------|
| **Phase‑3** | Small Arc A probe: try one encoder fine‑tuning experiment; if it fails, pivot to PQ‑focused or hybrid work. |
| **Phase‑4 / Monetization** | **Low.** Binary may only be viable under extreme memory constraints; product scope is narrow. |
| **Risk** | Higher. Phase‑3 may dead‑end; budget conservatively. |

---

### RED — Arc A NO

RED has two sub-cases:

#### RED-A — Clean Negative Result (Valuable)

**Pattern:**

- Core triggers F1 (Recall@10 at 512 bits never > 0.55) **or** F2 (binary < 0.5× PQ everywhere).
- Exploratory X2 and X3 both negative (K2/K3 triggered).
- **Results are clean and reproducible.**

**Meaning:**

| Dimension | Implication |
|-----------|-------------|
| **Phase‑3** | Pivot away from binary retrieval with confidence. Options: (a) pure PQ/OPQ optimization, (b) hybrid two‑stage indexes, (c) different research axis entirely. |
| **Phase‑4 / Monetization** | **Low.** Binary embeddings are not a viable product direction for this embedding family. |
| **Portfolio Value** | **High.** A well-documented negative result prevents wasting 6+ months chasing a dead end. |

#### RED-B — Execution Failure (Low Value)

**Pattern:**

- Results are noisy, incomplete, or poorly logged.
- Kill-switches triggered due to infrastructure issues, not clean negative signal.

**Meaning:**

| Dimension | Implication |
|-----------|-------------|
| **Phase‑3** | Unclear pivot; must re-evaluate whether to retry or abandon. |
| **Phase‑4 / Monetization** | **Low.** No actionable signal. |
| **Portfolio Value** | **Low.** Poorly-documented failure is a red flag. |

---

## 4. Concrete Future Paths

*All paths below are **hypotheses**, not commitments. They exist to give the student a map of plausible futures, not a contract. Phase‑2 does not validate any product or monetization claim.*

### If GREEN (Arc A GO)

1. **Quantization‑Aware Encoder Training (Hypothesis)**  
   Fine‑tune MiniLM or e5‑base with a loss that explicitly penalizes Hamming‑space neighbor mismatches (e.g., Hamming‑aware contrastive loss + anisotropy regularization).  
   *Target:* Improve BFI and Recall@10 beyond Phase‑2 baselines (specific thresholds TBD based on Phase‑2 results).

2. **Binary‑Friendly Embedding Service (Speculative Product Seed)**  
   Package the best Phase‑3 encoder as a drop‑in embedding API that outputs binary codes directly, targeting cost‑sensitive RAG or on‑device search.  
   *Monetization (speculative, not validated by Phase‑2):* Potential directions include consulting or specialized embedding APIs, but these require Phase‑3/4 success.

3. **Multi‑Branch Encoder (Hypothesis)**  
   Train an encoder with two heads: one optimized for float/PQ, one for binary. Users choose at inference time based on memory constraints.

---

### If YELLOW (Arc A MAYBE)

1. **Single Arc A Probe (Hypothesis)**  
   Run one tightly scoped encoder fine‑tuning experiment (≤20 A100‑h). If Recall@10 improves meaningfully (threshold TBD based on Phase‑2 baselines), escalate to GREEN path; otherwise, close Arc A.

2. **PQ/OPQ Optimization Study**  
   Shift focus to advanced PQ variants (OPQ, ScaNN) as the primary compression method; treat binary as a coarse first‑stage filter only.

3. **Hybrid Two‑Stage Index**  
   Use binary codes for fast candidate generation, then re‑rank with PQ or float. Benchmark end‑to‑end recall vs latency.

---

### If RED (Arc A NO)

1. **Document and Archive**  
   Write a clear post‑mortem explaining *why* binary failed (geometry too pathological, PQ too strong, etc.). This is a valuable negative result.

2. **Pivot to Different Research Axis**  
   Options: (a) learned index structures, (b) entirely different domain (e.g., efficient fine‑tuning, not retrieval). *(Note: KV-cache was killed by Hostile v2/v3 and is not a valid pivot.)*

3. **Compression Studio (Systems Focus)**  
   Abandon binary as a core method; instead, build a practitioner tool that helps users choose among PQ, OPQ, ScaNN, and scalar quantization based on their constraints.

---

## 5. When to Say "Good Path" vs "Not"

At the end of Phase‑2, read the results through this lens:

1. **Did Core produce clean, reproducible trade‑off curves?**  
   If yes, Phase‑2 is a success regardless of whether binary "won."

2. **Did binary reach the Yellow band (Recall@10 ≥ 0.55, ≥0.60× PQ) on at least one meaningful setting?**  
   If yes, Arc A is worth a small probe in Phase‑3.

3. **Did binary reach the Green band (Recall@10 ≥ 0.65, ≥0.80× PQ) on at least one meaningful setting?**  
   If yes, Arc A deserves a more serious Phase‑3 investment (but with high risk acknowledged).

4. **Did Exploratory X3 show that adapters can help (≥3 pp or ≥15% gap)?**  
   If yes, Phase‑3 can start with projection learning—lower risk, lower compute.

5. **Did Exploratory X2 show geometry matters (R² > 0.1)?**  
   If yes, Phase‑3 can use geometry regularization as a design lever.

**If none of the above hold**, the honest conclusion is:

> "Binary retrieval with current LLM embeddings is not competitive with PQ under realistic memory budgets. Arc A is unlikely to succeed without fundamental changes to encoder architecture or training, which exceed our compute constraints. We recommend pivoting."

**A RED outcome is a success of the measurement campaign**—it prevents wasting 6+ months (and potentially hundreds of A100‑hours) chasing a direction that was never going to work. The value of Phase‑2 is in answering the question, not in forcing a particular answer.

---

## Self‑Hostile Check for this Futures Map

### Attack 1: Are the Green bands too optimistic?

- **Recall@10 ≥ 0.65 at 256 bits** is still ambitious. Prior literature and Hostile logs suggest binary often lags PQ by 15–25 pp. Reaching 0.65 may require favorable datasets or models.
- **Mitigation:** The plan now explicitly labels Green as "Optimistic Hope" and Yellow as "Realistic Success."

### Attack 2: Is the X3 "≥3 pp or ≥15% gap" threshold realistic?

- A simple linear projection operating on embeddings that are already binarized through a random Gaussian projection may have very limited leverage. The training loss does not see the binarization step.
- **Mitigation:** The threshold has been lowered from ≥5 pp to ≥3 pp. The kill‑switch (K2) triggers at <2 pp after one config, so we don't burn compute chasing nothing.

### Attack 3: Is "Arc A GO" language too hype‑y?

- Calling it "GO" could inflate expectations. Even in the Green scenario, Arc A is still a **research direction**, not a guaranteed product.
- **Adjustment:** GREEN is now labeled "Optimistic Best-Case" with "High" risk and "Low–Medium (speculative)" monetization.

### Attack 4: Could PQ baselines be misconfigured, making binary look better than it should?

- Yes. Poor PQ training or too‑small training sets could understate PQ's real potential.
- **Mitigation:** E3/E6 include sanity checks vs float; any suspiciously weak PQ results must trigger re‑checks before being used in headline plots.

### Attack 5: Is the "monetization potential" section speculative?

- Yes. No Phase‑2 experiment directly validates a product. Monetization claims are **speculation** based on assuming Phase‑3/4 success.
- **Acknowledgment:** All monetization language now explicitly states "speculative, not validated by Phase-2."

### Attack 6: Are X1 variance thresholds statistically valid?

- No. With only 6 Tier-1 pairs, variance thresholds are noise.
- **Fix:** X1 bands now use ranking-based criterion (Kendall τ) instead of variance.

### Attack 7: Is KV-cache a valid pivot option?

- No. KV-cache was killed by Hostile v2/v3.
- **Fix:** KV-cache removed from RED pivot options.

---

## Post-Hostile Revision Note

**Date:** 2025-11-26  
**Revision:** Post-Hostile v1

**Changes based on hostile review:**

1. **E5 Green band:** Lowered from Recall@10 ≥ 0.70 to ≥ 0.65; relabeled as "Optimistic Hope" with Yellow as "Realistic Success."
2. **X1 bands:** Replaced variance thresholds with ranking-based criterion (Kendall τ ≥ 0.8).
3. **X3 Green band:** Lowered from ≥5 pp / ≥20–30% gap to ≥3 pp / ≥15% gap.
4. **E8/X2 Green bands:** Added explicit caveat that Green is unlikely; Yellow is the realistic positive outcome.
5. **GREEN scenario:** Relabeled as "Optimistic Best-Case"; monetization downgraded from "Medium–High" to "Low–Medium (speculative)"; risk upgraded from "Moderate" to "High."
6. **RED scenario:** Split into RED-A (clean negative result, valuable) and RED-B (execution failure, low value).
7. **GREEN Path 1:** Removed fantasy numbers (BFI ≥ 0.75, Recall@10 ≥ 0.80); replaced with "targets TBD based on Phase-2 results."
8. **GREEN Path 2:** Added "speculative, not validated by Phase-2" to monetization language.
9. **YELLOW Path 1:** Removed "≥10 pp" threshold; replaced with "meaningful improvement (TBD)."
10. **RED Path 2:** Removed KV-cache from pivot options (killed by Hostile v2/v3).
11. **E2/E3/E4/E7 Yellow bands:** Added specific criteria instead of vague "minor quirks" or "slightly weak."

This document is now **post-hostile-approved** and consistent with Hostile Iteration 4 constraints.

---

*End of Phase‑2 Results, Targets, and Future Paths*

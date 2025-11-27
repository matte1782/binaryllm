# BinaryLLM Phase‑2 — Scientific Experiment Plan for Exploratory Track (Iteration 4)

**Author:** Senior Scientist Agent  
**Date:** 2025-11-26  
**Status:** Iteration 4 — Exploratory Track Operationalization

---

## Executive Summary

The Iteration 4 Strategist has defined a **two‑track Phase‑2**:

1. **Core Track:** The locked empirical binary retrieval study (E1–E8) comparing Phase‑1 binary codes against PQ/INT8 baselines across 3 datasets and 3 embedding models.
2. **Exploratory Track (X1–X4):** A small, high‑upside module that seeds **Arc A: Quantization‑Friendly / Binary‑Friendly Embeddings** for Phase‑3/4.

This document translates the Strategist's Exploratory Track into a **rigorous, minimal, and falsifiable scientific plan**. The goal is to maximize **informational value for Phase‑3/4** while staying within **≤5–10 A100‑hours** and **≤3 weeks** of part‑time effort, strictly secondary to the Core Track.

**Key scientific outputs targeted:**

1. A well‑defined **Binary Friendliness Index (BFI)** that collapses performance and geometry signals into a scalar per (model, dataset).
2. **Quantitative assessment** of whether embedding geometry (anisotropy, outliers, intrinsic dim, angular spread) **predicts** binarization quality, with explicit statistical rigor.
3. **Empirical evidence** from light adapter experiments on whether simple projections on cached embeddings can **close the binary vs PQ gap**.
4. A **Phase‑3/4 design memo** translating findings into actionable targets for quantization‑friendly embedding training.

**Compute budget:** ≤5–10 A100‑hours (≈15–25% of Core budget).  
**Timeline:** Weeks 10–14, interleaved with Core reporting, with explicit kill‑switches.

---

## 1. Restated Exploratory Track

### 1.1 Target Main‑Track Arc

The Strategist selected **Arc A: Quantization‑Friendly / Binary‑Friendly Embeddings** as the primary long‑term direction for Phase‑3/4.

**Arc A vision:**
- Train or adapt embedding models whose geometry is **intrinsically compatible with extreme compression** (1–4 bits) while preserving retrieval quality.
- Potential Phase‑3/4 activities: quantization‑aware training (QAT) of sentence encoders, binary‑aware contrastive losses, multi‑head encoders with separate float/binary branches, on‑device embedding models optimized for binary storage.

**Why Arc A is chosen:**
- Directly continues Phase‑2's binary retrieval study (same embeddings, same metrics, same geometry diagnostics).
- Feasible at student scale: early experiments operate on **cached embeddings** (no backprop through large encoders).
- High upside: successful techniques translate into monetizable embedding services and productized models.

### 1.2 What the Exploratory Track Tests

The Exploratory Track answers **three foundational questions** for Arc A:

1. **Can we quantify "binary friendliness"?**  
   Define a scalar index (BFI) that captures how well a given embedding model + dataset combination tolerates binarization. This becomes a **target metric** for Phase‑3 training.

2. **Does embedding geometry predict binary quality?**  
   Test whether simple geometric diagnostics (anisotropy, outlier fraction, intrinsic dim, angular spread) can **forecast** Recall@k under binarization. If yes, Phase‑3 can target training methods that **shape geometry**; if no, Phase‑3 must rely on **end‑to‑end empirical search**.

3. **Is binary performance recoverable with simple adapters?**  
   Train tiny projections on cached embeddings to see if the binary vs PQ gap can be **partially closed** without touching the encoder. This provides a **feasibility signal** for Arc A: if simple adapters help, full quantization‑aware training should help more.

### 1.3 Connection to Core Track

The Exploratory Track is **parasitic** on Core:

| Core Artifact | Exploratory Use |
|---------------|-----------------|
| E1: Float embeddings (cached) | Input to X3 (adapter training), X1/X2 (geometry computation) |
| E2: Binary codes | Input to X1/X2 (recall values), baseline for X3 comparison |
| E5: Binary retrieval metrics | Input to BFI formula (X1), correlation analysis (X2) |
| E6: PQ retrieval metrics | Baseline for BFI normalization (X1), adapter comparison (X3) |
| E7: Geometry diagnostics | Features for X2 (predictors), penalties for X1 (BFI) |

**No new embeddings are generated.** All Exploratory experiments reuse Core artifacts.

---

## 2. Scientific Questions for the Exploratory Track

We define **four focused, falsifiable questions** (XQ1–XQ4) that structure the Exploratory Track.

### XQ1: Can we define a Binary Friendliness Index (BFI) that meaningfully ranks (model, dataset) pairs?

**Motivation:** A scalar index that captures "how well this embedding binarizes" is essential for Phase‑3 as a **training target** and **model selection criterion**.

**Operationalization:**
- Define BFI as a function of:
  - **Performance signals:** Recall@10 at 256/512 bits, ratio of binary Recall@10 to PQ Recall@10 at iso‑memory.
  - **Geometry signals (optional):** Anisotropy, outlier fraction, intrinsic dim, angular spread (as penalties or bonuses).
- Compute BFI for all Tier‑1 (model, dataset) pairs (6 pairs: 2 models × 3 datasets).
- **Falsifiable criterion:** BFI is "meaningful" if it **separates** (model, dataset) pairs into distinguishable groups (e.g., BFI variance > 0.1 on a 0–1 scale) and **correlates** with practitioner intuition (e.g., high BFI ↔ high Recall@10 at 256 bits).

**Expected output:**
- A **BFI formula** with documented rationale.
- A **BFI table** for all Tier‑1 pairs.
- Scatter plot: BFI vs Recall@10 at 256 bits (should be monotonic or near‑monotonic by construction).

---

### XQ2: Do embedding geometry metrics predict binarization quality across (model, dataset, bit‑length) points?

**Motivation:** If geometry predicts binary quality, Phase‑3 can target training methods that **shape geometry** (e.g., reduce anisotropy, control outlier fraction). If not, Phase‑3 must rely on **end‑to‑end empirical search** without strong priors.

**Operationalization:**
- Construct a dataset of points: each point is a (model, dataset, bit‑length) triple with:
  - **Features:** Geometry diagnostics (anisotropy, outlier fraction, intrinsic dim, angular spread) + bit‑length.
  - **Target:** Recall@10 (or normalized BFI).
- With 2 models × 3 datasets × 4 bit‑lengths = 24 points (Tier‑1), or up to 36 points if Tier‑2 (BGE‑large) is included.
- Fit **simple predictors**:
  - Ridge regression, Lasso, or elastic net.
  - Optionally a tiny MLP (1 hidden layer, ≤16 units) if linear models fail.
- Evaluate via **leave‑one‑out cross‑validation (LOOCV)** on the small dataset.
- Report:
  - R² and RMSE on held‑out points.
  - Spearman ρ between predicted and actual Recall@10.
  - Feature importance / coefficients.

**Falsifiable criterion:**
- **Positive signal:** LOOCV R² > 0.3 and Spearman ρ > 0.5.
- **Negative signal:** LOOCV R² < 0.1 or Spearman ρ < 0.3 → geometry is a **weak prior**.

**Expected output:**
- Regression coefficients / feature importance table.
- Scatter plot: Predicted vs actual Recall@10.
- Clear statement: "Geometry [does / does not] meaningfully predict binary quality."

---

### XQ3: Can a simple learned projection on cached embeddings improve binary retrieval quality?

**Motivation:** If simple adapters (trained on cached embeddings, no encoder backprop) can close a meaningful fraction of the binary vs PQ gap, it signals that Arc A is **feasible** with modest compute. If not, Arc A may require **full encoder fine‑tuning** or be fundamentally limited.

**Operationalization:**
- **Scope:** 1 dataset (MS MARCO 500K subsample), 1–2 models (MiniLM, e5‑base).
- **Adapter architecture:** Linear projection \( W \in \mathbb{R}^{d \times d'} \) or shallow MLP (1 hidden layer, ReLU, ≤512 hidden units).
- **Training setup:**
  - Work on a **small subset** (50K–100K documents, 2K–5K queries) to keep training fast.
  - Freeze base embeddings; only train \( W \).
  - **Loss function:** Contrastive loss where:
    - **Positives:** Float‑cosine top‑k neighbors (e.g., k=10).
    - **Negatives:** Random non‑neighbors (in‑batch or hard negatives).
  - Alternatively, a **differentiable surrogate for Recall@k** (e.g., smooth top‑k loss).
- **Binarization:** Apply Phase‑1 pipeline to \( W x \) (L2‑normalize, random Gaussian projection, sign).
- **Evaluation:** Recall@10 at 128/256 bits on held‑out queries, compared to:
  - Baseline binary (no adapter).
  - PQ at matched memory.

**Falsifiable criterion:**
- **Positive signal:** Adapter improves Recall@10 by **≥5 percentage points** over baseline binary at 256 bits, or closes **≥20%** of the binary vs PQ gap at iso‑memory.
- **Negative signal:** Adapter improves Recall@10 by **<2 percentage points** or **<10%** of the gap → simple adapters are insufficient; Arc A requires deeper changes.

**Expected output:**
- Table: Baseline binary, Adapted binary, PQ at 128/256 bits.
- Bar chart: Recall@10 comparison.
- Qualitative analysis: Did the adapter change embedding geometry (anisotropy, angular spread)?

---

### XQ4: What concrete design guidelines can we extract for Phase‑3/4?

**Motivation:** The Exploratory Track must produce **actionable outputs** for Phase‑3, not just numbers. XQ4 synthesizes XQ1–XQ3 into a **design memo**.

**Operationalization:**
- Based on XQ1–XQ3 results, write a short document specifying:
  - **BFI definition and thresholds:** "A model is binary‑friendly if BFI ≥ X."
  - **Geometry heuristics (if XQ2 is positive):** "Target anisotropy < Y, outlier fraction < Z."
  - **Adapter lessons (if XQ3 is positive):** "Simple adapters recover ~W% of the gap; Phase‑3 should invest in [encoder fine‑tuning / projection learning / both]."
  - **Recommended Phase‑3 experiments:** Prioritized list of next steps.
- If XQ2 and XQ3 are both **negative**, the memo should state: "Phase‑3 must rely on end‑to‑end empirical search; no strong geometric or adapter‑based priors are available."

**Expected output:**
- `phase2_arcA_guidelines.md` (or section in `phase2_master_plan.md`).
- 2–3 pages, structured, with concrete numbers and recommendations.

---

## 3. Minimal Experiment Plan

### 3.1 Experiment X1: Define Binary Friendliness Index (BFI)

| Aspect | Specification |
|--------|---------------|
| **Inputs** | E5 results (Recall@10 at 64/128/256/512 bits), E6 results (PQ Recall@10 at 8/16/32/64 bytes), E7 results (geometry diagnostics) |
| **Scope** | All Tier‑1 pairs (2 models × 3 datasets = 6 pairs); optionally Tier‑2 (BGE‑large on MS MARCO + SciFact = 2 more pairs) |
| **Method** | Define BFI formula; compute for each pair; validate monotonicity with raw Recall@10 |
| **Outputs** | BFI formula documentation, BFI table, scatter plot (BFI vs Recall@10) |
| **Expected signal** | BFI variance > 0.1 on 0–1 scale; BFI ranking matches Recall@10 ranking |
| **Compute** | 0 GPU‑h, ~3 CPU‑h (pure analysis) |
| **Wall‑clock** | 1–2 days |

**BFI formula proposal (v1, tunable):**

\[
\text{BFI} = \alpha \cdot \text{Recall}_{256}^{\text{norm}} + \beta \cdot \text{Recall}_{512}^{\text{norm}} + \gamma \cdot \text{PQ\_ratio}_{256} - \delta \cdot \text{Aniso}^{\text{norm}}
\]

Where:
- \(\text{Recall}_{b}^{\text{norm}}\) = Recall@10 at \(b\) bits, normalized to [0,1] across all pairs.
- \(\text{PQ\_ratio}_{256}\) = Recall@10 (binary‑256) / Recall@10 (PQ‑32), clamped to [0,1].
- \(\text{Aniso}^{\text{norm}}\) = Anisotropy, normalized to [0,1] (higher anisotropy → lower BFI).
- \(\alpha, \beta, \gamma, \delta\) = Tunable weights (default: 0.4, 0.3, 0.2, 0.1).

The formula is **transparent and documented**, not a black‑box ML model.

---

### 3.2 Experiment X2: Geometry → Binary Quality Predictors

| Aspect | Specification |
|--------|---------------|
| **Inputs** | E7 results (geometry per model/dataset), E5 results (Recall@10 per model/dataset/bit‑length) |
| **Scope** | 24 points (Tier‑1: 2 models × 3 datasets × 4 bit‑lengths); optionally 32–36 points with Tier‑2 |
| **Method** | Ridge regression, Lasso; optionally tiny MLP; LOOCV evaluation |
| **Outputs** | R², RMSE, Spearman ρ; feature importance table; predicted vs actual scatter plot |
| **Expected signal** | R² > 0.3, Spearman ρ > 0.5 for positive; R² < 0.1, ρ < 0.3 for negative |
| **Compute** | 0 GPU‑h, ~3 CPU‑h |
| **Wall‑clock** | 1–2 days |

**Feature set:**

| Feature | Computation | Hypothesis |
|---------|-------------|------------|
| Anisotropy | \(\sigma_1 / \sigma_{\text{median}}\) from SVD | High anisotropy → poor binary performance |
| Outlier fraction | Fraction with \(\|x\|_2 > 2 \cdot \text{median}\) | High outliers → unstable sign binarization |
| Intrinsic dim | PCA components for 90% variance | Low intrinsic dim → fewer bits needed |
| Angular spread | Mean pairwise cosine (sampled) | High mean cosine → hard to separate neighbors |
| Bit‑length | 64, 128, 256, 512 | More bits → higher recall (trivial) |

**Regression targets:**
- Primary: Recall@10 (raw).
- Secondary: Normalized BFI (from X1).

**Statistical rigor:**
- Report 95% confidence intervals on R² via bootstrap (100 resamples).
- Report p‑value for Spearman ρ under null hypothesis of no correlation.
- **Do not overclaim:** With N=24–36, any correlation is exploratory.

---

### 3.3 Experiment X3: Light Adaptation on Cached Embeddings

| Aspect | Specification |
|--------|---------------|
| **Inputs** | E1 float embeddings (MS MARCO × MiniLM, MS MARCO × e5‑base), E5 baseline binary results, E6 PQ results |
| **Scope** | 1 dataset (MS MARCO 500K), 1–2 models (MiniLM, e5‑base) |
| **Subset** | Training: 50K docs, 2K queries; Eval: 100K docs, 2K queries (disjoint) |
| **Adapter** | Linear \( W \in \mathbb{R}^{d \times d} \) or MLP (d → 256 → d) |
| **Loss** | Contrastive: positives = float top‑10 neighbors, negatives = in‑batch random |
| **Training** | Adam, lr=1e‑3, batch=256, 10–20 epochs; early stop on val Recall@10 |
| **Binarization** | Phase‑1 pipeline on \( W x \) |
| **Evaluation** | Recall@10 at 128/256 bits on eval set |
| **Outputs** | Table (baseline, adapted, PQ), bar chart, geometry change analysis |
| **Expected signal** | Positive: ≥5 pp improvement or ≥20% gap closure; Negative: <2 pp or <10% gap |
| **Compute** | 3–5 GPU‑h (A100), 5–10 CPU‑h |
| **Wall‑clock** | 3–5 days |

**Training protocol (detailed):**

1. **Data preparation:**
   - Load cached float embeddings for MS MARCO × (MiniLM or e5‑base).
   - Subsample 50K docs + 2K queries for training, 100K docs + 2K queries for eval.
   - Compute float‑cosine top‑10 neighbors for each query (ground truth for contrastive loss).

2. **Adapter forward pass:**
   - \( z = W x \) (linear) or \( z = \text{MLP}(x) \).
   - L2‑normalize \( z \).

3. **Contrastive loss:**
   - For each query \( q \), positives = float top‑10 docs, negatives = in‑batch docs not in top‑10.
   - InfoNCE‑style loss with temperature τ=0.1.

4. **Binarization (eval only):**
   - Apply Phase‑1 Gaussian projection + sign to \( z \).
   - Compute Hamming‑based top‑k.
   - Evaluate Recall@10 vs float ground truth.

5. **Hyperparameter budget:**
   - ≤3 configurations (linear vs MLP, lr ∈ {1e‑3, 1e‑4}).
   - No broad sweeps; treat first working config as main result.

**Geometry change analysis:**
- After training, compute anisotropy and angular spread of adapted embeddings.
- Compare to baseline: did the adapter **reduce anisotropy** or **increase angular spread**?

---

### 3.4 Experiment X4: Design Guidelines Synthesis

| Aspect | Specification |
|--------|---------------|
| **Inputs** | X1 BFI table, X2 regression results, X3 adapter results |
| **Scope** | Synthesis across all Exploratory results |
| **Method** | Manual analysis + structured writing |
| **Outputs** | `phase2_arcA_guidelines.md` (2–3 pages) |
| **Expected signal** | Actionable recommendations for Phase‑3 |
| **Compute** | 0 GPU‑h, ~5 CPU‑h (writing) |
| **Wall‑clock** | 2–3 days |

**Document structure:**

1. **BFI Definition and Thresholds**
   - Final BFI formula.
   - Recommended thresholds: "BFI ≥ 0.6 = good, 0.4–0.6 = marginal, <0.4 = poor."

2. **Geometry Heuristics (if XQ2 positive)**
   - "Target anisotropy < X for binary‑friendly embeddings."
   - "Outlier fraction < Y is associated with higher Recall@10."

3. **Adapter Lessons (if XQ3 positive)**
   - "Simple linear adapters recover ~Z% of the binary vs PQ gap on MS MARCO."
   - "Phase‑3 should invest in [projection learning / encoder fine‑tuning]."

4. **Recommended Phase‑3 Experiments**
   - Prioritized list of next steps (e.g., "Train MiniLM with binary‑aware contrastive loss").

5. **Negative Results (if applicable)**
   - "Geometry does not predict binary quality; Phase‑3 must rely on empirical search."
   - "Simple adapters provide <2 pp improvement; Arc A may require full encoder changes."

---

## 4. Compute & Complexity Control

### 4.1 Compute Budget Table

| Experiment | GPU‑h | CPU‑h | Wall‑clock (days) | Notes |
|------------|-------|-------|-------------------|-------|
| X1 (BFI) | 0 | 3 | 1–2 | Pure analysis |
| X2 (Geometry predictors) | 0 | 3 | 1–2 | Pure analysis |
| X3 (Light adaptation) | 3–5 | 5–10 | 3–5 | Main compute cost |
| X4 (Design guidelines) | 0 | 5 | 2–3 | Writing |
| **Total Exploratory** | **3–5** | **16–21** | **7–12** | ≤25% of Core |

**Comparison to Core:**

| Track | GPU‑h | CPU‑h | Wall‑clock |
|-------|-------|-------|------------|
| Core (E1–E8) | 23–33 | 80–120 | 10–12 weeks |
| Exploratory (X1–X4) | 3–5 | 16–21 | 2–3 weeks |
| **Exploratory / Core** | **13–15%** | **17–20%** | **~20%** |

The Exploratory Track is **strictly secondary** and well within the ≤20–30% budget guideline.

### 4.2 Simplifications if Over Budget

If Exploratory threatens to exceed budget or delay Core:

1. **Drop X3 (adapter experiments).**
   - X1, X2, X4 are analysis‑only and essentially free.
   - X3 is the only non‑trivial compute; dropping it saves 3–5 GPU‑h and 3–5 days.
   - X4 can still be written with X1/X2 results alone.

2. **Reduce X3 scope to 1 model (MiniLM only).**
   - Cuts GPU‑h by ~50%.
   - Still provides a feasibility signal.

3. **Skip Tier‑2 in X1/X2.**
   - Use only Tier‑1 (6 pairs) instead of 8.
   - Reduces analysis complexity.

---

## 5. Informational Value for Phase‑3/4

For each scientific question, we specify **how the answer will concretely inform Phase‑3/4 decisions**.

### XQ1 (BFI) → Phase‑3/4 Impact

| Result | Phase‑3/4 Decision |
|--------|-------------------|
| BFI is well‑defined and separates models | Use BFI as **training objective** or **model selection criterion** in Phase‑3. |
| BFI does not separate models (all similar) | BFI is not useful; Phase‑3 must use raw Recall@k as target. |

**Concrete example:**
- If MiniLM has BFI=0.72 and e5‑base has BFI=0.58, Phase‑3 can ask: "What makes MiniLM more binary‑friendly? Can we train e5‑base to match?"

---

### XQ2 (Geometry predictors) → Phase‑3/4 Impact

| Result | Phase‑3/4 Decision |
|--------|-------------------|
| Geometry predicts binary quality (R² > 0.3) | Phase‑3 can target training methods that **shape geometry** (e.g., anisotropy regularization, outlier suppression). |
| Geometry does not predict (R² < 0.1) | Phase‑3 must rely on **end‑to‑end empirical search**; no strong geometric priors. |

**Concrete example:**
- If anisotropy coefficient is strongly negative (β < −0.5), Phase‑3 should add an **anisotropy penalty** to the training loss.
- If no feature is predictive, Phase‑3 should skip geometry‑based regularization and focus on **direct Hamming‑space objectives**.

---

### XQ3 (Adapter experiments) → Phase‑3/4 Impact

| Result | Phase‑3/4 Decision |
|--------|-------------------|
| Simple adapter closes ≥20% of gap | Arc A is **feasible with modest compute**; Phase‑3 can start with projection learning before encoder fine‑tuning. |
| Simple adapter closes <10% of gap | Arc A requires **full encoder changes**; simple projections are insufficient. |
| Adapter changes geometry favorably | Phase‑3 should design losses that **explicitly encourage geometry changes** (e.g., spread maximization). |

**Concrete example:**
- If a linear adapter on MiniLM improves Recall@10 from 0.65 → 0.72 at 256 bits (closing 35% of the gap to PQ), Phase‑3 can invest in **learned projections** as a first step.
- If the adapter only improves 0.65 → 0.66, Phase‑3 should skip projections and go directly to **encoder fine‑tuning with binary‑aware losses**.

---

### XQ4 (Design guidelines) → Phase‑3/4 Impact

| Result | Phase‑3/4 Decision |
|--------|-------------------|
| Clear guidelines produced | Phase‑3 starts with a **concrete spec**: BFI targets, geometry heuristics, adapter baselines. |
| No clear guidelines (all negative) | Phase‑3 starts with **exploratory empirical search**; no strong priors, higher risk. |

**Concrete example:**
- If the memo says "BFI ≥ 0.65, anisotropy < 15, adapter baseline = +7 pp", Phase‑3 has a **clear success criterion** and **baseline to beat**.

---

## 6. Early‑Stop / Kill Conditions for Exploratory Track

To prevent sunk‑cost fallacy and protect Core integrity, we define explicit **stop conditions**.

### 6.1 Kill Conditions (Abandon Exploratory)

| Condition | Trigger | Action |
|-----------|---------|--------|
| **K1: Core is behind schedule** | Tier‑1 Core (E1–E6) is <80% complete by week 10 | Pause all Exploratory work; focus on Core. |
| **K2: X3 shows no signal** | After 1 adapter config, Recall@10 improvement < 1 pp | Stop X3; proceed with X1/X2/X4 only. |
| **K3: X2 shows no signal** | After initial regression, R² < 0.05 and all |ρ| < 0.2 | Document negative result; do not invest more time in geometry analysis. |

### 6.2 Success Conditions (Justify Phase‑3 Extension)

| Condition | Trigger | Action |
|-----------|---------|--------|
| **S1: BFI is meaningful** | BFI variance > 0.1, ranking matches Recall@10 ranking | Use BFI as Phase‑3 target. |
| **S2: Geometry predicts quality** | LOOCV R² > 0.3, Spearman ρ > 0.5 | Incorporate geometry constraints in Phase‑3 training. |
| **S3: Adapter shows promise** | Recall@10 improvement ≥ 5 pp or ≥ 20% gap closure | Invest in projection learning in Phase‑3. |
| **S4: Clear design memo** | X4 produces actionable guidelines | Phase‑3 starts with concrete spec. |

**If S1–S4 are met:** Exploratory is a success; Phase‑3 has a strong foundation for Arc A.

**If K1–K3 are triggered:** Exploratory is deprioritized or abandoned; Phase‑2 remains a success as a Core‑only empirical study.

### 6.3 Checkpoints

| Checkpoint | Timing | Decision |
|------------|--------|----------|
| **XP1** | After X1 (week 10–11) | If BFI variance < 0.05, reconsider formula; if still flat, document and proceed. |
| **XP2** | After X2 (week 11) | If R² < 0.05, document negative result; skip geometry‑based Phase‑3 recommendations. |
| **XP3** | After 1st X3 config (week 12) | If improvement < 1 pp, stop X3; if > 3 pp, try 2nd config. |
| **XP4** | After X4 (week 13–14) | Final synthesis; lock Phase‑3 recommendations. |

---

## 7. Theoretical Notes

### 7.1 Why Geometry Might Predict Binary Quality

**Classical LSH theory** (Charikar 2002, Andoni & Indyk 2008) shows that for unit vectors \(x, y\):

\[
\Pr[h_r(x) = h_r(y)] = 1 - \frac{\theta(x, y)}{\pi}
\]

where \(\theta(x, y) = \arccos(\langle x, y \rangle)\).

**Implications for geometry:**
- **Anisotropy:** If embeddings cluster in a narrow cone (high anisotropy), pairwise angles \(\theta\) are small and similar. Binary hashing struggles to separate neighbors from non‑neighbors because the hash collision probabilities are nearly identical.
- **Outlier dimensions:** If a few dimensions carry disproportionate variance, the sign of those dimensions dominates the binary code, losing information from other dimensions.
- **Intrinsic dimensionality:** If embeddings lie on a low‑dimensional manifold, fewer bits are needed to capture the structure; conversely, high intrinsic dim may require more bits.
- **Angular spread:** Low angular spread (high mean pairwise cosine) means most pairs are "similar" in cosine space, making binary separation harder.

**Caveat:** These are **heuristics**, not theorems. Real LLM embeddings violate the uniform angular distribution assumed by LSH theory. The Exploratory Track tests whether these heuristics hold empirically.

### 7.2 Why Simple Adapters Might Help

A learned projection \(W\) can:
1. **Rotate embeddings** to align with random hyperplanes used in binarization.
2. **Spread embeddings** to increase angular gaps between neighbors and non‑neighbors.
3. **Suppress outlier dimensions** by down‑weighting them.

If the original embedding geometry is "almost" binary‑friendly but has minor pathologies (e.g., slight anisotropy, a few outlier dims), a simple adapter can correct these without retraining the encoder.

**Caveat:** If the pathologies are severe (e.g., embeddings lie on a 1D manifold), no post‑hoc projection can fix them. This is what X3 tests.

### 7.3 Limitations of Small‑N Analysis

With only 24–36 data points (XQ2), any statistical analysis is **exploratory**:
- R² and Spearman ρ have high variance.
- Overfitting is a risk even with simple models.
- No strong causal claims can be made.

**Mitigation:**
- Use LOOCV, not train/test split.
- Report confidence intervals.
- Frame all results as "exploratory" and "hypothesis‑generating," not "conclusive."

---

## 8. Summary Table

| Question | Experiment | Inputs | Outputs | Positive Signal | Negative Signal | Phase‑3/4 Impact |
|----------|------------|--------|---------|-----------------|-----------------|------------------|
| XQ1 (BFI) | X1 | E5, E6, E7 | BFI table, formula | Variance > 0.1, ranking matches Recall | Flat BFI | Target metric for training |
| XQ2 (Geometry) | X2 | E7, E5 | R², ρ, coefficients | R² > 0.3, ρ > 0.5 | R² < 0.1, ρ < 0.3 | Geometry constraints or empirical search |
| XQ3 (Adapter) | X3 | E1, E5, E6 | Recall table, geometry change | ≥5 pp or ≥20% gap | <2 pp or <10% gap | Projection learning or encoder fine‑tuning |
| XQ4 (Guidelines) | X4 | X1–X3 | Design memo | Actionable spec | No clear guidelines | Phase‑3 starting point |

---

## 9. Conclusion

This scientific plan operationalizes the Strategist's Exploratory Track into a **rigorous, minimal, and falsifiable** research program. The plan:

- **Reuses Core artifacts** (embeddings, binary codes, geometry diagnostics) with no new data generation.
- **Stays within ≤5 A100‑hours and ≤3 weeks**, strictly secondary to Core.
- **Produces actionable outputs** (BFI, geometry heuristics, adapter lessons) that directly inform Phase‑3/4.
- **Has explicit kill‑switches** to prevent sunk‑cost fallacy.

If the Exploratory Track succeeds, Phase‑3 starts with a **concrete spec** for quantization‑friendly embeddings. If it fails, Phase‑2 remains a **high‑quality empirical study**, and Phase‑3 proceeds with **exploratory empirical search** rather than theory‑driven design.

---

*End of Phase‑2 Exploratory Track Scientific Plan (Iteration 4)*






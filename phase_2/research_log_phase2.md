## Phase‑2 Research Log

### 2025-11-26T00:00:00Z — `/research-strategist //research_phase2_freeze_and_objectives_v2`

- **Context:** Post‑Iteration‑4 Hostile approval; Strategist+Scientist joint run to hard‑freeze Phase‑2 experiments and objectives.
- **Decisions:** Locked **Core Track E1–E8** (binary vs PQ empirical study) and **Exploratory Track X1–X4** (BFI + light adaptation) with explicit Tier‑1/2/3 priorities and compute/time caps (≤65 A100‑hours, target 30–40).
- **Clarifications:** Defined per‑experiment questions, metrics, and realistic target bands (hope / still useful / negative) consistent with Hostile realism; killed any residual production‑engine or fantasy‑speedup narratives.
- **Kill‑switches:** Reaffirmed and centralized Core failure criteria (F1–F3) and Exploratory kill conditions (K1–K3), including hard guardrails on X3 (≤5 A100‑hours, ≤3 configs).
- **Future path:** Wrote `phase2_final_plan.md` as the canonical Phase‑2 contract and shaped Exploratory outcomes to directly inform Phase‑3/4 Arc A (quantization‑friendly embeddings) without expanding scope.

---

### 2025-11-26T01:00:00Z — `/research-architect //research_phase2_results_targets_and_paths_v1`

- **Context:** Futures Architect run to produce an interpretation document mapping Phase‑2 experiments to concrete outcomes and future directions.
- **Artifact:** Created `phase_2/phase2_results_targets_and_paths.md`.
- **Contents:**
  1. Short explanation of why Phase‑2 exists (measurement, not product).
  2. Per‑experiment (E1–E8, X1–X4) descriptions with Green/Yellow/Red target bands tied to specific metrics.
  3. Global scenario map (GREEN = Arc A GO, YELLOW = Arc A MAYBE, RED = Arc A NO) with implications for Phase‑3/4 and monetization.
  4. Concrete future paths per scenario (e.g., encoder training, PQ focus, pivot).
  5. Decision heuristic: how to read final results and decide "worth more investment" vs "close this line."
  6. Self‑Hostile sanity check attacking optimistic thresholds, adapter assumptions, and monetization speculation.
- **Usage:** This document is the student's interpretive guide—consult it after experiments complete to translate numbers into next‑step decisions.

---

### 2025-11-26T02:00:00Z — `/research-hostile //research_phase2_results_paths_hostile_v1`

- **Context:** Hostile review of `phase2_results_targets_and_paths.md` to stress-test bands, scenario mappings, and future-path claims.
- **Artifact:** Created `phase_2/phase2_results_targets_and_paths_review_hostile.md`.
- **Main concerns:**
  1. **E5/X3 Green bands are too optimistic**—Recall@10 ≥ 0.70 and ≥5 pp adapter gains are ambitious given Hostile v3/v4 warnings; recommend relabeling as "Optimistic Hope" or lowering thresholds.
  2. **X1 (BFI) variance thresholds are statistically meaningless** on a 6-point sample; recommend replacing with ranking-based criterion (Kendall τ).
  3. **GREEN scenario monetization is oversold**—"Medium–High" is speculative; Phase‑2 does not validate any product; recommend downgrade to "Low–Medium (speculative)."
  4. **Fantasy numbers in future paths** (BFI ≥ 0.75, Recall@10 ≥ 0.80, ≥10 pp improvement) have no Phase‑2 grounding; recommend removal or explicit "aspirational" label.
  5. **KV-cache listed as pivot option** contradicts Hostile v2/v3 kill; recommend removal.
- **Verdict:** APPROVE WITH MINOR CORRECTIONS—document is mostly honest; suggested patch summary provided.

---

### 2025-11-26T03:00:00Z — `/research-architect //research_phase2_results_paths_patch_v1`

- **Context:** Patch run to apply hostile review corrections to `phase2_results_targets_and_paths.md`.
- **Artifact:** Updated `phase_2/phase2_results_targets_and_paths.md` (Post-Hostile v1).
- **Key changes applied:**
  1. E5 Green band lowered from ≥0.70 to ≥0.65; relabeled "Optimistic Hope" with Yellow as "Realistic Success."
  2. X1 bands replaced variance thresholds with Kendall τ ≥ 0.8 (ranking match).
  3. X3 Green band lowered from ≥5 pp / ≥20–30% gap to ≥3 pp / ≥15% gap.
  4. GREEN scenario monetization downgraded to "Low–Medium (speculative)"; risk upgraded to "High."
  5. RED scenario split into RED-A (clean negative, valuable) and RED-B (execution failure, low value).
  6. Fantasy numbers removed from GREEN Path 1; KV-cache removed from RED pivot options.
  7. E2/E3/E4/E7 Yellow bands made specific instead of vague.
- **Status:** Futures map is now **post-hostile-approved** and consistent with Hostile Iteration 4.



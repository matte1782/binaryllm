# BinaryLLM — Phase 1 Progress Log (v2)

**Project:** `binary_llm`  
**Phase:** 1 — Binary Embedding & Similarity Engine v0.1  
**Scope:** H1 – Binary embeddings preserve retrieval neighborhoods at sufficient code length  
**Last update:** [METTI DATA]

---

## 1. High-level status

- ✅ **Research foundation**
  - `docs/binaryllm_report_v2.md` — validated theoretical report (math + literature + hypotheses).
  - `docs/binaryllm_paper_arxiv_v1.md` — arXiv-style draft, consistent with the report.

- ✅ **Architecture**
  - `docs/binaryllm_phase1_architecture_v1.md` — Phase 1 architecture **APPROVATA** dal validator:
    - Obiettivo chiaro: **Binary Embedding & Similarity Engine v0.1** (H1-focused).
    - Moduli definiti: `core`, `quantization`, `eval`, `utils`, `experiments`, `variants`.
    - Task T1–T10 definiti con invarianti e test plan.

- ✅ **Infrastructure / Agents**
  - `/researcher`, `/validator`, `/formatter`, `/architect_binaryllm`, `/engineer_binaryllm`, `/tester_binaryllm`, `/hostile_binaryllm` operano con:
    - Divisione ruoli (research / arch / code / test / hostile review).
    - Nessun coding senza architettura validata.
    - Nessuna modifica senza test e review ostile.

---

## 2. Task status (Phase 1 – Engine v0.1)

| Task | Descrizione breve                                      | Stato | Note chiave |
|------|--------------------------------------------------------|-------|-------------|
| T1   | Skeleton packages `src/` + `tests/`                    | ✅    | Directory + `__init__` minimi, smoke test import, review ostile PASS. |
| T2A  | `FloatEmbeddingBatch` + `BinaryCodeBatch`              | ✅    | Invarianti shape/dtype, normalized flag, convenzioni `{-1,+1}` / `{0,1}`, error messages allineati alle regex dei test. |
| T2B  | Dataset catalog + `EmbeddingDataset` / `QueryDataset`  | ✅    | `DatasetSpec` / `EncoderSpec` con registry, error-path testati, allineamento con H1 (normalizzazione obbligatoria dove richiesto). |
| T3   | Binarization + projection operators                    | ✅    | `binarize_sign`, random hyperplanes; test su Hamming–angolo; doppio audit ostile (standard + t3_audit) PASS. |
| T4   | Bit packing/unpacking `{0,1}` → `uint64`               | ⏳    | **Next step**. Bit layout va fissato ora per futuri kernel GPU. |
| T5   | Similarity metrics (cosine vs Hamming)                 | ⏳    | Da fare dopo T4. |
| T6   | Retrieval metrics (top-k, overlap, nDCG, Recall)       | ⏳    | Dipende da T4 + T5. |
| T7   | Classification eval (aux)                              | ⏳    | Esplicitamente secondaria rispetto a H1. |
| T8   | Config / seed / IO / logging utils                     | ⏳    | Necessari per runner e reproducibility completa. |
| T9   | `phase1_binary_embeddings` runner + config example     | ⏳    | Glue finale per esperimenti H1. |
| T10  | `binary_embedding_engine` façade                       | ⏳    | Interfaccia riusabile per fasi future. |

---

## 3. Cosa è stato consolidato finora

### 3.1 Embeddings & datasets

- **`FloatEmbeddingBatch`**
  - Garantisce:
    - `shape = (N, d)`, `N > 0`, `d > 0`.
    - Niente NaN/Inf.
    - Se `normalized=True` → ||x||₂ ≈ 1 con epsilon documentato.
  - Error messages:
    - Esplicitano il campo problematico + “embeddings” per diagnosi stabile.

- **`BinaryCodeBatch`**
  - Interno: `codes_pm1 ∈ {-1,+1}`.
  - Logico: `codes_01 ∈ {0,1}` con mapping `-1 → 0`, `+1 → 1`.
  - `code_bits` coerente con ultima dimensione.

- **Dataset catalog + dataset wrappers**
  - `DatasetSpec` + `EncoderSpec` → sorgente di verità per:
    - dimensioni embedding,
    - formato file,
    - normalizzazione richiesta,
    - supporto similarity/retrieval/classification.
  - `EmbeddingDataset` e `QueryDataset`:
    - validano richieste contro il catalog,
    - centralizzano la logica di controllo, riducendo “stringhe magiche”.

### 3.2 Binarizzazione & proiezioni

- Implementata la **binarizzazione** coerente con il report:
  - `sign` con convenzione definita,
  - proiezione random hyperplane compatibile con teoria Hamming–angolo.
- Test:
  - Validano:
    - convenzioni di segno,
    - shape invariants,
    - comportamento con input vuoti o non validi,
    - sanità statistica Hamming–angolo per grandi `code_bits`.

---

## 4. Prossimo step (Phase 1)

**Prossimo task disciplinato:**  
👉 **T4 — Bit packing/unpacking `{0,1}` → `uint64`**

Motivazioni:
- Fissa **ora** il layout di bit (word size, ordine, allineamento) per:
  - efficienza di memoria nei dataset grandi,
  - compatibilità futura con kernel CUDA (XNOR+popcount),
  - coerenza di Hamming distance a livello di storage.

Flusso per T4:
1. `/tester_binaryllm` definisce i test per `pack_codes` / `unpack_codes` (test-first).
2. `/engineer_binaryllm` implementa `src/quantization/packing.py` per soddisfare i test, senza scope creep.
3. `/hostile_binaryllm` fa review ostile del patch T4 (invarianti, performance, future GPU-compatibility).

Questo log v2 congela chiaramente lo stato **prima di T4**.

# BinaryLLM – Phase 1 – **T11 Progress Log**  
### Façade Extraction & Runner Refactor (BinaryEmbeddingEngine ↔ Runner)

**Date:** 23/11/2025  
**Status:** ✅ T11 COMPLETED — Hostile Reviewer: **PASS**

---

## 1. Obiettivo di T11

**Scopo di T11**

Ristrutturare la pipeline Phase 1 per rispettare pienamente l’architettura:

- Il **runner** (`src/experiments/runners/phase1_binary_embeddings.py`) deve diventare **ultra-sottile**:
  - carica la config,
  - setta il seed globale,
  - chiama *una volta* la façade,
  - logga il risultato.

- La **façade** (`src/variants/binary_embedding_engine.py`) diventa il **singolo punto di orchestrazione** per Phase 1:
  - normalizzazione embeddings,
  - proiezione randomica (T3),
  - binarizzazione (T3),
  - bit-packing (T4),
  - calcolo similarità (T5),
  - retrieval metrics (T6),
  - classification evaluation (T7),
  - integrazione con config/seed/logging (T8),
  - compatibilità con il runner (T9),
  - mantenendo invariati tutti i contratti T10.

**Vincolo assoluto:**  
👉 Nessun cambiamento di comportamento rispetto a T10 (metriche, shape, determinismo, schema risultati).

---

## 2. Stato iniziale prima di T11

Dopo T10:

- **BinaryEmbeddingEngine** era già presente e validato.
- Il **runner T9** conteneva ancora parte della logica di orchestrazione:
  - caricava embeddings,
  - creava proiezione,
  - binarizzava,
  - calcolava metriche direttamente con i moduli T3–T7.
- La façade veniva usata in modo limitato, non come **single entrypoint**.

Questo violava implicitamente la visione architetturale:

> “Façade = orchestratore unico. Runner = glue sottile.”

T11 nasce per **correggere questa asimmetria**.

---

## 3. Lavoro svolto in T11 (vista ad alto livello)

### 3.1. Refactor del Runner (T9 → T11)

**Prima**:  
Il runner:

- legava insieme:
  - dataset loading,
  - proiezione,
  - binarizzazione,
  - packing,
  - similarity,
  - retrieval,
  - classification,
- e gestiva direttamente il logging.

**Dopo T11**:

- Il runner fa soltanto:
  1. `config = load_config(path)`
  2. `set_global_seed(config.seed)`
  3. `engine = BinaryEmbeddingEngine(...)`
  4. `result = engine.run(...)`
  5. `write_log_entry(result, ...)`
- Nessuna chiamata diretta a:
  - `RandomProjection`,
  - `binarize_sign`,
  - `pack_codes`,
  - `cosine_similarity_matrix`,
  - `hamming_distance_matrix`,
  - `ndcg_at_k` / `recall_at_k`,
  - classification evaluator.

Il runner è ora **un guscio sottile e controllabile**, pronto per futuri runtime / CLIs.

---

### 3.2. Rafforzamento della Façade (BinaryEmbeddingEngine)

**Responsabilità consolidate in `BinaryEmbeddingEngine.run`:**

1. **Pre-processing**  
   - Normalizzazione L2 degli embeddings se richiesto.
   - Validazione shapes & dtypes.

2. **Proiezione & Binarizzazione**  
   - Usa `RandomProjection` (T3) con seed dell’esperimento.
   - Applica binarizzazione sign % \{-1,+1\}.
   - Converte in `{0,1}` e poi in packed (T4).

3. **Calcolo Similarità (T5)**  
   - `cosine_similarity_matrix`
   - `hamming_distance_matrix`
   - `neighbor_overlap_at_k` + Spearman

4. **Retrieval (T6)**  
   - `topk_neighbor_indices_cosine`
   - `topk_neighbor_indices_hamming`
   - `ndcg_at_k`, `recall_at_k`

5. **Classification (T7)**  
   - Delegato correttamente a `src/eval/classification.evaluate_classification`
   - Nessuna logica di training/F1 dentro la façade.

6. **Output Schema (Phase 1 façade-local)**  
   - `encoder_name`, `dataset_name`, `code_bits`, `projection_type`, `seed`, `normalize`
   - `metrics`:
     - `similarity`
     - `retrieval`
     - `classification` (o `None` se non richiesto / non supportato)
   - `binary_codes`:
     - `"pm1"`: shape (N, code_bits)
     - `"01"`: shape (N, code_bits)
     - `"packed"`: packing 64-bit conforme a T4.

---

### 3.3. Logging Hardening (Runner + T8)

Durante T11, è emerso un problema pratico:

- Il façade restituisce numpy arrays e strutture non JSON-serializzabili.
- Il logger deve:
  - salvare i risultati,
  - senza modificare l’oggetto in memoria.

**Soluzione implementata:**

- Aggiunta di uno **sanitizer di logging** nel runner:
  - crea una **deep copy sanitizzata** del payload (solo per logging),
  - converte `binary_codes` in liste native (o strutture JSON-compatibili),
  - lascia il risultato in memoria **intatto**.

Hostile reviewer ha confermato:

> “logging hardening is acceptable and does not mutate in-memory results.”

---

## 4. Hostile Reviewer – Iterazione Finale

Output finale (T11 hotfix):

- **Verdict:** `PASS`
- **Summary chiave:**
  - Runner **thin** e completamente delegante.
  - Façade orchestratore unico.
  - Logging sanificato per JSON.
  - L’intera suite `pytest` passa.

**Note del reviewer:**

- `binary_codes` è l’unico campo attualmente sanificato; in futuro nuovi tipi non JSON-safe potrebbero richiedere estensione dello `_sanitize_for_logging`.
- Determinismo:
  - tests confermano stabilità con stesso seed per:
    - façade,
    - runner,
    - full pipeline.

---

## 5. Invarianti confermati dopo T11

### 5.1. Invarianti architetturali

- ✅ Runner non contiene logica numerica.
- ✅ Façade concentra la pipeline Phase 1.
- ✅ Nessuna mutazione di registri globali.
- ✅ Nessuno scope creep (niente nuove metriche o modalità).

### 5.2. Invarianti numerici

- ✅ Proiezioni, binarizzazione, packing: identici a T3–T4.
- ✅ Similarità: cosine/Hamming identici a T5.
- ✅ Retrieval: top-k, nDCG, recall identici a T6.
- ✅ Classification: invariata (T7).
- ✅ Determinismo completo per stesso seed.

### 5.3. Invarianti di logging

- ✅ Logging schema identico a T8/T9 per Phase 1.
- ✅ Aggiunto solo strato di sanificazione JSON-safe.
- ✅ Nessun cambiamento di contenuto semantico nei risultati.

---

## 6. Timeline Errori & Fix (T11)

1. **Prima bozza:**
   - Runner ancora troppo “grasso”.
   - Logging fragile con numpy arrays.
   - Hostile → REVISE.

2. **Hotfix T11:**
   - Runner ridotto ai 4 passi core (config → seed → façade → logging).
   - Façade assorbe l’orchestrazione.
   - Aggiunto `_sanitize_for_logging` per `binary_codes`.
   - Test aggiornati per verificare che il runner:
     - non chiami eval modules direttamente,
     - usi solo la façade.

3. **Hostile Round v2:**
   - Tutti i test verdi.
   - Nessun drift di comportamento.
   - Verdict: **PASS**.

---

## 7. Stato Phase 1 dopo T11

| Task | Descrizione | Stato |
|------|-------------|-------|
| T1 | Skeleton project | ✅ |
| T2 | Core embeddings & dataset abstractions | ✅ |
| T3 | Binarizzazione & projection operators | ✅ |
| T4 | Bit packing/unpacking | ✅ |
| T5 | Similarity metrics (cosine, Hamming, overlap) | ✅ |
| T6 | Retrieval metrics (top-k, nDCG, recall) | ✅ |
| T7 | Classification evaluation wrapper | ✅ |
| T8 | Config, seed, IO, logging | ✅ |
| T9 | Phase 1 runner (v1, pre-refactor) | ✅ |
| T10 | BinaryEmbeddingEngine façade (orchestrator, v1) | ✅ |
| **T11** | **Runner→Façade refactor & logging hardening** | ✅ **COMPLETED** |

**Risultato:**  
🟩 **Phase 1 pipeline completamente consolidata, refactorata e valida.**  
T11 chiude il cerchio architetturale: abbiamo un **single entrypoint** pulito per tutte le future estensioni (Phase 2 – KV, blocks, runtime).

---

## 8. Next Steps (High-Level)

Ora che T11 è concluso, i passi naturali (da valutare insieme):

1. **T12 – Result Schema v2 Hardening**
   - Aggiungere:
     - `status`,
     - `error` object strutturato,
     - `normalization` block,
     - `system` metadata.
   - Allineare façade, runner e logger a uno schema unico, versionato.

2. **Phase 2 Architecture Kickoff**
   - Binary KV cache,
   - Binary blocks (attention/MLP),
   - End-to-end BinaryLLM runtime.

---

🔚 **T11 Log concluso.**  
Quando sei pronto, possiamo passare a:

> **“Prepariamo i prompt NVIDIA-grade per T12”**

oppure iniziare la **Phase 2 architecture design** in modalità Jensen Huang.  

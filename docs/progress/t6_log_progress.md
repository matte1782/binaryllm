## 10. Engineering Phase — T6 (Retrieval Metrics)

### 10.1 Task T6 Definition

> **T6 — Retrieval metrics and top-k neighbor evaluation**

Obiettivo T6 (in linea con `binaryllm_phase1_architecture_v1.md` §5, voce “T7 – Similarity & retrieval evaluation”):

- Implementare la parte **retrieval** del motore:
  - Top-k neighbor retrieval su:
    - cosine (baseline float embeddings),
    - Hamming (codici binari {0,1}).
  - Metriche di ranking:
    - **neighbor overlap@k** (H1 core),
    - **nDCG@k** (DCG con log₂, normalizzato per query),
    - **Recall@k**.
- Rendere il comportamento:
  - **deterministico** (seed obbligatorio),
  - **allineato al report H1** (cosine + Hamming, niente euristiche nascoste),
  - pienamente testato via `/tester_binaryllm` e validato da `/hostile_binaryllm`.

File principali coinvolti:

- `src/eval/retrieval.py`
- `tests/test_eval_retrieval.py`

### 10.2 Prima Iterazione T6 — Implementazione & Test

**/tester_binaryllm (T6 v1)**

Il tester ha definito una suite di test per:

- **Top-k retrieval**:
  - `topk_neighbor_indices_cosine` e `topk_neighbor_indices_hamming`.
  - Validazione di:
    - shape e tipi di input,
    - vincoli su `k` (1 ≤ k ≤ N),
    - presenza del parametro `seed` per determinismo.
- **Metriche di retrieval**:
  - `neighbor_overlap_at_k`:
    - computa |N_cos(i) ∩ N_ham(i)| / k mediata sulle query.
  - `ndcg_at_k`:
    - DCG con pesi `1/log₂(rank+1)` e **normalizzazione per IDCG**.
  - `recall_at_k`:
    - true positive / tot rilevanti per query, poi media.
- **Determinismo**:
  - test che richiamano top-k due volte con lo stesso seed e confrontano le liste.
  - test error-path per `seed=None` (richiesta ValueError con `seed` e `required` nel messaggio).

**/engineer_binaryllm (T6 v1)**

L’ingegnere ha implementato:

- `topk_neighbor_indices_cosine`:
  - calcola una matrice di similarità cosine su embedding L2-normalizzati (o assume normalized=True).
  - usa un ordinamento deterministico:
    - punteggio primario: similarità,
    - tie-breaking: priorità derivata da un permutation seed-based.
- `topk_neighbor_indices_hamming`:
  - delega a Hamming distance su codici {0,1},
  - ordina in modo analogo con priorità seed-based.
- `neighbor_overlap_at_k`:
  - implementato come pura sovrapposizione:
    - overlap@k(i) = |N_cos(i) ∩ N_ham(i)| / k,
    - risultato finale = media sulle query.
- `ndcg_at_k`:
  - per ogni query:
    - calcola DCG con log₂,
    - calcola IDCG da ranking ideale (relevance sort decrescente),
    - restituisce `DCG / IDCG` o 0 se IDCG=0.
- `recall_at_k`:
  - conta quanti elementi rilevanti entrano nel top-k,
  - divide per il numero totale di elementi rilevanti per query.

**/hostile_binaryllm (T6 v1 – REVISE)**

Il primo giro di `/hostile_binaryllm` ha trovato **violazioni di specifica**:

- `ndcg_at_k` inizialmente restituiva solo DCG (non normalizzato) pur chiamandosi nDCG.
- I top-k permettevano `seed=None` con tie-breaking deterministico “nascosto”, invece di forzare un errore come richiesto dalla spec.
- `_validate_k` era troppo restrittivo (`k >= N` invece di `k > N`).

Verdetto: **REVISE**, con richieste:

- Rendere `ndcg_at_k` una vera nDCG (DCG normalizzato per IDCG).
- Richiedere sempre un seed esplicito per il top-k (`seed=None` → ValueError).
- Allineare `_validate_k` a 1 ≤ k ≤ N.
- Aggiungere test espliciti per nDCG normalizzato e per il seed obbligatorio.

### 10.3 Seconda Iterazione T6 — Fix & Allineamento Test

**/tester_binaryllm (T6 v2)**

Il tester ha:

- Aggiornato i test nDCG per aspettarsi valori in [0, 1] (non raw DCG).
- Aggiunto test:

  - per `seed=None` che deve sollevare ValueError con `seed` e `required` nel messaggio;
  - per verificare che 1 ≤ k ≤ N sia accettato, mentre k > N sollevi errore.

- Sistemato `test_retrieval_metrics_are_deterministic` per:

  - passare sempre un seed esplicito sia per cosine sia per Hamming,
  - verificare determinismo ripetendo le chiamate con lo stesso seed.

**/engineer_binaryllm (T6 v2)**

L’ingegnere ha:

- Aggiornato `_validate_k` a:

  - accettare 1 ≤ k ≤ N,
  - rifiutare k > N con un messaggio esplicito.

- Implementato `ndcg_at_k` come **nDCG** corretta:

  - DCG con log₂,
  - IDCG per ogni query,
  - ritorno `dcg / idcg` (0 se idcg == 0).

- Modificato i top-k per:

  - richiedere `seed` come intero obbligatorio,
  - sollevare ValueError se `seed is None` con messaggio contenente `seed` e `required`.

**/hostile_binaryllm (T6 v2 – PASS)**

Il secondo giro di `/hostile_binaryllm` ha verificato:

- **Spec alignment:**

  - cosine/Hamming top-k implementano puro ordering per similarità/distanza,
  - overlap@k = |N_cos ∩ N_ham| / k, senza bonus o euristiche,
  - nDCG@k = DCG / IDCG con log₂, 0 se IDCG = 0,
  - recall@k = TP / tot rilevanti,
  - `_validate_k` con 1 ≤ k ≤ N,
  - `seed=None` → ValueError.

- **Test suite:**

  - tutti i test T6 passano (`pytest tests/test_eval_retrieval.py`),
  - l’intera suite `pytest -q` è verde,
  - determinismo completo confermato con seed fisso.

Verdetto finale T6:

> “T6 iteration 2 now fully matches the architecture: seeds are mandatory (tests cover both error paths and determinism), k-bounds follow 1 ≤ k ≤ N, and nDCG is DCG normalized by per-query IDCG with log2 discounts. No heuristics or scope creep were introduced, and both implementation and tests are aligned with spec.”

T6 è quindi **completamente congelato** come layer di retrieval Phase 1.

---

## 3️⃣ Prossimi passi (Jensen style)

Guardando l’architettura Phase 1  e ciò che hai già completato, la roadmap naturale è:

1. **T4 – IO utilities**
   - `src/utils/io.py` + `tests/test_utils_io.py`.
   - Obiettivo:
     - caricare embedding/label in base al `dataset_catalog`,
     - gestire NaN, file mancanti, formati errati con error schema (per dopo, T9),
     - preparare piccoli file synthetic sotto `tests/data/phase1_synthetic/`.
   - Questo sblocca l’integrazione end-to-end reale (non solo synthetic in memoria).

2. **T8 – Auxiliary classification evaluation**
   - `src/eval/classification.py` + test.
   - Linear classifier (logistic o ridge) su:
     - float embeddings,
     - binary embeddings (es. {−1,+1} come float).
   - Seedato via `utils/seed.py` (che arriverà in T10).
   - Importante: rimane **auxiliary** rispetto ad H1.

3. **T9 – Config & result schemas + logging**
   - `src/utils/config.py`, `src/utils/logging.py`, `tests/test_utils_config_and_logging.py`.
   - Definire:
     - schema config v2,
     - schema result (metrics, encoder/dataset, hypotheses ["H1"], git_hash, error object, ecc.) .
   - Questo è critico per avere vere run Phase 1 tracciabili.

4. **T10 – Seed management & determinism**
   - `src/utils/seed.py` + integrazione negli altri moduli.
   - Unificare:
     - seeding per proiezioni, classificazione, retrieval, eventuali random helper.
   - Test di run end-to-end deterministici.

5. **T11 – Variant façade**
   - `src/variants/binary_embedding_engine.py`.
   - Orchestrare pipeline:
     - config → dataset IO → embeddings → binarization → packing → similarity/retrieval/classification → logging.

6. **T12 – Runner & configs**
   - `src/experiments/runners/phase1_binary_embeddings.py`,
   - `src/experiments/configs/phase1_binary_embeddings.yaml`,
   - test per pipeline completa + regressione su config evolutivi.

Se vuoi, nel prossimo passo posso:

- scriverti i **tre prompt enterprise** per il prossimo task (io suggerirei di attaccare T4 IO utilities o T8 classification, ma T4 è più logico lato “pipeline”), come abbiamo fatto per T5/T6:
  - `/tester_binaryllm` (T4),
  - `/engineer_binaryllm` (T4),
  - `/hostile_binaryllm` (T4),

tutti già allineati alla spec, ai log e allo stile “NVIDIA / Jensen mode” che stiamo mantenendo.
::contentReference[oaicite:6]{index=6}

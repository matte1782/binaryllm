# BinaryLLM – Phase 1 – T12 Progress Log  
**Result Schema v2 & Error Pipeline**  
*Stato: COMPLETATO – Hostile Reviewer PASS (Round 3)*

---

## 1. Scopo di T12

**Obiettivo di T12:**

- Introdurre e congelare la **Result Schema v2** per la Phase 1.
- Rendere il **runner** una macchina a stati ben definita che:
  - In caso di successo, restituisce un payload **strutturato**, completo e stabile nel tempo.
  - In caso di errore, **non lancia eccezioni “nude”**, ma ritorna `status="error"` con un oggetto `error` ricco e coerente.
- Allineare **logging** + **runner** + **façade** con:
  - `version = "phase1-v2"`
  - `status ∈ {"success","error"}`
  - blocchi metrici separati:
    - `similarity_metrics`
    - `retrieval_metrics`
    - `classification_metrics`
  - campi extra:
    - `normalization`
    - `system` (python_version, platform, hostname)
    - `error` (stage, message, exception_type, ecc.)
    - `binary_codes` sanitizzate per il log.

Tutto senza rompere gli invariants già congelati in **T1–T11**.

---

## 2. Stato iniziale prima di T12

Prima di T12, il sistema era in modalità “v1 schema”:

- Runner (T9/T11):
  - Restituisce un dict con:
    - `encoder_name`, `dataset_name`, `code_bits`, `projection_type`, `runner`, `seed`
    - `metrics` (dict unico annidato)
    - `hypotheses`, `config_fingerprint`, `instrumentation`
  - Logging:
    - Scrive un JSON lineare con pochi campi richiesti (v1).
- Errori:
  - Alcuni casi lanciavano eccezioni direttamente (es. seed mancante, config invalidi).
  - Non esisteva un `status="error"` + `error={...}` strutturato.
- Nessun concetto formale di:
  - `version`
  - `system`
  - blocchi metrici separati (`similarity_metrics`, `retrieval_metrics`, `classification_metrics`).

T12 doveva quindi:
- Migrare tutto a **v2**,  
- Senza rompere T9/T10/T11,  
- E introducendo un **error pipeline** coerente.

---

## 3. Iterazione 1 – Hostile Review: REVISE

### 3.1 Problemi principali emersi

1. **Runner non completamente wrappato da try/except**

   - `_extract_seed` e altre validazioni iniziali lanciavano `ValueError` dirette.
   - T12 spec richiedeva: *“runner deve sempre ritornare un payload strutturato”*.
   - Quindi: alcune failure producevano ancora eccezioni “nude”.

2. **Test ancora su schema v1**

   - `tests/test_experiments_phase1_pipeline.py` ancora legato a:
     - `metrics` annidato,
     - assenza di `version`, `status`, `system`, ecc.
   - Nessun test che verificasse realmente:
     - `version = "phase1-v2"`
     - `status = "success" / "error"`
     - `normalization`, `system`
     - blocchi metrici separati.

3. **Runner-refactor tests (T11) in divergenza**

   - `tests/test_phase1_runner_refactor.py`:
     - Assumevano che il runner ritornasse **payload façade v1** “as-is”.
     - T12 aveva aggiunto il **wrapper v2** (envelope), rompendo il contratto dei test.

4. **Logging REQUIRED_LOG_FIELDS aggiornato, ma non testato**

   - `src/utils/logging.REQUIRED_LOG_FIELDS` aggiornato a v2,
   - Nessun test che verificasse:
     - che i log fossero conformi,
     - che il logger gestisse correttamente `binary_codes` numpy-heavy.

### 3.2 Azioni dopo Iterazione 1

- Definire un v2 schema chiaro per:
  - success payload,
  - error payload.
- Aggiornare i test runner/pipeline per:
  - **aspettarsi v2**,
  - coprire sia success che error.
- Progettare un nuovo set di test per `logging.write_log_entry` orientato a v2.

---

## 4. Iterazione 2 – Hostile Review: REVISE (focus logging)

Dopo i primi fix su runner + schema v2:

### 4.1 Problema critico: `tests/test_utils_logging.py` rotto

- Durante esecuzione:

  ```text
  SyntaxError: from __future__ imports must occur at the beginning of the file
Motivo:

Secondo from __future__ import annotations in mezzo al file (~line 105).

Inoltre:

La parte bassa del file conteneva ancora test v1 (_valid_log_entry, metrics top-level, nessun version/status/system).

4.2 Altre osservazioni
Error pipeline v2 (runner) in gran parte ok,

Ma mancava:

un test esplicito su errori di tipo config/seed,

una coverage robusta su write_log_entry con payload success + error.

5. Iterazione 3 – Fix finale & Hostile PASS
Questa iterazione ha chiuso definitivamente T12.

5.1 Pulizia di tests/test_utils_logging.py
Rimosso il secondo from __future__ import annotations.

Eliminato l’intero blocco legacy v1:

_valid_log_entry,

test che si aspettavano metrics top-level,

test che non consideravano version/status/system.

Il file ora contiene solo test v2 che verificano:

Success logging:

payload v2 completo (success) passato a write_log_entry.

write_log_entry:

non muta il payload originale,

scrive una riga JSON con sort_keys=True,

converge con REQUIRED_LOG_FIELDS aggiornato (incl. timestamp),

converte binary_codes da numpy a tipi JSON-friendly nella copia loggata.

Error logging:

payload v2 con status="error" + error={stage,message,exception_type}.

Stessi invariants:

nessuna mutazione del payload originale,

tutti i campi richiesti presenti,

gestione JSON sicura.

5.2 Runner – error pipeline & config/seed error
Runner ora:

Per i casi coperti da T12:

NaN nelle embeddings,

projection_type non supportato,

errori di configurazione/seed,

Non lancia eccezioni “nude” ma:

ritorna status="error",

popola error.exception_type,

popola error.stage con tag coerenti (load_embeddings, config_load, seed_extraction, ecc.),

riempie altri campi v2 con valori sensati o None in caso di errore precoce.

Caso speciale mantenuto (per compatibilità T9/T11):

UnknownDatasetError / UnknownEncoderError possono ancora propagare come eccezioni.

È una deroga consapevole alla regola “tutto wrappato”, mantenuta perché i test esistenti si aspettano pytest.raises(UnknownDatasetError) in certi scenari.

5.3 Verifica finale – Hostile Reviewer
Logging:

REQUIRED_LOG_FIELDS allineato a v2.

write_log_entry:

fa deepcopy,

passa da _sanitize_payload per binary_codes,

non muta l’input,

genera JSON line v2-compliant.

Tests:

tests/test_utils_logging.py → Import OK, nessun futuro doppio.

Nessuna traccia di schema v1.

Runner:

Success path produce payload v2 corretto.

Error path produce payload v2 con status="error" + error.

Determinismo confermato:

Stessa config + seed → stessi:

blocchi metrici,

binary_codes,

meta-dati chiave (version, encoder_name, dataset_name, projection_seed, status).

Unica eccezione: timestamp + tempi di instrumentation, per definizione variabili.

Esito finale:
✅ Hostile Reviewer T12 Round 3 → PASS
✅ pytest -q → suite completa verde

6. Result Schema v2 – Stabile & Congelata
6.1 Success payload (forma concettuale)
jsonc
Copia codice
{
  "version": "phase1-v2",
  "status": "success",
  "encoder_name": "...",
  "dataset_name": "...",
  "dataset_format": "npy|parquet|...",
  "code_bits": 32,
  "projection_type": "gaussian",
  "projection_seed": 123,
  "runner": "phase1_binary_embeddings",
  "seed": 123,
  "hypotheses": ["H1"],
  "metrics_requested": ["similarity", "retrieval", "classification"],
  "normalization": { "l2": true },
  "similarity_metrics": { ... },
  "retrieval_metrics": { ... },
  "classification_metrics": null | { ... },
  "instrumentation": { "total_ms": ..., ... },
  "system": {
    "python_version": "...",
    "platform": "...",
    "hostname": "..."
  },
  "binary_codes": {
    "pm1": <N x code_bits>,
    "01": <N x code_bits>,
    "packed": <N x ceil(code_bits/64)>
  },
  "error": null,
  "timestamp": "ISO8601..."
}
6.2 Error payload (forma concettuale)
jsonc
Copia codice
{
  "version": "phase1-v2",
  "status": "error",
  "encoder_name": "... or null",
  "dataset_name": "... or null",
  "dataset_format": "... or null",
  "code_bits": 32 or null,
  "projection_type": "gaussian|... or null",
  "projection_seed": 123 or null,
  "runner": "phase1_binary_embeddings",
  "seed": 123 or null,
  "hypotheses": [... or []],
  "metrics_requested": [... or []],
  "normalization": { "l2": ... } or null,
  "similarity_metrics": null,
  "retrieval_metrics": null,
  "classification_metrics": null,
  "instrumentation": {},
  "system": {
    "python_version": "...",
    "platform": "...",
    "hostname": "..."
  },
  "binary_codes": null,
  "error": {
    "stage": "config_load | seed_extraction | load_embeddings | engine_init | engine_run | ...",
    "message": "dettagli umani dell’errore",
    "exception_type": "ValueError | RuntimeError | ...",
    // opzionali: dataset_name, encoder_name, ecc.
  },
  "timestamp": "ISO8601..."
}
7. Lessons Learned – Jensen / NVIDIA Mode
Schema v2 deve essere blindato nei test, non solo nei commenti.

I test di logging sono fondamentali quanto quelli di math:

se i log non sono stabili → la riproducibilità salta.

Ogni cambiamento a:

REQUIRED_LOG_FIELDS,

error pipeline,
richiede nuova copertura test mirata.

Alcune deroghe alla teoria (es. UnknownDatasetError che esce) vanno:

decise consapevolmente,

documentate in codice e nei test,

non “riparate” in modo casuale in un refactor successivo.

from __future__ import ... va sempre all’inizio del file
(errore banale ma blocca l’intera suite).

8. Stato Phase 1 dopo T12
Task	Stato
T1 – Skeleton	✅
T2 – Core Embeddings	✅
T3 – Binarization	✅
T4 – Packing	✅
T5 – Similarity	✅
T6 – Retrieval Metrics	✅
T7 – Classification Eval	✅
T8 – Config / IO / Logging v1	✅
T9 – Phase1 Runner (v1 schema)	✅
T10 – BinaryEmbeddingEngine	✅
T11 – Runner Refactor → Façade	✅
T12 – Result Schema v2 + Error Pipeline	✅ (APPENA COMPLETATO)

Phase 1 ora ha:

schema v2 stabile,

error pipeline strutturata,

logging deterministico,

pipeline end-to-end congelata pronta per Phase 2.


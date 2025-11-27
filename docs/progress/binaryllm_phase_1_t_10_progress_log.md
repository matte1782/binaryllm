# BinaryLLM – Phase 1 – T10 Progress Log

## **BinaryEmbeddingEngine Façade – Full Engineering & Validation Report**
**Stato:** Validato (Hostile Reviewer v2 – PASS)

Questo documento riassume in maniera completa e strutturata tutte le fasi, i problemi, le revisioni e le soluzioni applicate durante il Task T10 (BinaryEmbeddingEngine) della Phase 1 di BinaryLLM.

---
## **1. Obiettivo del Task T10**
Creare la façade unificata `BinaryEmbeddingEngine`, responsabile dell’orchestrazione dell’intera pipeline Phase 1:
- Normalizzazione
- Proiezione randomica
- Binarizzazione
- Bit‑packing
- Similarity (T5)
- Retrieval (T6)
- Classification (T7, delegata)
- Rispetto totale degli invariants architetturali
- Zero side-effects e totale determinismo

La façade non introduce nuova logica numerica, ma coordina moduli già validati.

---
## **2. Problemi rilevati – Iterazione 1 (REVISE)**

### **❌ Issue 1 – Constructor validation assente**
Il costruttore non generava messaggi di errore corretti per campi mancanti. Si generava un `TypeError` nativo, non un `ValueError` con il campo specifico.

**Risoluzione:** Implementata validazione esplicita con messaggi regex‑friendly.

---
### **❌ Issue 2 – mean_hamming non normalizzato**
La metrica usava Hamming “grezzo”, con valori >10. Il contratto Phase 1 richiede **H_norm = H / code_bits ∈ [0,1]**.

**Risoluzione:** Implementato Hamming normalizzato.

---
### **❌ Issue 3 – Logica di classification nella façade**
Il façade eseguiva:
- training classifier
- calcolo F1

Questo violava l’invariant:
> "All eval math must reside under src/eval/** and never inside the façade."

**Risoluzione:** Spostato tutto in `src/eval/classification`. Engine ora delega.

---
### **❌ Issue 4 – Mancato rispetto dei capability flags**
Il façade calcolava metriche anche su dataset che dichiarano `supports_* = False`.

**Risoluzione:** Aggiunto `_ensure_metric_supported`.

---
### **❌ Issue 5 – Mutazione dei registri globali**
La façade inseriva dataset/encoder nei registri globali senza dichiararlo.

**Risoluzione:** Rimosso completamente. Zero global state mutation.

---
### **❌ Issue 6 – Error messages inconsistenti**
Alcuni errori non includevano il nome del campo richiesto.

**Risoluzione:** Tutti gli errori ora rispettano strict pattern.

---
## **3. Problemi rilevati – Iterazione 2 (REVISE)**

### **❌ Issue 7 – projection_type error contract**
Il test richiede il pattern:
```
projection_type.*unsupported
```
l’engine produceva messaggio diverso.

**Risoluzione:** Corretto messaggio: ora include `unsupported`.

---
### **❌ Issue 8 – retrieval_k <= 0 non sollevava errore**
`retrieval_k=0` veniva letto come `fallback_k` → nessun errore.

**Risoluzione:** Validazione esplicita prima del fallback.

---
## **4. Validazione Finale – Hostile Reviewer v2 (PASS)**
Tutte le correzioni applicate sono state validate.

### **Invariants confermati:**
- binary_codes pm1, 01 → shape corretta `(N, code_bits)`
- packed codes → compatibili T4
- mean_hamming ∈ [0,1]
- mean_cosine ∈ [-1,1]
- spearman similarity metric valida
- retrieval: top‑k deterministic, nDCG normalizzato, recall funzionante
- classificazione delegata correttamente
- nessun drift o side effect globale
- nessuna violazione architetturale

**Phase 1 resta deterministica al 100%.**

---
## **5. Lessons Learned (Jensen/NVIDIA Mode)**
1. Il façade non deve mai introdurre nuova logica, solo orchestrare.
2. Gli errori devono essere sempre espliciti, regex‑friendly, field‑name‑first.
3. Zero mutazioni globali.
4. Invariants architetturali sacri.
5. Test‑first → no heuristics.

---
## **6. Stato della Phase 1 dopo T10**
| Task | Stato |
|------|--------|
| T1 – Skeleton | ✔️ |
| T2 – Core Embeddings | ✔️ |
| T3 – Binarization | ✔️ |
| T4 – Packing | ✔️ |
| T5 – Similarity | ✔️ |
| T6 – Retrieval | ✔️ |
| T7 – Classification | ✔️ |
| T8 – Config/Logging/IO | ✔️ |
| T9 – Runner | ✔️ |
| **T10 – Façade** | **✔️ COMPLETATO** |

Phase 1 → **COMPLETATA AL 100%**.

---
## **7. Prossimi Step**
- **T11 – Façade Extraction & Refactor**
- **T12 – Logging & Config Schema Upgrade (v2)**
- Preparazione Phase 2 Architecture.

---


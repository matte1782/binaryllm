1. Goal

Primary objective (Phase 1 ONLY)

Deliver a Binary Embedding & Similarity Engine v0.1 that:

Ingests precomputed float embeddings from existing encoders.

Applies a well-specified projection + binarization + packing pipeline to produce binary codes.

Evaluates similarity preservation (Hamming vs cosine), retrieval quality, and optional auxiliary classification.

Logs structured, reproducible artifacts (metrics + metadata) that directly target H1 of binaryllm_report_v2.md.

Out of scope for Phase 1

No training of encoders, no Transformer blocks, no KV-cache, no full runtime. Only frozen embeddings, projection/binarization, evaluation, and logging.

2. Constraints & Alignment

Alignment with docs/binaryllm_report_v2.md

Implements and measures H1 – Binary embeddings preserve retrieval neighborhoods at sufficient code length:

Uses the random hyperplane / Hamming vs cosine theory (§3.3–3.4) as the reference for similarity metrics.

Enforces L2 normalization by default for H1 experiments to satisfy the angular-distance assumptions.

Respects rate–distortion (§3.5): 1‑bit is lossy; metrics and comments do not assume near-losslessness.

Treats classification as auxiliary, not a core hypothesis, unless explicitly tied to later hypotheses.

Relevant Project Rules

Architecture invariants (Project §3):

src/core/ = embeddings abstractions + dataset catalog (no experiment logic).

src/quantization/ = binarization and packing; no dataset or experiment logic.

src/eval/ = metrics only; no model/binarization logic.

src/experiments/ = configs + runners only; no math logic.

src/variants/ = high-level pipeline façades; no direct I/O.

Numerical & binary invariants (Project §4):

Explicit bit-width parameters: code_bits, projection_type, projection_seed.

Binary conventions: internal {−1,+1}, storage {0,1}, with fixed mapping -1→0, +1→1.

Experiment & reproducibility (Project §5):

All experiments are config-driven, seeded, logged with config + hypotheses + system metadata.

User “Military Strict Mode”:

No hidden responsibilities, no vague APIs, no “magic” behavior.

All assumptions explicit, all flows test-first and falsifiable.

3. Files & Modules Impacted

3.1 Existing files (referenced, not modified)

docs/binaryllm_report_v2.md

Responsibility: Validated research foundation (including H1).

README.md

Responsibility: High-level project overview and phase description.

3.2 TO BE CREATED – Core source modules (Phase 1 only)

src/core/__init__.py

Responsibility: Declare core package namespace (no logic).

src/core/embeddings.py

Responsibility: Define FloatEmbeddingBatch and BinaryCodeBatch as logical (unpacked) representations of embeddings and binary codes.

Clarification: BinaryCodeBatch holds only logical {−1,+1} and {0,1} arrays and metadata; it does not own any packed storage or packing logic.

src/core/datasets.py

Responsibility: Provide lightweight dataset wrappers (embedding corpora, queries, classification sets) built from files defined in the dataset catalog.

src/core/dataset_catalog.py

Responsibility: Central registry mapping:

dataset_name → file formats, required columns, expected embedding dimension, normalization requirement.

encoder_name → expected embedding dimension, dtype, file conventions (paths, layouts).

src/quantization/__init__.py

Responsibility: Declare quantization package namespace.

src/quantization/binarization.py

Responsibility: Implement projection + binarization operators on float embeddings (purely logical).

Includes:

Sign/scale-sign binarizers ({-1,+1} output).

Projection operators parameterized by projection_type and projection_seed.

src/quantization/packing.py

Responsibility: Convert logical {0,1} code arrays into bit-packed buffers and back, with explicit word size and layout.

Clarification: This is the only module that knows about bit-packing; BinaryCodeBatch remains unpacked.

src/eval/__init__.py

Responsibility: Declare eval package.

src/eval/similarity.py

Responsibility: Compute cosine similarities (float) and Hamming distances (binary), plus correlation/monotonicity metrics.

src/eval/retrieval.py

Responsibility: Implement retrieval under cosine and Hamming, compute top‑k neighbor overlap, nDCG@k, Recall@k.

src/eval/classification.py

Responsibility: Auxiliary, evaluation-only classification on float vs binary embeddings (simple linear classifier interface; no general training loops).

src/utils/__init__.py

Responsibility: Declare utils package.

src/utils/io.py

Responsibility: Load embeddings, labels, and metadata from formats described in dataset_catalog, and write packed codes/metrics artifacts.

src/utils/seed.py

Responsibility: Single entrypoint to set RNG seeds across libraries (e.g., Python, NumPy, PyTorch if used).

src/utils/config.py

Responsibility: Load and validate Phase 1 experiment configs against the explicit schema v2, failing hard on unknown or malformed fields.

src/utils/logging.py

Responsibility: Emit metrics and metadata following the result schema v2 (JSON/CSV), including error records.

src/experiments/__init__.py

Responsibility: Declare experiments package.

src/experiments/configs/phase1_binary_embeddings.yaml (and additional configs under src/experiments/configs/phase1/)

Responsibility: Store Phase 1 experiment configs adhering to the config schema v2.

src/experiments/runners/phase1_binary_embeddings.py

Responsibility: Read config, set seeds, instantiate datasets and variants, invoke the variant façade, and write logs/artifacts (no numerical logic).

src/variants/__init__.py

Responsibility: Declare variants package.

src/variants/binary_embedding_engine.py

Responsibility: Single façade for Phase 1 pipeline (Binary Embedding & Similarity Engine v0.1) that:

Accepts a fully validated, in-memory config object.

Wires together datasets, binarization, packing, eval modules.

Returns an in-memory result structure suitable for logging.

Clarification:

Runners call only this façade for Phase 1; the façade does not call runners.

No responsibility overlap: variants = pipeline logic, runners = config + orchestration + logging calls.

3.3 TO BE CREATED – Tests & data

tests/__init__.py

Responsibility: Declare tests package.

tests/test_quantization_binarization.py

Responsibility: Unit tests for binarization operators and bit-packing (numerical correctness + invariants).

tests/test_eval_similarity.py

Responsibility: Unit tests for similarity and Hamming vs cosine relations.

tests/test_eval_retrieval.py

Responsibility: Unit tests for retrieval, neighbor overlap, and tie-handling determinism.

tests/test_eval_classification.py

Responsibility: Tests for auxiliary classification evaluation, including degradation expectations.

tests/test_utils_config_and_logging.py

Responsibility: Validate config schema enforcement and logging result schema (keys and types), including failure-path logs.

tests/test_experiments_phase1_pipeline.py

Responsibility: Integration tests for the Phase 1 end-to-end pipeline (config → runner → façade → logs).

tests/test_regression_phase1_configs.py

Responsibility: Regression tests ensuring backward compatibility for existing configs and metrics.

tests/test_determinism_and_performance.py

Responsibility: Deterministic-run tests (same seed → same metrics) and basic performance sanity tests (binarization/packing time in reasonable bounds).

tests/data/phase1_synthetic/ (TO BE CREATED directory)

Responsibility: Contains small synthetic datasets (embeddings + labels/queries) used by unit/integration tests:

Well-controlled angles and neighbor structures.

Known decision boundaries for classification.

Fixed, versioned artifacts.

tests/data/golden/ (TO BE CREATED directory)

Responsibility: Contains golden-run artifacts (configs + expected metrics JSON) for regression tests:

Reference runs for specific code_bits, projection types.

Used to detect unintended changes in metrics or logging schema.

4. Design & Data Flow

4.1 Ownership Boundaries (v2 clarification)

BinaryCodeBatch (in src/core/embeddings.py)

Holds only:

Logical codes in {−1,+1} and/or {0,1} space.

Metadata: code_bits, num_samples, normalized flag for source embeddings, etc.

Does not know about packed storage, word sizes, or GPU layouts.

Packing (src/quantization/packing.py)

Owns:

Mapping {0,1} matrices → packed bit buffers.

Inverse mapping packed buffers → {0,1} matrices.

Bit layout and hardware-facing details (word size, alignment).

Embedding/variant code that needs packed representation calls this module explicitly.

4.2 End-to-end Phase 1 pipeline

Config loading & validation (src/utils/config.py)

Inputs:

YAML/JSON file adhering to config schema v2.

Steps:

Load file.

Validate fields and values (see 4.3).

Reject configs with:

Missing required fields,

Unsupported values,

Unknown keys (hard failure).

Output:

An in-memory, typed config object passed to src/variants/binary_embedding_engine.py.

Dataset selection & IO (src/core/dataset_catalog.py, src/core/datasets.py, src/utils/io.py)

Use dataset_name and encoder_name from config to:

Look up expected file paths, formats, columns, embedding dimension, normalization requirement.

IO module loads:

Float embeddings (float32/float16 as specified).

Query sets and relevance labels or ground-truth neighbors.

Classification labels if configured.

Dataset wrappers enforce:

Shape/dtype consistency with catalog expectations.

Handling of NaNs/Infs, empty sets, dimension mismatches:

Trigger structured error logging (see 4.6) and abort run.

Pre-processing & normalization (src/core/embeddings.py)

FloatEmbeddingBatch enforces:

shape = (N, d) with N ≥ 1, d ≥ 1.

For H1-tagged experiments:

L2 normalization MUST be applied by default (normalized=True).

If normalization is disabled via config:

The normalized flag is False.

Logs must record normalization: false in result schema.

Projection & binarization (src/quantization/binarization.py)

Projection operator is selected via projection_type and projection_seed.

Supported projection_type values:

"gaussian":

Each projection vector ~ N(0, I), initialized deterministically from projection_seed.

"rademacher":

Each projection vector has i.i.d. entries in {−1,+1}, scaled appropriately, seeded.

"prelearned_linear":

Uses pre-existing linear projection parameters loaded from disk (path taken from config/catalog); no randomness at init time, but still logs projection_seed if used elsewhere.

For "gaussian" and "rademacher":

projection_seed is mandatory and controls all randomness.

Binarization:

Apply projection to produce projected floats shape (N, code_bits).

Apply sign (or scaled-sign) binarizer to obtain {−1,+1} codes.

Map to {0,1} using fixed mapping -1→0, +1→1.

Packing (src/quantization/packing.py)

Input: {0,1} code matrix of shape (N, code_bits).

Layout:

Word size: 64-bit (e.g., conceptual uint64 words).

Bit ordering within a word: least-significant bit (LSB) corresponds to the smallest column index in the packed group.

Row-major layout: codes for each sample are stored contiguously across words.

Alignment:

Packed arrays are conceptually aligned to 8-byte boundaries for GPU-friendliness.

Output:

Packed buffer representation + metadata (e.g., bits per code, padding).

Reverse operation (unpack) is exact and tested.

Similarity & retrieval evaluation (src/eval/similarity.py, src/eval/retrieval.py)

Similarity:

Compute cosine similarity matrix (or on-demand pairs) from normalized float embeddings.

Compute Hamming distances from {0,1} codes (no need to unpack if downstream uses bit-ops; but Phase 1 can operate either on logical or packed representation).

Derive correlation/monotonicity metrics between cosine similarity and (1 − normalized Hamming).

Retrieval:

For each query:

Obtain top‑k neighbors under cosine (float baseline).

Obtain top‑k neighbors under Hamming (binary).

Compute:

Top‑k neighbor overlap (H1 metric).

nDCG@k, Recall@k.

Auxiliary classification (src/eval/classification.py)

Phase 1 classification scope:

Evaluation-only, using simple linear or logistic classifiers over embeddings.

Training location:

A small, self-contained routine inside src/eval/classification.py trains classifiers on:

Float embeddings,

Binary embeddings (e.g., {−1,+1} mapped to floats).

Randomness control:

All classifier training randomness (e.g., parameter initialization, data shuffling) is seeded using src/utils/seed.py.

Outputs:

Accuracy/F1 and relative degradation from float to binary.

Status:

Auxiliary, non-blocking: missing or failing classification does not block H1 success but must be logged and tested as specified.

Variant façade (src/variants/binary_embedding_engine.py)

Receives:

Validated config object + handles to I/O and logging utilities.

Orchestrates:

Dataset loading via catalog/datasets/io.

Embedding normalization.

Projection + binarization + packing.

Similarity/retrieval/classification evaluations as requested by metrics[].

Instrumentation timing: measure binarization_time_ms, packing_time_ms (and optionally other key stages).

Returns:

An in-memory result object matching the result schema v2 (metrics + metadata + instrumentation).

Results logging (src/utils/logging.py)

Uses the result object from façade to:

Emit a JSON record (and optionally CSV row) with fields exactly matching the result schema v2.

If any stage fails, emit a structured error record (see 4.6).

Runner (src/experiments/runners/phase1_binary_embeddings.py)

Main steps:

Parse CLI arguments or environment to locate config file.

Call config.load_and_validate() to obtain config object.

Call seed.set_global_seed(config.seeds.main) (and additional seeds if present).

Call binary_embedding_engine.run(config) to get result object.

Call logging.write_result(result, output_dir) to persist artifacts.

4.3 Config schema v2 (binding)

Location: Implemented by src/utils/config.py. Applied to configs under src/experiments/configs/phase1/.

Top-level fields:

version (string, required)

Allowed: semantic-like strings, e.g., "1.0", "1.1".

Must match a known schema version; otherwise fail.

encoder_name (string, required)

Must be one of the names registered in dataset_catalog.encoder_registry.

Used to look up encoder metadata (dimension, dtype, file conventions).

embedding_files (list of strings, required, non-empty)

Each string: file path or glob as defined by dataset catalog for the chosen encoder/dataset.

Must exist and be compatible with dataset_format.

dataset_name (string, required)

Must be one of the names registered in dataset_catalog.dataset_registry.

dataset_format (string, required)

Allowed values: "npy", "parquet", "hdf5", "custom" (if catalog supports it).

Must match the format specified in the dataset catalog for dataset_name.

projection_type (string, required)

Allowed: "gaussian", "rademacher", "prelearned_linear".

For "prelearned_linear", additional fields (e.g., projection_path) become required.

code_bits (integer, required)

Must be positive, >= 1.

Must belong to a supported set (e.g., {32, 64, 128, 256}) enforced by config.

metrics (list of strings, required, non-empty)

Allowed elements:

"similarity"

"retrieval"

"classification" (auxiliary)

Unknown metric names → hard failure.

hypotheses (list of strings, required)

For Phase 1 configs: must include "H1".

Additional hypotheses can be allowed in the future but must be known IDs from the report.

seeds (object, required)

Fields:

main (integer, required): master seed.

Optional: projection, classification, etc. (integers).

If sub-seeds not provided, they derive deterministically from main.

output_dir (string, required)

Directory for logs and artifacts.

Must be writable; failure to create → hard error.

logging (object, required)

Fields:

level (string, required): "info", "debug", "warning", "error".

Optional: format (e.g., "json", "csv+json").

normalization (object, optional)

Fields:

l2 (boolean, default true for H1 experiments).

For H1, if l2=false, config is allowed but:

Must be explicitly logged as normalization.l2=false.

projection_params (object, optional)

For "gaussian" / "rademacher":

E.g., scale, dimension overrides.

For "prelearned_linear":

projection_path (string, required).

Unknown top-level or nested keys:

Result in hard validation failure with a structured error indicating the offending key and path.

4.4 Result schema v2 (binding)

Location: Implemented and enforced by src/utils/logging.py. Each run produces at least one JSON record:

Top-level fields:

version (string)

Mirrors config schema version, e.g., "1.0".

status (string)

"success" or "error".

encoder_name (string)

dataset_name (string)

dataset_format (string)

code_bits (integer)

projection_type (string)

projection_seed (integer or null)

metrics_requested (list of strings)

hypotheses (list of strings, e.g., ["H1"])

normalization (object)

l2 (boolean)

similarity_metrics (object or null)

Present iff "similarity" requested and run succeeded.

Fields (examples):

cosine_hamming_spearman (float)

mean_cosine (float)

mean_hamming (float)

Any additional clearly-named scalar metrics.

retrieval_metrics (object or null)

Present iff "retrieval" requested and run succeeded.

Fields:

topk_overlap (object)

Keys like "k=10", "k=50" → float overlap values.

ndcg (object)

"k=10", "k=50" → floats.

recall (object)

"k=10", "k=50" → floats.

classification_metrics (object or null)

Present iff "classification" requested and run succeeded.

Fields:

float_accuracy, binary_accuracy (floats)

float_f1, binary_f1 (floats)

accuracy_delta (float, binary − float)

instrumentation (object)

Fields:

binarization_time_ms (float)

packing_time_ms (float)

Optional: total_run_time_ms, etc.

system (object)

Fields:

cpu_model (string or null)

gpu_model (string or null)

num_gpus (integer)

git_hash (string or null)

If not available (e.g., not a git repo), set to null explicitly.

timestamp (string, ISO8601)

error (object or null)

If status="error", contains:

message (string)

stage (string; e.g., "load_embeddings", "projection_setup", "run_eval")

variant (string; "binary_embedding_engine_v0.1")

dataset_name (string or null)

file_path (string or null)

Optional: stack/exception type as string.

This schema is binding: tests must assert presence and type of these fields.

4.5 Link to H1

Configs:

All Phase 1 configs must set hypotheses: ["H1"] (at minimum).

Logging:

Result schema includes hypotheses field; H1 is always recorded when applicable.

Tests:

At least one integration test uses increasing code_bits and asserts monotonic or non-degrading neighbor overlap trends, explicitly labeled as an H1 scenario.

4.6 Error propagation rules

On:

NaNs/Infs in embeddings,

Empty dataset or query sets,

Shape/dimension mismatches,

Unsupported code_bits or projection_type,

The system must:

Log a structured error record matching the error object schema (with stage, variant, dataset_name, file_path).

Set status="error" and leave metrics sections as null.

Raise a clear exception at the runner level (no silent continuation or fallback to float-only).

4.7 GPU feasibility preparation

Bit-packing layout:

64-bit words, LSB-first within each word, row-major, 8‑byte alignment.

This is documented in src/quantization/packing.py and in developer docs.

Instrumentation hooks:

binarization_time_ms and packing_time_ms are mandatory, measured around:

Projection + binarization,

Packing steps.

Even though Phase 1 is CPU-centric, these timings will:

Provide baseline complexity checks,

Inform future CUDA implementations.

5. Task Decomposition (Micro-tasks)

Each task is scoped for ≤1 /engineer_binaryllm iteration.

T1 – Create base package skeleton

Description: Create src/, src/core/, src/quantization/, src/eval/, src/utils/, src/experiments/, src/variants/, tests/ with minimal __init__ files.

Files: All package __init__.py files.

Preconditions: None.

Tests: Simple import smoke test.

Risks: None major.

T2 – Implement dataset catalog

Description: Define dataset_catalog registries for dataset_name and encoder_name, including required file formats, columns, dimensions, normalization requirements.

Files: src/core/dataset_catalog.py.

Preconditions: T1.

Tests: Validate lookups and mismatches (dataset/encoder not registered) raise clear errors.

Risks: Overfitting to specific datasets; must stay extensible.

T3 – Embedding & dataset abstractions

Description: Implement FloatEmbeddingBatch, BinaryCodeBatch (logical only) and dataset wrappers that rely on catalog definitions.

Files: src/core/embeddings.py, src/core/datasets.py.

Preconditions: T2.

Tests: Shapes, dtypes, normalization flag invariants.

Risks: Accidentally leaking packing responsibilities into BinaryCodeBatch.

T4 – IO utilities

Description: Implement IO helpers that load embeddings/labels according to catalog formats; handle NaNs/empty inputs with structured errors.

Files: src/utils/io.py.

Preconditions: T2, T3.

Tests: Load synthetic files in tests/data/phase1_synthetic/; malformed files trigger error logs.

T5 – Binarization operators

Description: Implement projection operators (gaussian, rademacher, prelearned_linear) and sign/scale-sign binarization.

Files: src/quantization/binarization.py.

Preconditions: T3.

Tests: Sign convention tests, projection seeding determinism, small theoretical Hamming vs angle sanity checks.

Risks: Misalignment with report definitions for binarization.

T6 – Bit packing

Description: Implement {0,1} ↔ packed bit conversions with specified 64-bit layout and alignment.

Files: src/quantization/packing.py.

Preconditions: T5.

Tests: Round-trip correctness, edge cases, performance sanity.

Risks: Subtle bit-order bugs affecting Hamming distance.

T7 – Similarity & retrieval evaluation

Description: Implement cosine/Hamming similarity metrics and retrieval (top‑k, overlap, nDCG, Recall).

Files: src/eval/similarity.py, src/eval/retrieval.py.

Preconditions: T3, T5, T6.

Tests: Toy datasets and synthetic angles; deterministic tie-handling with seeds.

Risks: Incorrect normalizations or distance definitions.

T8 – Auxiliary classification evaluation

Description: Implement evaluation-only classification (linear classifier) for float vs binary, with deterministic training.

Files: src/eval/classification.py.

Preconditions: T3.

Tests: Known decision-boundary toy dataset; check degradation.

Risks: Scope creep into general-purpose training; must remain restricted.

T9 – Config & result schemas + logging

Description: Implement config schema v2 (including unknown-key rejection) and result schema v2; implement logging utilities and tests.

Files: src/utils/config.py, src/utils/logging.py, tests/test_utils_config_and_logging.py.

Preconditions: T1, T2.

Tests: Schema validation, unknown-key rejection, correct JSON field types (including error records).

Risks: Drift between schemas and actual logged fields.

T10 – Seed management & determinism

Description: Implement set_global_seed and enforce use in façade and evaluation modules.

Files: src/utils/seed.py, updates in façade and tests.

Preconditions: T1.

Tests: Deterministic-run tests for full pipeline with fixed config and seeds.

Risks: Hidden random sources not controlled.

T11 – Variant façade

Description: Implement binary_embedding_engine façade wiring config → datasets → binarization → packing → eval → results, including instrumentation.

Files: src/variants/binary_embedding_engine.py.

Preconditions: T3–T7, T9, T10.

Tests: Smoke integration tests; verify result object matches result schema v2.

Risks: Responsibility leakage from façade into runner or core modules.

T12 – Runner and configs

Description: Implement the Phase 1 runner and example configs under src/experiments/configs/phase1/ using the defined schema (with hypotheses: ["H1"]).

Files: src/experiments/runners/phase1_binary_embeddings.py, src/experiments/configs/phase1_binary_embeddings.yaml.

Preconditions: T9, T11.

Tests: tests/test_experiments_phase1_pipeline.py on synthetic data; config evolution regression tests in tests/test_regression_phase1_configs.py.

Risks: Bypassing façade; configs diverging from schema.

T13 – Synthetic & golden datasets

Description: Create synthetic and golden data + metrics artifacts for tests and regression.

Files: tests/data/phase1_synthetic/, tests/data/golden/, and corresponding tests.

Preconditions: T3, T7, T8, T11.

Tests: Golden-run comparison in tests/test_regression_phase1_configs.py.

Risks: Overfitting to synthetic patterns; must keep synthetic but representative.

6. Test Plan

6.1 Pytest commands

All tests:

pytest -q

Focused development subsets:

Core quantization + eval: pytest tests/test_quantization_binarization.py tests/test_eval_similarity.py tests/test_eval_retrieval.py

Config + logging: pytest tests/test_utils_config_and_logging.py tests/test_regression_phase1_configs.py

Full pipeline: pytest tests/test_experiments_phase1_pipeline.py tests/test_determinism_and_performance.py

6.2 Expected signals

Correctness & invariants

Binarization:

Exact sign convention; random projections produce empirical Hamming distances compatible with theory.

Packing:

unpack(pack(codes)) == codes for all test matrices.

Similarity/retrieval:

Identity behaviors (self-similarity, self-nearest neighbors) are correct.

Classification:

Known decision-boundary synthetic data yields expected classification metrics; binary degradation within expected bounds.

Determinism & config robustness

Re-running the same config with the same seeds yields identical metrics and logs (except timestamps).

Unknown config keys and malformed values reliably cause test failures (and proper error records).

H1-specific check

On synthetic H1-like data, increasing code_bits (e.g., 32→64→128) yields monotonic or non-degrading neighbor overlap trends, explicitly labeled as an H1 scenario.

Error handling

Injected errors (NaNs, empty inputs, shape mismatches) produce:

status="error",

Correctly filled error object (stage, variant, dataset_name, file_path),

No spurious metrics fields.

Performance sanity

binarization_time_ms and packing_time_ms for synthetic datasets are within reasonable bounds (no catastrophic blowup relative to problem size, e.g., linear scaling in N and code_bits for tested sizes).

6.3 Metrics relevance

Quality metrics

H1: neighbor overlap, nDCG@k, Recall@k, cosine-Hamming correlation.

Auxiliary: classification accuracy/F1 before and after binarization.

Systems metrics

Instrumentation times (binarization/packing) as early checks on computational feasibility.

System metadata (CPU/GPU model) logged for future comparison, even though Phase 1 doesn’t do deep systems benchmarking.

7. Risk Assessment

Numerical risks

Mis-implementation of sign or projection seeding could silently distort Hamming vs cosine relationships and invalidate H1 conclusions.

Incorrect or inconsistent normalization (or mis-logged normalization state) would break alignment with the random-hyperplane assumptions.

Architecture risks

If variant façade starts duplicating runner or IO responsibilities, separation of concerns would erode; explicit responsibilities and tests around façade usage mitigate this.

Dataset catalog may drift from actual file layouts if not maintained; centralizing all dataset/encoder definitions in one module and testing them with synthetic/golden data reduces this risk.

Feasibility & GPU risks

Bit-packing layout chosen now constrains later GPU kernels; documenting it explicitly and testing round-trips minimizes the risk of future incompatibilities.

Instrumentation is light but crucial; if omitted or incorrectly implemented, we’d lack baseline complexity estimates for GPU work.

Reproducibility risks

Without strict schema enforcement and regression tests, results might become incomparable across iterations; binding config/result schemas and adding golden tests directly addresses this.

Approved ARCHITECT plan v2.0 — ready for /engineer_binaryllm.



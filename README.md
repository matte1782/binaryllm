<![CDATA[<div align="center">

# 🔢 BinaryLLM

**Towards 1-Bit Latent Spaces for Efficient Large Language Models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 157 Passing](https://img.shields.io/badge/tests-157%20passing-brightgreen.svg)](#test-suite)
[![Phase: 1 Complete](https://img.shields.io/badge/phase-1%20complete-success.svg)](#phase-1-overview)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*A research-first framework exploring whether binary (1-bit) representations can serve as a viable computational substrate for LLM embeddings while preserving semantic structure.*

[Overview](#overview) •
[Quick Start](#quick-start) •
[Architecture](#architecture) •
[Usage](#usage) •
[Results](#phase-1-results) •
[Roadmap](#roadmap) •
[Citation](#citation)

</div>

---

## Overview

**BinaryLLM** investigates the fundamental question: *Can we compress high-dimensional float embeddings to 1-bit binary codes while preserving their semantic neighborhoods?*

This repository implements **Phase 1** of the BinaryLLM research program—a deterministic binary embedding pipeline that:

- 🎯 **Projects** float embeddings to binary codes via Gaussian random hyperplanes
- 📊 **Evaluates** similarity preservation (cosine ↔ Hamming correlation)
- 🔍 **Measures** retrieval quality (top-k overlap, nDCG, recall)
- 🏷️ **Tests** classification degradation with centroid classifiers
- ✅ **Guarantees** full determinism and reproducibility

### Why Binary Embeddings?

| Aspect | Float32 | Binary (1-bit) | Improvement |
|--------|---------|----------------|-------------|
| Storage per dim | 32 bits | 1 bit | **32× smaller** |
| Similarity compute | FLOPs (dot product) | XOR + popcount | **~10× faster** |
| Memory bandwidth | High | Minimal | Significant for KV-cache |

### Phase 1 Goals

1. **Validate Hypothesis H1**: Binary embeddings preserve nearest-neighbor structure with sufficient code length
2. **Establish deterministic baselines**: Reproducible metrics for future phases
3. **Freeze contracts**: Stable APIs, schemas, and golden tests for Phase 2+

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- NumPy ≥ 1.24
- PyYAML ≥ 6.0
- SciPy ≥ 1.10 (for metrics)

### Installation

```bash
# Clone the repository
git clone https://github.com/matte1782/binaryllm.git
cd binaryllm

# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Your First Experiment

```bash
# Run the golden synthetic experiment
python -m src.experiments.runners.phase1_binary_embeddings \
    --config tests/data/phase1_golden/config_phase1_synthetic_v1.yaml
```

### Run Tests

```bash
# Run all 157 tests
pytest -q

# Run with coverage
pytest --cov=src --cov-report=term-missing
```

---

## Architecture

```
BinaryLLM Phase 1 Architecture
══════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│                     Precomputed Float Embeddings                │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Phase 1 Binary Embedding Engine                     │
│  ┌─────────┐  ┌─────────┐  ┌────────────┐  ┌─────────────────┐  │
│  │ Runner  │→ │ Façade  │→ │Quantization│→ │   Evaluation    │  │
│  │ (I/O)   │  │(Pipeline)│  │(Binarize)  │  │(Sim/Ret/Class) │  │
│  └─────────┘  └─────────┘  └────────────┘  └─────────────────┘  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                   ┌────────────┴────────────┐
                   │                         │
                   ▼                         ▼
          Result Dict (RAM)          Structured Logs (JSON)
```

### Module Structure

```
src/
├── core/                    # Dataset catalog & embedding containers
│   ├── dataset_catalog.py   # Registry for datasets and encoders
│   ├── datasets.py          # Dataset wrappers with validation
│   └── embeddings.py        # Float/Binary embedding containers
│
├── quantization/            # Binary transformation pipeline
│   ├── binarization.py      # Random projection + sign binarization
│   └── packing.py           # Bit packing to uint64 (LSB-first)
│
├── eval/                    # Evaluation metrics
│   ├── similarity.py        # Cosine/Hamming + Spearman correlation
│   ├── retrieval.py         # Top-k overlap, nDCG, recall
│   └── classification.py    # Centroid classifier + degradation
│
├── experiments/
│   └── runners/
│       └── phase1_binary_embeddings.py  # Main experiment runner
│
├── variants/
│   └── binary_embedding_engine.py       # Core façade (BinaryEmbeddingEngine)
│
└── utils/
    ├── config.py            # YAML/JSON config loader + validation
    ├── logging.py           # Structured JSON logging (v2 schema)
    ├── seed.py              # Global seed management
    └── io.py                # File I/O helpers
```

### Key Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Separation of Concerns** | Runner handles I/O; Façade handles math; no cross-contamination |
| **Determinism** | Same seed + inputs = identical outputs, always |
| **Explicit Errors** | Structured error pipeline with named stages |
| **Schema Stability** | Result/Log Schema v2 is frozen and tested |

---

## Usage

### Configuration

Create a YAML config file:

```yaml
# my_experiment.yaml
runner: phase1_binary_embeddings
encoder_name: my_encoder
dataset_name: my_dataset
code_bits: 64                    # 32, 64, 128, or 256
projection_type: gaussian        # Only "gaussian" in Phase 1
seed: 42
tasks:
  - similarity
  - retrieval
  - classification
embedding_files:
  - path/to/embeddings.npy
classification_labels: path/to/labels.npy
output_dir: runs/my_experiment/
```

### Programmatic API

```python
from src.experiments.runners.phase1_binary_embeddings import run_phase1_experiment

# Run experiment
result = run_phase1_experiment("path/to/config.yaml")

# Check status
if result["status"] == "success":
    print(f"Cosine-Hamming Spearman: {result['similarity_metrics']['cosine_hamming_spearman']:.4f}")
    print(f"Top-k Overlap: {result['retrieval_metrics']['topk_overlap']['k=3']:.4f}")
    print(f"Accuracy Delta: {result['classification_metrics']['accuracy_delta']:.4f}")
else:
    print(f"Error at stage '{result['error']['stage']}': {result['error']['message']}")
```

### Using the Engine Directly

```python
import numpy as np
from src.core.dataset_catalog import get_dataset_spec, get_encoder_spec
from src.variants.binary_embedding_engine import BinaryEmbeddingEngine

# Load specs
encoder = get_encoder_spec("synthetic_encoder_4d")
dataset = get_dataset_spec("phase1_synthetic_toy")

# Create engine
engine = BinaryEmbeddingEngine(
    encoder_spec=encoder,
    dataset_spec=dataset,
    code_bits=64,
    projection_type="gaussian",
    seed=42,
    normalize=True,
)

# Run pipeline
embeddings = np.random.randn(100, 4).astype(np.float32)
labels = np.random.randint(0, 3, size=100)

result = engine.run(
    embeddings,
    metrics=["similarity", "retrieval", "classification"],
    retrieval_k=5,
    classification_labels=labels,
    return_full_code_bits=True,
)

# Access binary codes
binary_codes = result["binary_codes"]
print(f"PM1 codes shape: {binary_codes['pm1'].shape}")     # (100, 64)
print(f"Packed codes shape: {binary_codes['packed'].shape}") # (100, 1) for 64-bit
```

---

## Phase 1 Results

### Golden Synthetic Dataset Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Cosine-Hamming Spearman | ~0.94 | Strong rank correlation preserved |
| Top-3 Overlap | ~0.91 | High neighbor consistency |
| nDCG@3 | ~0.95 | Excellent ranking quality |
| Float Accuracy | 1.00 | Perfect on synthetic data |
| Binary Accuracy | ~0.92 | Expected degradation |
| Accuracy Delta | -0.08 | Confirms H1 degradation contract |

### Test Suite

- **157 tests** covering all modules
- **100% determinism** verified via golden regression
- **Cross-platform** tested (Linux, macOS, Windows)

---

## Environment & Requirements

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.11+ |
| RAM | 4 GB | 8 GB+ |
| Disk | 100 MB | 500 MB |

### Dependencies

```
numpy>=1.24.0
scipy>=1.10.0
pyyaml>=6.0
torch>=2.0.0  # Optional, for GPU metadata
pytest>=7.0.0  # Development only
```

See `requirements.txt` for pinned versions.

---

## Roadmap

### ✅ Phase 1: Binary Embedding Engine (Complete)

- [x] Gaussian random projection
- [x] Sign binarization with {-1,+1} → {0,1} mapping
- [x] LSB-first bit packing to uint64
- [x] Similarity, retrieval, and classification metrics
- [x] Deterministic pipeline with golden tests
- [x] Structured logging (Schema v2)

### 🔜 Phase 2: Binary KV-Cache (Planned)

- [ ] Binary attention key/value representations
- [ ] XNOR-popcount attention kernels
- [ ] Memory bandwidth benchmarks
- [ ] Long-context scaling experiments

### 🔮 Phase 3: Binary Transformer Components (Research)

- [ ] 1-bit projection layers
- [ ] Binary MLP blocks
- [ ] Hybrid binary/low-bit architectures

### 🚀 Phase 4: Full BinaryLLM Inference (Vision)

- [ ] End-to-end binary inference pipeline
- [ ] Production-ready CUDA kernels
- [ ] Integration with existing LLM frameworks

---

## Project Structure

```
binaryllm/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── CHANGELOG.md              # Version history
├── CONTRIBUTORS.md           # Project contributors
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Modern Python packaging
├── .gitignore                # Git ignore rules
│
├── src/                      # Source code
│   ├── core/                 # Core abstractions
│   ├── quantization/         # Binarization pipeline
│   ├── eval/                 # Evaluation metrics
│   ├── experiments/          # Experiment runners
│   ├── variants/             # Engine implementations
│   └── utils/                # Utilities
│
├── tests/                    # Test suite (157 tests)
│   ├── data/
│   │   └── phase1_golden/    # Golden regression artifacts
│   └── test_*.py             # Test modules
│
├── scripts/                  # Helper scripts
│   └── generate_phase1_golden.py
│
└── docs/                     # Documentation
    ├── architecture/         # Architecture docs
    ├── artifacts/            # Phase artifacts
    └── papers_arxiv/         # Research papers
```

---

## Author

**Matteo Panzeri**  
*Artificial Intelligence Bachelor Student*  
*University of Pavia, Italy*

📧 **Contact:**
- Personal: [matteo1782@gmail.com](mailto:matteo1782@gmail.com)
- Academic: [matteo.panzeri01@universitadipavia.it](mailto:matteo.panzeri01@universitadipavia.it)

---

## Citation

If you use BinaryLLM in your research, please cite:

```bibtex
@software{panzeri2025binaryllm,
  author = {Panzeri, Matteo},
  title = {BinaryLLM: Towards 1-Bit Latent Spaces for Efficient Large Language Models},
  year = {2025},
  url = {https://github.com/matte1782/binaryllm},
  note = {Phase 1: Binary Embedding Engine}
}
```

---

## Contributing

Contributions are welcome! Please read the following before contributing:

1. **Phase 1 is frozen** — No changes to core logic without explicit approval
2. **Run all tests** — `pytest -q` must pass (157 tests)
3. **Maintain determinism** — Same seed must produce identical outputs
4. **Follow conventions** — See existing code for style guidelines

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for contributor information.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**BinaryLLM** — Compressing knowledge, preserving meaning.

*Built with rigor. Designed for impact.*

⭐ Star this repo if you find it useful!

</div>
]]>
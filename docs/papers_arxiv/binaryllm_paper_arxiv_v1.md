# BinaryLLM Research Report v2 — Validated Edition

# Authors

Matteo Panzeri (with assistance from Cursor GPT‑5.1 agent workflow)

# Abstract

Large Language Models (LLMs) achieve strong performance by operating in high‑dimensional floating‑point latent spaces for token embeddings, attention, and MLP activations, but this comes at substantial cost in memory footprint, memory bandwidth, and energy—especially at large scales and long context lengths. Binary Neural Networks (BNNs), which constrain weights and often activations to \(\{-1,+1\}\), replace most floating‑point multiplications with bitwise XNOR and population count, offering substantial reductions in model size and compute cost. This paper synthesizes foundational BNN and low‑bit LLM literature, develops mathematical foundations for binary embeddings and rate–distortion trade‑offs, provides symbolic system‑level models for A100/H100/Blackwell GPUs, and specifies a concrete experimental protocol and hypotheses for BinaryLLM. The aim is to establish a rigorous, validated basis for exploring binary latent spaces and binary embeddings in LLMs under realistic GPU constraints.

# 1. Introduction

LLMs rely on dense floating‑point representations for embeddings, attention, and MLPs, typically using FP16 or FP32. As model and context sizes increase, the dominant costs shift from arithmetic to moving and storing large tensors: parameters, key–value (KV) caches, and intermediate activations. This raises practical limits on model deployment, especially for long‑context inference and resource‑constrained environments.

BNNs constrain weights and, in many cases, activations to 1‑bit values \(\{-1,+1\}\), enabling matrix multiplications to be implemented as highly parallel bitwise operations (XNOR+popcount) with substantial memory savings. Foundational BNN work in vision demonstrates that well‑designed binarized models can approach full‑precision accuracy on challenging tasks while achieving large improvements in theoretical efficiency [1]–[4].

Recent work extends extreme quantization to Transformers and LLMs, including fully binarized BERT models and 1‑bit Transformer architectures for autoregressive LLMs [8]–[10]. At the same time, binary embedding theory and empirical work on semantic hashing show that Hamming distance between sign‑quantized projections can approximate angular distance, giving a principled foundation for binary latent spaces and bit‑based similarity [5]–[7].

This paper consolidates these strands into a single validated BinaryLLM foundation: it reviews relevant literature, formalizes key mathematical relationships, outlines system‑level implications for modern NVIDIA GPUs, and specifies a rigorous experimental protocol and hypothesis set for evaluating binary embeddings, 1‑bit quantization in Transformer blocks, and binary KV‑caches.

# 2. Background

## 2.1 Quantization and Binarization

Quantization reduces numerical precision to lower the cost of storage and computation. In LLMs, established methods include 8‑bit and 4‑bit quantization for weights and activations, often combined with techniques to handle outlier channels and preserve accuracy. QLoRA shows that 4‑bit quantization with suitable scaling and low‑rank adapters can preserve near‑full‑precision performance for large LLaMA models [12]. LLM.int8 and SmoothQuant demonstrate that careful handling of activation statistics and channel‑wise scaling allow accurate INT8 and INT4 inference and post‑training quantization [11], [13].

BNNs represent an extreme point on this spectrum, with weights and activations constrained to \(\pm 1\), enabling dot‑products to be replaced by bitwise XNOR and popcount operations. BinaryConnect and Binarized Neural Networks established training procedures where binarization is applied in forward/backward passes while maintaining real‑valued master weights for updates [1], [2]. XNOR‑Net and Bi‑Real Net introduced architectural and training improvements (e.g., scaling factors, real‑valued shortcuts) that significantly narrow the accuracy gap to full‑precision networks on vision tasks [3], [4].

For LLMs, 1‑bit quantization techniques such as BitNet adapt similar ideas to Transformer architectures, showing that 1‑bit weight matrices can be viable for large‑scale language modeling under appropriate training regimes [10].

## 2.2 Binary Embedding Theory

Binary embeddings map high‑dimensional real vectors to binary codes such that Hamming distance approximates a chosen similarity measure, typically angular distance or cosine similarity. Semantic Hashing demonstrated empirically that short binary codes can preserve semantic neighborhoods for document retrieval, enabling very fast search in Hamming space [5].

Binary embedding theory formalizes this: random sign projections applied to vectors in \(\mathbb{R}^d\) produce binary codes whose Hamming distance concentrates around a quantity proportional to the angle between vectors. Jacques et al. show that such embeddings can achieve nearly optimal dimensionality in the sense of the Johnson–Lindenstrauss lemma, preserving pairwise distances up to small distortions with a number of bits that is near‑minimal [6]. Norouzi et al. provide efficient algorithms for exact nearest‑neighbor search in high‑dimensional Hamming spaces, demonstrating that binary embeddings are practical for large‑scale retrieval systems [7].

These results motivate the use of binary embeddings and bit‑latents in LLMs, provided that learned projections approximate the properties of random or well‑spread hyperplanes in the latent space.

## 2.3 Prior Low-bit LLM Work

Several lines of work apply low‑bit and extreme quantization to Transformers and LLMs:

- BiBERT presents a fully binarized BERT‑base model with binarized weights and activations plus additional mechanisms (Bi‑Attention and distillation). It achieves competitive performance on GLUE and other NLP tasks while significantly reducing FLOPs and model size, though a measurable accuracy gap remains [8].
- BinaryBERT explores 1‑bit and low‑bit quantization of BERT with careful scaling and distillation, showing that substantial compression can be obtained with modest accuracy loss on multiple NLP benchmarks [9].
- BitNet introduces a 1‑bit Transformer architecture for decoder‑only LLMs using BitLinear layers, reporting competitive language modeling performance with reduced memory and energy compared to FP16 [10].
- QLoRA, LLM.int8, and SmoothQuant provide strong baselines for 4‑/8‑bit LLMs, combining quantization with specialized scaling and handling of outlier channels to preserve performance [11]–[13].

These works collectively indicate that:

1. Transformer models tolerate aggressive quantization when combined with appropriate architectural and training modifications.
2. Mixed‑precision regimes (e.g., keeping embeddings or output heads at higher precision) are often necessary for best quality.
3. There is still limited systematic work on fully binarizing large decoder‑only LLMs, especially with binary KV‑caches and binary internal latents across long context lengths.

# 3. Binarization Theory

This section summarizes the conceptual aspects of binarization and its implications for LLMs.

BNNs apply a binarization operator to weights and, often, activations. In its simplest form, binarization maps a real number to one of two values, typically \(\pm 1\). When applied to vectors and matrices and combined with appropriate encoding (e.g., mapping \(\{+1,-1\}\) to \(\{1,0\}\)), dot‑products can be computed via XNOR and popcount operations.

BinaryConnect and subsequent work show that it is beneficial to retain real‑valued master weights and apply binarization only during forward and backward passes [1], [2]. Bi‑Real Net demonstrates that real‑valued shortcuts can significantly mitigate the representational loss from binarization [4]. These ideas are central to any attempt to binarize Transformer blocks in LLMs.

Binary embedding theory, as discussed in Section 2.2, connects sign‑based projections to angular distance, suggesting that binary embeddings can approximate real‑valued similarity structures. In the BinaryLLM context, this motivates:

- Binary token or document embeddings for retrieval and similarity tasks.
- Binary latent representations (bit‑latents) for internal Transformer states, especially where similarity structure is critical (e.g., attention, retrieval‑augmented components).

Rate–distortion theory provides an information‑theoretic view of how much distortion is inevitable when representing continuous latent variables with a fixed number of bits, setting expectations for 1‑bit versus multi‑bit representations.

# 4. Mathematical Derivations

This section provides the key derivations used in the BinaryLLM foundation.

## 4.1 Binarization Operator

Let \(x \in \mathbb{R}\) be a scalar and \(w \in \mathbb{R}^d\) a weight vector. The sign binarization operator is

$$
\mathcal{B}(x) = \mathrm{sign}(x) =
\begin{cases}
1, & x \ge 0, \\
-1, & x < 0.
\end{cases}
$$

For a vector \(w\), binarization is applied elementwise. To reduce approximation error, a scaled sign binarization is often used:

$$
\hat{w} = \alpha \,\mathrm{sign}(w),
$$

where \(\alpha > 0\) is chosen (per layer or per channel) to minimize mean‑squared error

$$
\alpha^\star = \arg\min_\alpha \mathbb{E}\big[\lVert w - \alpha\,\mathrm{sign}(w)\rVert^2\big].
$$

Under zero‑mean, i.i.d. coordinates, \(\alpha^\star\) can be expressed in terms of \(\mathbb{E}[|w_i|]\), and in practice is estimated from sample statistics over weights or activations.

## 4.2 Straight-Through Estimator

The binarization function \(\mathcal{B}\) is non‑differentiable and has zero derivative almost everywhere. To enable gradient‑based training, BNNs use a Straight‑Through Estimator (STE) that approximates the derivative of \(\mathcal{B}\) with respect to its input.

For a scalar \(x\) with binarized output \(y = \mathcal{B}(x)\), a common STE is

$$
\frac{\partial y}{\partial x} \approx
\begin{cases}
1, & |x| \le 1, \\
0, & \text{otherwise},
\end{cases}
$$

or in a simplified version, \(1\) on a chosen interval and \(0\) outside. This approximation allows gradients to flow through binarization in backpropagation while preserving the binarized behavior in the forward pass.

## 4.3 Random Hyperplane Theorem

Consider two non‑zero vectors \(u, v \in \mathbb{R}^d\) and a random projection \(r \sim \mathcal{N}(0, I_d)\). Define scalar random variables

$$
X = r^\top u, \quad Y = r^\top v.
$$

Under the Gaussian assumption, \((X, Y)\) is jointly normal with correlation

$$
\rho = \frac{\mathbb{E}[XY]}{\sqrt{\mathbb{E}[X^2]\mathbb{E}[Y^2]}} = \frac{u^\top v}{\|u\| \, \|v\|} = \cos \theta,
$$

where \(\theta\) is the angle between \(u\) and \(v\). A classical result used in locality‑sensitive hashing and binary embedding theory states that the probability a random hyperplane (defined by \(r\)) separates the two points is

$$
P[\mathrm{sign}(X) \neq \mathrm{sign}(Y)] = \frac{\theta}{\pi}.
$$

This forms the core of using random hyperplanes to approximate angular distance.

## 4.4 Hamming vs Cosine Relationship

Take \(m\) independent random projections \(r_1,\dots,r_m \sim \mathcal{N}(0, I_d)\) and define binary codes

$$
b(u) = \big(\mathrm{sign}(r_1^\top u),\dots,\mathrm{sign}(r_m^\top u)\big) \in \{-1,+1\}^m,
$$
$$
b(v) = \big(\mathrm{sign}(r_1^\top v),\dots,\mathrm{sign}(r_m^\top v)\big) \in \{-1,+1\}^m.
$$

Let \(H(b(u), b(v))\) denote the Hamming distance between these codes (the number of coordinates where they differ). Each coordinate differs with probability \(\theta/\pi\), so

$$
\mathbb{E}[H(b(u), b(v))] = m \, P[\mathrm{sign}(X) \neq \mathrm{sign}(Y)] = m \frac{\theta}{\pi}.
$$

The normalized Hamming distance thus satisfies

$$
\mathbb{E}\left[\frac{H(b(u), b(v))}{m}\right] = \frac{\theta}{\pi},
$$

which is a monotonic function of \(\theta\) and therefore of the cosine similarity \(\cos\theta\). Concentration inequalities (e.g., Hoeffding’s inequality) imply that \(\frac{H}{m}\) concentrates around \(\theta/\pi\) as \(m\) grows, preserving relative angular distances with high probability. This justifies using Hamming distance between binary embeddings as a proxy for cosine similarity between their real‑valued counterparts, particularly when projections approximate a set of reasonably distributed hyperplanes.

## 4.5 Rate–Distortion Argument

Let \(Z\) be a memoryless Gaussian source with variance \(\sigma^2\), i.e., \(Z \sim \mathcal{N}(0,\sigma^2)\). Consider quantizing \(Z\) at an average rate \(R\) bits per scalar in the mean‑squared error sense. The Shannon rate–distortion function for such a source is

$$
R(D) = \frac{1}{2}\log_2\left(\frac{\sigma^2}{D}\right), \quad 0 < D \le \sigma^2,
$$

where \(D\) is the minimum achievable distortion (MSE) at rate \(R\). Solving for \(D\) yields

$$
D = \sigma^2 2^{-2R}.
$$

At \(R = 1\) bit per scalar (corresponding to a 1‑bit quantizer under these assumptions), the theoretical lower bound on distortion is

$$
D_{\min} = \frac{\sigma^2}{4}.
$$

For BinaryLLM, this implies:

1. 1‑bit bit‑latents cannot be arbitrarily close to full‑precision latents; there is irreducible distortion on the order of a constant fraction of the latent variance.
2. Moving from 1 bit to a small number of bits (e.g., 2–4) per scalar reduces this lower bound sharply, making hybrid 1–2‑bit designs potentially attractive.
3. Distortion in intermediate representations will propagate through the network; training must make the model robust to this quantization noise and allocate bits where they yield the greatest reduction in effective distortion relative to task requirements.

# 5. System & Hardware Analysis

This section summarizes symbolic models and qualitative considerations for BinaryLLM deployment on NVIDIA A100, H100, and Blackwell GPUs.

## 5.1 Memory Scaling

Let \(N\) be the total number of parameters, \(b\) the bit‑width per parameter (\(b \in \{1,2,4,8,16\}\)), and \(C_{\text{mem}}\) the device memory capacity in bytes. The total parameter storage is

$$
M_{\text{params}}(N, b) = N \cdot \frac{b}{8} \quad \text{[bytes]}.
$$

The maximum number of parameters that fit in memory (ignoring overheads) is

$$
N_{\max}(b) = \frac{8\,C_{\text{mem}}}{b}.
$$

For two bit‑widths \(b_1\) and \(b_2\), the relative parameter capacity is

$$
\frac{N_{\max}(b_1)}{N_{\max}(b_2)} = \frac{b_2}{b_1}.
$$

Thus, relative to FP16 (\(b=16\)), full binarization (\(b=1\)) increases the theoretical parameter capacity by a factor of 16, with analogous scaling for KV‑caches and activations, modulated by sequence length, number of layers, and the extent of binarization.

## 5.2 FLOPs and Operation Throughput

Consider a standard Transformer with model dimension \(d\), sequence length \(L\), and \(n_L\) layers. Ignoring constant factors, the total operation count for a forward pass scales as

$$
\mathrm{Ops}_{\text{FP}} \propto n_L \cdot L \cdot d^2.
$$

Switching from FP16 to 1‑bit quantization does not change this asymptotic operation count, but it changes:

- The operation type: from floating‑point multiply‑accumulate to bitwise XNOR+popcount plus scaling.
- The achievable throughput on the hardware, which depends on how well binary kernels utilize GPU resources.

Define effective peak throughputs:

- \(T_{\text{fp16}}^{\text{GPU}}\): FP16 tensor‑core operations (per second).
- \(T_{\text{int8}}^{\text{GPU}}\): INT8 tensor‑core operations (per second).
- \(T_{\text{bin}}^{\text{GPU}}\): effective throughput of optimized binary matmul kernels (per second).

The ideal execution time for \(\mathrm{Ops}_{\text{FP}}\) operations at each precision is

$$
t_{\text{fp16}} \approx \frac{\mathrm{Ops}_{\text{FP}}}{T_{\text{fp16}}^{\text{GPU}}}, \qquad
t_{\text{bin}} \approx \frac{\mathrm{Ops}_{\text{FP}}}{T_{\text{bin}}^{\text{GPU}}}.
$$

The ideal speedup of binary over FP16 is then

$$
S_{\text{bin/fp16}}^{\text{ideal}} = \frac{t_{\text{fp16}}}{t_{\text{bin}}} = \frac{T_{\text{bin}}^{\text{GPU}}}{T_{\text{fp16}}^{\text{GPU}}}.
$$

In practice, \(T_{\text{bin}}^{\text{GPU}}\) must be determined via microbenchmarks on A100, H100, and Blackwell GPUs using carefully implemented bit‑packed kernels.

## 5.3 Energy Model

Let \(P_{\text{GPU}}\) be the average power draw during inference, \(T_{\text{eff}}\) the effective realized throughput (operations per second), and \(\mathrm{Ops}_{\text{FP}}\) the total operation count for a forward pass. The energy per operation is

$$
E_{\text{op}} = \frac{P_{\text{GPU}}}{T_{\text{eff}}} \quad \text{[Joules/op]},
$$

and the total energy per forward pass is

$$
E_{\text{total}} = E_{\text{op}} \cdot \mathrm{Ops}_{\text{FP}}.
$$

Assuming similar power envelopes for FP16 and binary configurations (an empirical question), and denoting effective throughputs by \(T_{\text{eff,fp16}}\) and \(T_{\text{eff,bin}}\), the energy ratio is

$$
\frac{E_{\text{total,bin}}}{E_{\text{total,fp16}}} \approx \frac{T_{\text{eff,fp16}}}{T_{\text{eff,bin}}}.
$$

Because data movement often dominates energy in large models, reductions in data volume via lower bit‑widths are important. Binary latents reduce bytes per scalar by a factor proportional to \(1/b\). For KV‑caches and long‑context inference, such reductions can substantially lower both memory and interconnect energy, provided packing/unpacking overheads are controlled and binary kernels achieve sufficient compute efficiency.

## 5.4 GPU Kernel and Hardware Considerations

On NVIDIA A100 and H100 GPUs, tensor cores support FP16/BF16/TF32/FP8 and INT8 formats; there is no publicly documented 1‑bit matrix multiply mode. Binary matmuls must therefore be implemented via CUDA cores using integer instructions, bit‑packing, XNOR, and popcount, with per‑tile scaling and re‑centering.

Binary matrix multiplication kernels can adopt GEMM‑like tiling, staging packed words in shared memory and operating on blocks. Challenges include:

- Efficient bit‑packing/unpacking.
- Maintaining memory coalescing with bit‑packed layouts.
- Achieving high occupancy while managing register and shared memory usage.

Even with bit‑packing, KV‑cache and activation traffic can keep workloads memory‑bound. Binary representations reduce per‑element bytes by factors up to 16 vs FP16, but if compute throughput \(T_{\text{bin}}^{\text{GPU}}\) is significantly below tensor‑core \(T_{\text{fp16}}^{\text{GPU}}\), overall speedups may be modest. Measuring \(T_{\text{bin}}^{\text{GPU}}\) and comparing tokens/s against well‑tuned 4‑bit and 8‑bit tensor‑core baselines is therefore necessary.

FeRAM‑based BNN accelerators indicate that binary‑optimized hardware can deliver very high efficiency (TOPS/W) for smaller networks [15]. While these designs are not specific to LLMs, they suggest that hardware–software co‑design for binary LLM workloads could further enhance energy efficiency beyond what is possible on general‑purpose GPUs.

*Figure 1: Overview of the BinaryLLM pipeline (placeholder).*

# 6. Literature Review

The literature relevant to BinaryLLM falls into several categories:

- Foundational BNN methods that define training procedures and architectural patterns for binarized networks [1]–[4].
- Binary embedding theory and Hamming‑space retrieval work that provide a rigorous and algorithmic foundation for binary codes [5]–[7].
- NLP‑specific binarization and low‑bit LLM work, including fully binarized BERT‑style models and 1‑bit Transformer architectures [8]–[10].
- General low‑bit quantization methods for LLMs, such as QLoRA, LLM.int8, and SmoothQuant, which provide strong 4‑bit and 8‑bit baselines and techniques for handling outlier channels [11]–[13].
- Hardware work on binary‑optimized accelerators, such as FeRAM‑based binary‑weighted networks, that demonstrate how hardware specialization can dramatically improve energy efficiency [15].

The detailed literature summary, including key findings, limitations, and relevance to BinaryLLM, is provided in Section 6 and implicitly in the bibliography.

# 7. Hypotheses

The BinaryLLM project is guided by the following testable hypotheses.

**H1 – Binary embeddings preserve retrieval neighborhoods at sufficient code length.**  
For a fixed LLM, learned binary embeddings of dimension \(m\) for document/text representations will, on BEIR datasets:

- Achieve \(\text{Overlap}_k \ge 0.95\) between Hamming‑based and cosine‑based top‑k neighbor sets for \(k \in \{10, 50\}\) once \(m\) exceeds a chosen threshold, and
- Maintain nDCG@k within a pre‑specified margin relative to full‑precision embeddings.

**H2 – Binary KV‑caches can maintain perplexity within bounded degradation.**  
For a decoder‑only LLM with 1‑bit KV‑caches (with per‑channel scaling) and 4‑bit or higher internal computations:

- WikiText‑103 validation perplexity remains within 5% relative of a strong 4‑bit baseline across targeted context lengths, and
- GLUE/BEIR performance remains within a fixed absolute margin (e.g., 1–2 points) of the 4‑bit baseline.

**H3 – Binary attention/MLP blocks plus distillation can match 4‑bit LLMs on NLU.**  
A BitNet‑style 1‑bit LLM with binarized attention and MLP layers, trained with distillation from a full‑precision teacher, will:

- Achieve GLUE metrics (accuracy/F1/MCC) within 1 point absolute of a 4‑bit QLoRA baseline, while
- Using fewer bits per parameter and lower end‑to‑end memory.

**H4 – Binary token embeddings are more performance‑sensitive than internal latents.**  
Under a fixed total bit budget:

- Binarizing only token embeddings (Variant A) will cause larger degradation in GLUE/BEIR performance relative to full precision than binarizing selected internal blocks while keeping embeddings at higher precision (Variant B).

**H5 – Binary LLMs can realize net energy gains at long context.**  
On A100, H100, and Blackwell GPUs, at sufficiently long context lengths:

- BinaryLLM variants with binary KV‑caches and substantial binary blocks achieve strictly lower Joules/token and equal or higher tokens/s than well‑optimized 4‑bit baselines.

**H6 – Task sensitivity to binarization is empirical.**  
No prior assumption is made about which task families (retrieval, classification, language modeling, reasoning) are more robust to binarization. Instead, this is treated as an open question to be answered by systematically comparing BinaryLLM and 4‑bit baselines across GLUE, BEIR, language modeling, and selected reasoning benchmarks.

# 8. Experimental Protocol

The experimental protocol binds each hypothesis to concrete evaluations.

## 8.1 Models and Variants

- Base full‑precision LLM: decoder‑only Transformer with parameter count on the order of \(1\)–\(7\)B parameters, trained or finetuned in FP16 (or similar).
- Quantized baselines:
  - 4‑bit LLM using QLoRA‑style quantization and adapters.
  - 8‑bit LLM using LLM.int8‑style matrices and outlier channel handling.
- BinaryLLM variants:
  - Binary‑Embeddings model: binary token embeddings and binary pooled features, with internal layers at 4‑bit or higher.
  - Binary‑Blocks model: selected attention and/or MLP blocks binarized (weights and selected activations), rest at higher precision.
  - Binary‑KV model: KV‑caches stored at 1‑bit per scalar (with per‑channel scaling), with internal computations at 4‑bit or higher.
  - Fully Binary model: smaller‑scale Transformer encoder (e.g., BERT‑style), closely replicating BiBERT and related designs, possibly with BinaryLLM‑specific modifications.

## 8.2 Datasets

- Language modeling / perplexity:
  - WikiText‑103 as a standard benchmark.
  - Optionally, a well‑defined subset of a larger web corpus (e.g., a C4‑like dataset) for auxiliary evaluation.
- NLP classification and NLU:
  - GLUE benchmark tasks (e.g., MNLI, QQP, QNLI, SST‑2).
- Retrieval and similarity:
  - BEIR benchmark datasets (e.g., MS MARCO, Natural Questions, SciFact).
- Reasoning / QA (task sensitivity):
  - A clearly specified subset of reasoning or knowledge‑heavy benchmarks (e.g., selected MMLU tasks).

## 8.3 Metrics

- Language modeling:
  - Perplexity (PPL) and negative log‑likelihood on held‑out sets.
- Classification / NLU:
  - Accuracy, F1 score, and Matthews correlation coefficient (MCC) where appropriate.
- Retrieval / similarity:
  - nDCG@k and Recall@k on BEIR.
  - Top‑k neighbor overlap between binary and full‑precision embeddings:

    $$
    \text{Overlap}_k = \frac{1}{|Q|}\sum_{q \in Q} \frac{|N_k^{\text{binary}}(q) \cap N_k^{\text{float}}(q)|}{k},
    $$

    where \(Q\) is a set of queries and \(N_k^{\cdot}(q)\) is the top‑k nearest neighbors under Hamming (binary) or cosine (float) distance.

- Energy / systems:
  - Tokens per second (throughput) under a controlled hardware and software setup.
  - Joules per token for long‑context inference, measured via GPU power telemetry integrated over time.

## 8.4 Statistical Significance

- Number of seeds:
  - At least three independent runs per configuration (model plus quantization setting).
- Reporting:
  - Report mean and standard deviation for each metric across seeds.
  - For comparisons (e.g., BinaryLLM vs 4‑bit baseline), perform appropriate two‑sided significance tests (paired t‑test or non‑parametric alternative), with p‑value thresholds (e.g., \(p < 0.05\)) and corrections for multiple comparisons where necessary.
- Pass/fail criteria:
  - Each hypothesis is tied to explicit numeric thresholds (e.g., relative PPL deltas, neighbor overlaps, absolute metric differences). A hypothesis is considered supported or rejected based on whether observed metrics and statistical tests fall within or outside these thresholds.

# 9. Expected Contributions

The BinaryLLM effort, as defined by the validated report, is expected to contribute:

1. A consolidated theoretical foundation for binary embeddings and 1‑bit latent spaces in LLMs, grounded in established BNN and binary embedding theory.
2. Symbolic system‑level models for memory, compute, and energy scaling of binary and low‑bit LLMs on A100/H100/Blackwell GPUs.
3. A rigorous experimental protocol and hypothesis set that link theoretical expectations to measurable metrics on standard language modeling, NLU, and retrieval benchmarks.
4. A structured roadmap for exploring where binarization is most and least tolerable in LLM architectures (embeddings vs internal blocks vs KV‑caches).

These contributions are contingent on successful execution of the proposed experiments and careful interpretation of their results.

# 10. Limitations

The current work intentionally remains at the level of theory, symbolic modeling, and experimental design. Limitations include:

- No empirical results are yet reported for BinaryLLM variants; all performance and energy advantages must be established by future experiments.
- Hardware‑level modeling is symbolic and depends on empirical measurement of effective throughputs and energy usage for binary kernels versus tensor‑core baselines.
- The rate–distortion analysis uses Gaussian assumptions and scalar quantization; real LLM latents may deviate from these assumptions.
- Task sensitivity to binarization is explicitly treated as an open question; conclusions will depend on observed behavior across tasks and architectures.

These limitations are acknowledged and are to be addressed in subsequent experimental phases.

# 11. Conclusion

This paper provides a validated, NVIDIA‑grade foundation for the BinaryLLM project. It consolidates key BNN, binary embedding, and low‑bit LLM literature; formulates essential mathematical relationships (binarization operators, Hamming vs cosine distance, rate–distortion bounds); and outlines symbolic models for memory, compute, and energy on A100, H100, and Blackwell GPUs. It also specifies a precise experimental protocol and hypothesis set, tying theoretical considerations directly to measurable metrics on standard datasets.

The next phase of BinaryLLM work is to implement and evaluate the outlined BinaryLLM variants, rigorously measure their quality and efficiency relative to strong 4‑bit/8‑bit baselines, and use the resulting evidence to refine or reject the proposed hypotheses. The final goal is to distill these findings into a BinaryLLM protocol that prescribes bit allocations, training procedures, and hardware mappings for practical deployment of energy‑efficient, binarized LLMs.

# References

[1] M. Courbariaux, Y. Bengio, and J.‑P. David, “BinaryConnect: Training Deep Neural Networks with binary weights during propagations,” *Advances in Neural Information Processing Systems*, 2015.  
[2] M. Courbariaux, I. Hubara, D. Soudry, R. El‑Yaniv, and Y. Bengio, “Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or −1,” arXiv:1602.02830, 2016.  
[3] M. Rastegari, V. Ordonez, J. Redmon, and A. Farhadi, “XNOR‑Net: ImageNet Classification Using Binary Convolutional Neural Networks,” *European Conference on Computer Vision (ECCV)*, 2016.  
[4] Z. Liu, W. Luo, B. Wu, X. Yang, W. Liu, and K.‑T. Cheng, “Bi‑Real Net: Binarizing Deep Network towards Real‑Network Performance,” *European Conference on Computer Vision (ECCV)*, 2018.  
[5] R. Salakhutdinov and G. Hinton, “Semantic Hashing,” *International Journal of Approximate Reasoning*, 2009.  
[6] L. Jacques, J. N. Laska, P. T. Boufounos, and R. G. Baraniuk, “Robust 1‑bit Compressive Sensing via Binary Stable Embeddings of Sparse Vectors,” *IEEE Transactions on Information Theory*, 2013.  
[7] M. Norouzi, A. Punjani, and D. J. Fleet, “Fast Exact Search in Hamming Space with Multi‑Index Hashing,” *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2012.  
[8] H. Qin, Y. Ding, M. Zhang, Q. Yan, A. Liu, Q. Dang, Z. Liu, and X. Liu, “BiBERT: Accurate Fully Binarized BERT,” arXiv:2203.06390, 2022.  
[9] Y. Zhang et al., “BinaryBERT: Pushing the Limit of BERT Quantization,” arXiv:2012.15701, 2020.  
[10] H. Wang, S. Ma, L. Dong, S. Huang, H. Wang, L. Ma, F. Yang, R. Wang, Y. Wu, and F. Wei, “BitNet: Scaling 1‑bit Transformers for Large Language Models,” arXiv:2310.11453, 2023.  
[11] T. Dettmers, M. Lewis, S. Shleifer, and L. Zettlemoyer, “LLM.int8(): 8‑bit Matrix Multiplication for Transformers at Scale,” *International Conference on Learning Representations (ICLR)*, 2022.  
[12] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Efficient Finetuning of Quantized LLMs,” *International Conference on Machine Learning (ICML)*, 2023.  
[13] Z. Xiao, J. Chen, S. Yan, Y. Zhang, and R. Jin, “SmoothQuant: Accurate and Efficient Post‑Training Quantization for Large Language Models,” arXiv:2211.10438, 2023.  
[14] Y. Guo, A. Zhang, Y. Zhang, and Z. Chen, “Least Squares Binary Quantization of Neural Networks,” arXiv:2001.02786, 2020.  
[15] Binary‑Weighted Neural Networks Using FeRAM Array for Low‑Power AI Computing, *Nanomaterials (MDPI)*, 2023.



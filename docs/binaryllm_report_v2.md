# BinaryLLM Research Report v2 — Validated Edition

## Abstract

Large Language Models (LLMs) achieve strong performance by operating in high‑dimensional floating‑point latent spaces for token embeddings, attention, and MLP activations, but this comes at substantial cost in memory footprint, memory bandwidth, and energy—especially at large scales and long context lengths. Binary Neural Networks (BNNs), which constrain weights and often activations to \(\{-1,+1\}\), replace most floating‑point multiplications with bitwise XNOR and population count, offering substantial reductions in model size and compute cost. This report synthesizes foundational BNN and low‑bit LLM literature, develops mathematical foundations for binary embeddings and rate–distortion trade‑offs, provides symbolic system‑level models for A100/H100/Blackwell GPUs, and specifies a concrete experimental protocol and hypotheses for BinaryLLM. The aim is to establish a rigorous, validated basis for exploring binary latent spaces and binary embeddings in LLMs under realistic GPU constraints.

---

## 1. Introduction

LLMs rely on dense floating‑point representations for embeddings, attention, and MLPs, typically using FP16 or FP32. As model and context sizes increase, the dominant costs shift from arithmetic to moving and storing large tensors: parameters, KV‑caches, and intermediate activations. This raises practical limits on model deployment, especially for long-context inference and resource‑constrained environments.

BNNs constrain weights and, in many cases, activations to 1‑bit values \(\{-1,+1\}\), enabling matrix multiplications to be implemented as highly parallel bitwise operations (XNOR+popcount) with substantial memory savings. Foundational BNN work in vision demonstrates that well‑designed binarized models can approach full‑precision accuracy on challenging tasks while achieving large improvements in theoretical efficiency.

Recent work extends extreme quantization to Transformers and LLMs, including fully binarized BERT models and 1‑bit Transformer architectures for autoregressive LLMs. At the same time, theory on binary embeddings shows that Hamming distance between sign‑quantized projections can approximate angular distance, giving a principled foundation for binary latent spaces and bit‑based similarity.

This report consolidates these strands into a single, validated BinaryLLM foundation: it reviews relevant literature, formalizes key mathematical relationships, outlines system‑level implications for modern NVIDIA GPUs, and specifies a rigorous experimental protocol and hypothesis set for evaluating binary embeddings, 1‑bit quantization in Transformer blocks, and binary KV‑caches.

---

## 2. Background

### 2.1 Quantization & Binarization

Quantization reduces numerical precision to lower the cost of storage and computation. In LLMs, established methods include 8‑bit and 4‑bit quantization for weights and activations, often combined with techniques to handle outlier channels and preserve accuracy. QLoRA shows that 4‑bit quantization with suitable scaling and low‑rank adapters can preserve near full‑precision performance for large LLaMA models. LLM.int8 and SmoothQuant demonstrate that careful handling of activation statistics and channel‑wise scaling allow accurate INT8 and INT4 inference and post‑training quantization.

BNNs represent an extreme point on this spectrum, with weights and activations constrained to \(\pm 1\), enabling dot‑products to be replaced by bitwise XNOR and popcount operations. BinaryConnect and Binarized Neural Networks established training procedures where binarization is applied in forward/backward passes while maintaining real‑valued master weights for updates. XNOR‑Net and Bi‑Real Net introduced architectural and training improvements (e.g., scaling factors, real‑valued shortcuts) that significantly narrow the accuracy gap to full‑precision networks on vision tasks.

For LLMs, 1‑bit quantization techniques such as BitNet adapt similar ideas to Transformer architectures, showing that 1‑bit weight matrices can be viable for large‑scale language modeling under appropriate training regimes.

### 2.2 Binary Embedding Theory

Binary embeddings map high‑dimensional real vectors to binary codes such that Hamming distance approximates a chosen similarity measure, typically angular distance or cosine similarity. Semantic Hashing demonstrated empirically that short binary codes can preserve semantic neighborhoods for document retrieval, enabling very fast search in Hamming space.

Binary embedding theory formalizes this: random sign projections applied to vectors in \(\mathbb{R}^d\) produce binary codes whose Hamming distance concentrates around a quantity proportional to the angle between vectors. The work of Jacques et al. shows that such embeddings can achieve nearly optimal dimensionality in the sense of the Johnson–Lindenstrauss lemma, preserving pairwise distances up to small distortions with a number of bits that is near‑minimal.

On the algorithmic side, Norouzi et al. show how to perform efficient exact nearest‑neighbor search in high‑dimensional binary code spaces via multi‑index hashing, demonstrating that binary embeddings are practical for large‑scale retrieval systems.

These results motivate the use of binary embeddings and bit‑latents in LLMs, provided that learned projections approximate the properties of random or well‑spread hyperplanes in the latent space.

### 2.3 Prior Low-bit LLM Work

Several lines of work apply low‑bit and extreme quantization to Transformers and LLMs:

- **BiBERT** presents a fully binarized BERT‑base model with binarized weights and activations plus additional mechanisms (Bi‑Attention and distillation). It achieves competitive performance on GLUE and other NLP tasks while significantly reducing FLOPs and model size, though a measurable accuracy gap remains.
- **BinaryBERT** explores 1‑bit and low‑bit quantization of BERT with careful scaling and distillation, showing that substantial compression can be obtained with modest accuracy loss on multiple NLP benchmarks.
- **BitNet** introduces a 1‑bit Transformer architecture for decoder‑only LLMs using BitLinear layers, reporting competitive language modeling performance with reduced memory and energy compared to FP16.
- **QLoRA**, **LLM.int8**, and **SmoothQuant** provide strong baselines for 4‑/8‑bit LLMs, combining quantization with specialized scaling and handling of outlier channels to preserve performance.

These works collectively indicate that:

- Transformer models tolerate aggressive quantization when combined with appropriate architectural and training modifications.
- Mixed‑precision regimes (e.g., keeping embeddings or output heads at higher precision) are often necessary for best quality.
- There is still limited systematic work on **fully** binarizing large decoder‑only LLMs, especially with binary KV‑caches and binary internal latents across long context lengths.

---

## 3. Mathematical Foundations

### 3.1 Binarization Operator (sign, scaled-sign)

Let \(x \in \mathbb{R}\) be a scalar and \(w \in \mathbb{R}^d\) a weight vector.

**Sign binarization** for a scalar:

\[
\mathcal{B}(x) = \mathrm{sign}(x) =
\begin{cases}
+1 & \text{if } x \ge 0,\\
-1 & \text{if } x < 0.
\end{cases}
\]

For a vector \(w\), binarization is applied elementwise.

**Scaled sign binarization** introduces a real‑valued scale \(\alpha > 0\) to reduce approximation error:

\[
\hat{w} = \alpha \,\mathrm{sign}(w),
\]

with \(\alpha\) chosen per‑layer or per‑channel to minimize mean‑squared error:

\[
\alpha^\star = \arg\min_\alpha \mathbb{E}\big[\lVert w - \alpha\,\mathrm{sign}(w)\rVert^2\big].
\]

Under zero‑mean, i.i.d. coordinates, \(\alpha^\star\) can be expressed in terms of \(\mathbb{E}[|w_i|]\); in practice, it is often estimated via statistics over a mini‑batch or a layer’s weights.

### 3.2 Straight-Through Estimator

The binarization operator \(\mathcal{B}\) is non‑differentiable and has zero derivative almost everywhere. To allow gradient‑based training, BNNs commonly use a Straight‑Through Estimator (STE) that approximates the gradient of \(\mathcal{B}\) w.r.t. its input.

Given a scalar \(x\) and binarized output \(y = \mathcal{B}(x)\), the STE approximates:

\[
\frac{\partial y}{\partial x} \approx
\begin{cases}
1 & |x| \le 1,\\
0 & \text{otherwise},
\end{cases}
\]

or, in a simplified form, \(1\) within a chosen range and \(0\) outside. This approximation allows gradients to flow through binarization in backpropagation while keeping the forward and backward binarized behavior at test time.

### 3.3 Random Hyperplane Theorem

Consider two non‑zero vectors \(u, v \in \mathbb{R}^d\) and a random projection \(r \sim \mathcal{N}(0, I_d)\). Define scalar random variables:

\[
X = r^\top u,\quad Y = r^\top v.
\]

Under the Gaussian assumption, \((X, Y)\) is jointly normal with correlation:

\[
\rho = \frac{\mathbb{E}[XY]}{\sqrt{\mathbb{E}[X^2]\mathbb{E}[Y^2]}} = \frac{u^\top v}{\|u\|\|v\|} = \cos \theta,
\]

where \(\theta\) is the angle between \(u\) and \(v\).

A classical result used in locality‑sensitive hashing and binary embedding theory is:

\[
P[\mathrm{sign}(X) \neq \mathrm{sign}(Y)] = \frac{\theta}{\pi}.
\]

This states that the probability that a random hyperplane (defined by \(r\)) separates the two points \(u\) and \(v\) is proportional to the angle between them.

### 3.4 Hamming vs Cosine Relationship

Now take \(m\) independent random projections \(r_1,\dots,r_m \sim \mathcal{N}(0, I_d)\) and define binary codes:

\[
b(u) = (\mathrm{sign}(r_1^\top u),\dots,\mathrm{sign}(r_m^\top u)) \in \{-1,+1\}^m,
\]
\[
b(v) = (\mathrm{sign}(r_1^\top v),\dots,\mathrm{sign}(r_m^\top v)) \in \{-1,+1\}^m.
\]

Let \(H(b(u), b(v))\) denote the Hamming distance between these binary codes (the number of coordinates where they differ). Since each coordinate differs with probability \(\theta/\pi\) and the coordinates are i.i.d.:

\[
\mathbb{E}[H(b(u), b(v))] = m \cdot P[\mathrm{sign}(X) \neq \mathrm{sign}(Y)] = m \cdot \frac{\theta}{\pi}.
\]

Thus the normalized Hamming distance satisfies:

\[
\mathbb{E}\left[\frac{H(b(u), b(v))}{m}\right] = \frac{\theta}{\pi},
\]

which is a monotonic function of \(\theta\) and therefore of the cosine similarity \(\cos\theta\). Concentration inequalities (e.g., Hoeffding’s inequality) imply that \(\frac{H}{m}\) will concentrate around \(\theta/\pi\) as \(m\) grows, preserving relative angular distances with high probability.

In BinaryLLM, projections used to generate binary embeddings may be learned rather than purely random, but this result serves as an ideal reference: to the extent that learned projections behave like a set of reasonably distributed hyperplanes in the feature space, Hamming distance between binary codes will track cosine similarity of the underlying real‑valued representations.

### 3.5 Rate–Distortion Argument

Let \(Z\) be a memoryless Gaussian source with variance \(\sigma^2\), i.e., \(Z \sim \mathcal{N}(0,\sigma^2)\), and consider quantizing \(Z\) at an average rate \(R\) bits per scalar in the mean‑squared error sense. The Shannon rate–distortion function for such a source is:

\[
R(D) = \frac{1}{2}\log_2\left(\frac{\sigma^2}{D}\right), \quad 0 < D \le \sigma^2,
\]

where \(D\) is the minimum achievable distortion (MSE) at rate \(R\). Solving for \(D\):

\[
D = \sigma^2 2^{-2R}.
\]

At \(R = 1\) bit per scalar (corresponding to a 1‑bit quantizer under these assumptions), the theoretical lower bound on distortion is:

\[
D_{\min} = \frac{\sigma^2}{4}.
\]

This formalizes the intuition that, even with an optimal quantizer, 1‑bit representations cannot recover more than a fraction of the variance of the original signal. For multi‑dimensional latents with approximately independent components, this bound applies componentwise, providing a lower bound on the MSE introduced by 1‑bit quantization.

For BinaryLLM, this implies:

- 1‑bit bit‑latents cannot be arbitrarily close to full‑precision latents; there is irreducible distortion.
- Moving from 1 bit to, e.g., 2–4 bits per scalar reduces the lower bound on distortion sharply, making hybrid 1–2‑bit designs potentially attractive.
- Training must make the model robust to this quantization noise and allocate bits where they yield the greatest reduction in effective distortion relative to task requirements.

---

## 4. System Modeling (Symbolic)

### 4.1 Memory Scaling

Let:

- \(N\) be the total number of parameters.
- \(b\) be the bit‑width per parameter (\(b \in \{1,2,4,8,16\}\)).
- \(C_{\text{mem}}\) be the device memory capacity in bytes.

The total parameter storage is:

\[
M_{\text{params}}(N, b) = N \cdot \frac{b}{8} \quad \text{[bytes]}.
\]

The maximum number of parameters that fit in memory (ignoring overheads) is:

\[
N_{\max}(b) = \frac{8\,C_{\text{mem}}}{b}.
\]

For two bit‑widths \(b_1\) and \(b_2\), the relative parameter capacity is:

\[
\frac{N_{\max}(b_1)}{N_{\max}(b_2)} = \frac{b_2}{b_1}.
\]

Thus, relative to FP16 (\(b=16\)), full binarization (\(b=1\)) increases the theoretical parameter capacity by a factor of \(16\), with analogous scaling for KV‑caches and activations, modulated by sequence length, number of layers, and whether all components are binarized.

### 4.2 FLOPs Scaling

Consider a standard Transformer with:

- \(d\) = model dimension,
- \(L\) = sequence length,
- \(n_L\) = number of layers.

Ignoring constant factors, the total operation count for a forward pass scales as:

\[
\mathrm{Ops}_{\text{FP}} \propto n_L \cdot L \cdot d^2.
\]

Switching from FP16 to 1‑bit quantization does **not** change the asymptotic count of these linear operations in terms of vector lengths and dimensions; instead, it changes:

- The **operation type**: from floating‑point multiply‑accumulate to bitwise XNOR+popcount plus scaling.
- The **achievable throughput** on the hardware (operations per second), which depends on how well binary kernels can utilize GPU resources.

Define effective peak throughputs:

- \(T_{\text{fp16}}^{\text{GPU}}\) for FP16 tensor‑core operations.
- \(T_{\text{int8}}^{\text{GPU}}\) for INT8 tensor‑core operations.
- \(T_{\text{bin}}^{\text{GPU}}\) for optimized binary matmul kernels using bit‑packing and XNOR+popcount.

The ideal time to execute \(\mathrm{Ops}_{\text{FP}}\) operations at each precision is:

\[
t_{\text{fp16}} \approx \frac{\mathrm{Ops}_{\text{FP}}}{T_{\text{fp16}}^{\text{GPU}}}, \qquad
t_{\text{bin}} \approx \frac{\mathrm{Ops}_{\text{FP}}}{T_{\text{bin}}^{\text{GPU}}}.
\]

The ideal speedup of binary over FP16 is:

\[
S_{\text{bin/fp16}}^{\text{ideal}} = \frac{t_{\text{fp16}}}{t_{\text{bin}}} = \frac{T_{\text{bin}}^{\text{GPU}}}{T_{\text{fp16}}^{\text{GPU}}}.
\]

In practice, \(T_{\text{bin}}^{\text{GPU}}\) must be measured empirically for A100, H100, and Blackwell GPUs using carefully implemented bit‑packed kernels.

### 4.3 Energy Model

Let:

- \(P_{\text{GPU}}\) be the average power draw during inference,
- \(T_{\text{eff}}\) be the effective realized throughput (operations per second),
- \(\mathrm{Ops}_{\text{FP}}\) be the total operation count for a forward pass.

The energy per operation is:

\[
E_{\text{op}} = \frac{P_{\text{GPU}}}{T_{\text{eff}}} \quad \text{[Joules/op]}.
\]

Total energy per forward pass:

\[
E_{\text{total}} = E_{\text{op}} \cdot \mathrm{Ops}_{\text{FP}}.
\]

Assuming similar power envelopes for configurations using FP16 and binary (which is an empirical question), and denoting effective throughputs by \(T_{\text{eff,fp16}}\) and \(T_{\text{eff,bin}}\), the ratio of total energies is:

\[
\frac{E_{\text{total,bin}}}{E_{\text{total,fp16}}} \approx \frac{T_{\text{eff,fp16}}}{T_{\text{eff,bin}}}.
\]

Additionally, since moving data often dominates energy in large models, the reduction in data volume via lower bit‑widths is crucial. Binary latents reduce bytes transferred per scalar by a factor proportional to \(1/b\). For KV‑caches and long‑context inference, such reductions can substantially lower both memory and interconnect energy, provided that packing/unpacking overheads are controlled and that the binary kernels achieve sufficient compute efficiency.

---

## 5. Experimental Protocol

### 5.1 Models and Variants

- **Base full‑precision LLM**:  
  - Decoder‑only Transformer with parameter count on the order of \(1\)–\(7\)B parameters, trained or finetuned in FP16 (or similar).
- **Quantized baselines**:
  - 4‑bit LLM using QLoRA‑style quantization and adapters.
  - 8‑bit LLM using LLM.int8‑style matrices and outlier channel handling.
- **BinaryLLM variants**:
  - **Binary‑Embeddings model**: binary token embeddings and binary pooled features, with internal layers at 4‑bit or higher precision.
  - **Binary‑Blocks model**: selected attention and/or MLP blocks binarized (weights and selected activations), rest at higher precision.
  - **Binary‑KV model**: KV‑caches stored at 1‑bit per scalar (with per‑channel scaling), with internal computations at 4‑bit or higher.
  - **Fully Binary model**: smaller‑scale Transformer encoder (e.g., BERT‑style), closely replicating BiBERT and related designs, possibly with BinaryLLM‑specific modifications.

### 5.2 Datasets

- **Language modeling / perplexity**:
  - WikiText‑103 as a standard benchmark.
  - Optionally, a well‑defined subset of a larger web corpus (e.g., C4‑like) for auxiliary evaluation.
- **NLP classification and NLU**:
  - GLUE benchmark tasks (e.g., MNLI, QQP, QNLI, SST‑2) for classification and natural language understanding.
- **Retrieval and similarity**:
  - BEIR benchmark datasets (e.g., MS MARCO, Natural Questions, SciFact) to evaluate binary embeddings and Hamming‑space retrieval.
- **Reasoning / QA (for task sensitivity)**:
  - A clearly specified subset of reasoning or knowledge‑heavy benchmarks (e.g., selected MMLU tasks), to test how binarization affects more challenging capabilities.

### 5.3 Metrics

- **Language modeling**:
  - Perplexity (PPL) and negative log‑likelihood on held‑out sets.
- **Classification / NLU**:
  - Accuracy, F1 score, and Matthews correlation coefficient (MCC) where appropriate per task.
- **Retrieval / similarity**:
  - nDCG@k and Recall@k on BEIR.
  - Top‑k neighbor overlap between binary and full‑precision embeddings:

    \[
    \text{Overlap}_k = \frac{1}{|Q|}\sum_{q \in Q} \frac{|N_k^{\text{binary}}(q) \cap N_k^{\text{float}}(q)|}{k},
    \]

    where \(Q\) is a set of queries and \(N_k^{\cdot}(q)\) is the top‑k nearest neighbors under Hamming (binary) or cosine (float) distance.

- **Energy / systems**:
  - Tokens per second (throughput) under a controlled hardware and software setup.
  - Joules per token for long‑context inference, measured via GPU power telemetry integrated over time.

### 5.4 Statistical Rigor

- **Number of seeds**:
  - At least 3 independent runs per configuration (model + quantization setting).
- **Reporting**:
  - Report mean and standard deviation for each metric across seeds.
  - For comparisons (e.g., BinaryLLM vs 4‑bit baseline), perform appropriate two‑sided significance tests (paired t‑test or non‑parametric alternative), with p‑value thresholds (e.g., \(p < 0.05\)) and corrections for multiple comparisons where necessary.
- **Pass/fail criteria**:
  - Each hypothesis is tied to explicit numeric thresholds (e.g., relative PPL deltas, neighbor overlaps, absolute metric differences). A hypothesis is considered supported or rejected based on whether the observed metrics and statistical tests fall within or outside these thresholds.

---

## 6. Literature Summary Table

The key literature relevant to BinaryLLM is summarized in the table below.

| Paper | Year | Domain | Key Findings | Limitations | Relevance to BinaryLLM |
| --- | --- | --- | --- | --- | --- |
| **BinaryConnect: Training Deep Neural Networks with binary weights during propagations** (M. Courbariaux, Y. Bengio, J.‑P. David, *NeurIPS*) | 2015 | BNN fundamentals | Introduces BinaryConnect, which binarizes weights (to \(\pm 1\)) during forward/backward passes while maintaining real‑valued master weights for updates. Shows competitive accuracy on CIFAR‑10/SVHN with large memory and compute savings in the forward path. | Only weights are binarized; activations remain real. Evaluated on relatively small CNNs and vision tasks. | Establishes the core “binarize‑for‑propagation, keep real master weights” paradigm, likely necessary for stable BinaryLLM training. |
| **Binarized Neural Networks** (M. Courbariaux et al., arXiv) | 2016 | Fully binarized CNNs | Extends BinaryConnect to binarize both weights and activations to \(\pm 1\), enabling most multiplications to be replaced by bitwise XNOR and popcount. Demonstrates good performance on MNIST, CIFAR‑10, and SVHN with strong theoretical speed/memory benefits. | Accuracy gap vs full precision widens on more complex datasets. No Transformer or NLP experiments. | Provides the canonical framework for fully binarized networks, including training tricks (STE, scaling factors) that inform BinaryLLM design. |
| **XNOR‑Net: ImageNet Classification Using Binary Convolutional Neural Networks** (M. Rastegari et al., *ECCV*) | 2016 | BNN + efficiency | Proposes XNOR‑Net, approximating convolutions with binary weights and activations plus per‑filter scaling. Achieves substantial speedups and memory reductions on ImageNet with reasonable accuracy. | Focused on CNNs for images; architecture differs significantly from Transformers. | Demonstrates that bitwise XNOR+popcount kernels can yield significant system‑level gains, motivating similar kernels for binary attention/MLPs. |
| **Bi‑Real Net: Binarizing Deep Network towards Real‑Network Performance** (Z. Liu et al., *ECCV*) | 2018 | Deep BNN optimization | Introduces Bi‑Real Net with real‑valued shortcut connections and an improved training algorithm, significantly narrowing the accuracy gap between 1‑bit and real‑valued ResNets on ImageNet. | Vision‑only evaluation; architectural tricks not directly validated in sequence models. | Shows that suitable use of real‑valued shortcuts and tailored optimization can mitigate binarization damage, a likely requirement for binary Transformer blocks. |
| **Semantic Hashing** (R. Salakhutdinov, G. Hinton, *International Journal of Approximate Reasoning*) | 2009 | Binary embeddings & retrieval | Learns short binary codes for documents such that Hamming distance approximates semantic similarity, enabling very fast retrieval via binary codes. Demonstrates that binary embeddings can preserve semantic neighborhoods. | Uses shallow models and document features; not LLM latents. | Foundational empirical evidence that binary embeddings can preserve semantic relationships, informing BinaryLLM’s binary embedding design. |
| **Binary Embeddings with Nearly Optimal Dimensionality for the Johnson–Lindenstrauss Lemma** (L. Jacques et al., *IEEE Trans. Information Theory*) | 2013 | Theory of binary embeddings | Provides theoretical guarantees that random sign projections followed by binary quantization can preserve pairwise angular distances up to small distortion, with near‑optimal embedding dimension. | Assumes random projections and idealized settings; does not address learned deep features. | Supplies rigorous backing for Hamming‑space similarity approximating cosine distance, directly supporting bit‑latent similarity metrics. |
| **Efficient Similarity Search in Hamming Space Using Multi‑Index Hashing** (M. Norouzi et al., *CVPR*) | 2012 | Binary similarity search | Proposes multi‑index hashing for exact nearest‑neighbor search in high‑dimensional binary codes, demonstrating practical, large‑scale retrieval in Hamming space. | Focused on vision features; not specific to LLM latents. | Shows that large‑scale infrastructure can exploit binary codes efficiently, relevant for BinaryLLM retrieval/memory subsystems. |
| **BiBERT: Accurate Fully Binarized BERT** (H. Qin et al., arXiv:2203.06390) | 2022 | Fully binarized Transformer encoder (NLP) | Presents BiBERT, a BERT‑base model with fully binarized weights and activations plus Bi‑Attention and distillation. Achieves competitive results on GLUE and other NLP tasks with greatly reduced FLOPs and model size. | Still exhibits measurable accuracy degradation vs full‑precision BERT; evaluated on encoder‑only tasks. | Direct proof that fully binarized Transformer encoders can handle real NLP tasks, providing a concrete starting point for BinaryLLM design. |
| **BinaryBERT: Pushing the Limit of BERT Quantization** (arXiv:2012.15701) | 2020 | Extreme BERT quantization | Explores 1‑bit and low‑bit quantization of BERT with carefully designed scaling and distillation to reduce the accuracy gap, reporting substantial compression and modest performance loss on several NLP benchmarks. | Some configurations rely on mixed precision; not all layers fully binarized. | Complementary evidence that BERT‑style models tolerate extreme quantization when combined with careful training, informing BinaryLLM’s encoder components. |
| **BitNet: Scaling 1‑bit Transformers for Large Language Models** (H. Wang et al., arXiv:2310.11453) | 2023 | 1‑bit LLM architectures | Introduces BitNet, a 1‑bit Transformer architecture for autoregressive LLMs using BitLinear layers, demonstrating that 1‑bit weight matrices can support competitive language modeling performance with reduced memory and energy vs FP16. | Published results cover specific sizes and setups; full behavior across scales and tasks remains to be mapped. | Flagship precedent that 1‑bit Transformer blocks can scale to LLMs, a direct architectural reference for BinaryLLM. |
| **QLoRA: Efficient Finetuning of Quantized LLMs** (T. Dettmers et al., *ICML*) | 2023 | 4‑bit LLM quantization & finetuning | Proposes QLoRA with 4‑bit quantization and low‑rank adapters, enabling near‑full‑precision finetuning of large LLaMA models on modest hardware. Shows that 4‑bit quantization plus careful scaling can preserve performance. | Uses 4‑bit weights; does not reach 1‑bit. Focuses on finetuning, not full training. | Establishes best practices for low‑bit LLMs and provides a strong 4‑bit baseline for BinaryLLM comparisons. |
| **LLM.int8(): 8‑bit Matrix Multiplication for Transformers at Scale** (T. Dettmers et al., *ICLR*) | 2022 | INT8 LLM inference | Demonstrates that carefully implemented 8‑bit matmuls with outlier channel handling can serve large pretrained Transformers with negligible quality loss. | Limited to 8‑bit; no 1‑bit operations. | Highlights the importance of outlier channels and channel‑wise scaling, lessons that extend to 1‑bit quantization. |
| **SmoothQuant: Accurate and Efficient Post‑Training Quantization for Large Language Models** (Z. Xiao et al., arXiv) | 2023 | PTQ for LLMs (INT8/INT4) | Proposes SmoothQuant, shifting activation variability into weights to enable accurate INT8/INT4 PTQ for LLMs, showing that addressing activation outliers is crucial for aggressive quantization. | Does not operate at 1‑bit; requires calibration data. | Provides techniques for handling activation outliers and rebalancing scales, which are important when pushing activations toward 1‑bit. |
| **Least Squares Binary Quantization of Neural Networks** (Y. Guo et al., arXiv:2001.02786) | 2020 | Binary weight quantization algorithms | Formulates binary quantization as a least‑squares optimization problem, reducing approximation error between real and binary weights and improving accuracy over naive binarization on CNNs. | Evaluated on vision benchmarks; no direct Transformer/LLM results. | Supplies improved projection methods from real weights to binary codes, potentially useful for BinaryLLM weight binarization. |
| **Binary‑Weighted Neural Networks Using FeRAM Array for Low‑Power AI Computing** (Nanomaterials, MDPI) | 2023 | BNN hardware & energy | Demonstrates a FeRAM‑based synaptic array implementing binary‑weighted neural networks with large reductions in dynamic and standby power and high TOPS/W efficiency in a specific CMOS process. | Hardware‑specific; workloads are small networks, not LLMs. | Indicates that hardware specifically optimized for binary weights can drastically reduce energy, motivating hardware co‑design for BinaryLLM. |

---

## 7. Hypotheses (validated)

The following BinaryLLM‑specific hypotheses are explicitly testable and falsifiable under the experimental protocol.

**H1 – Binary embeddings preserve retrieval neighborhoods at sufficient code length.**  
Learned binary embeddings of dimension \(m\) for document/text representations will, on BEIR datasets:

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

---

## 8. Feasibility Analysis (GPU-level)

The feasibility of BinaryLLM on NVIDIA A100, H100, and Blackwell GPUs depends on both algorithmic and systems factors.

- **Tensor cores vs CUDA cores**:  
  A100 and H100 tensor cores are optimized for FP16/BF16/TF32/FP8 and INT8; there is no publicly documented 1‑bit matrix multiply mode. Binary matmuls must therefore be implemented via CUDA cores using integer instructions, bit‑packing, XNOR, and popcount, with per‑tile scaling and re‑centering.

- **Kernel structure**:  
  Binary matrix multiplication kernels can adopt GEMM‑like tiling, staging packed words in shared memory and operating on blocks. The main challenges are:
  - Efficient bit‑packing/unpacking,
  - Maintaining memory coalescing with bit‑packed layouts,
  - Achieving high occupancy while managing register and shared memory usage.

- **Bandwidth vs compute**:  
  Even with bit‑packing, KV‑cache and activation traffic can keep workloads memory‑bound. Binary representations reduce per‑element bytes by factors up to 16 vs FP16, but if compute throughput \(T_{\text{bin}}^{\text{GPU}}\) is significantly below tensor‑core \(T_{\text{fp16}}^{\text{GPU}}\), overall speedups may be modest. Measuring \(T_{\text{bin}}^{\text{GPU}}\) and comparing tokens/s vs low‑bit tensor‑core baselines is necessary.

- **Energy considerations**:  
  Energy per operation and per byte moved must be empirically profiled under realistic inference workloads. BinaryLLM’s primary energy advantage is expected from reduced memory traffic for parameters and KV‑caches, especially at long context lengths. Compute energy savings will depend on how much effective throughput bitwise kernels can achieve on each GPU generation.

- **Alternative hardware**:  
  FeRAM‑based BNN accelerators indicate that binary‑optimized hardware can deliver very high efficiency (TOPS/W) for small networks. While these are not LLM‑specific, they suggest that more aggressive hardware–software co‑design for binary LLM workloads could further enhance energy efficiency beyond what is possible on general‑purpose GPUs.

Overall, BinaryLLM is **feasible to prototype** on current NVIDIA GPUs via custom kernels and careful engineering, but any claimed speed or energy advantages relative to strong 4‑bit/8‑bit baselines must be validated via detailed microbenchmarks and end‑to‑end measurements.

---

## 9. Discussion

The BinaryLLM approach is grounded in three pillars:

1. **Foundational BNN methods** that demonstrate how to train binarized networks and mitigate the accuracy gap via architectural and training modifications (real‑valued shortcuts, scaling, STE, distillation).
2. **Binary embedding theory** that rigorously relates Hamming distance between binary codes to angular distance and thus cosine similarity, providing a principled basis for binary embeddings and bit‑latents.
3. **Low‑bit LLM practice** that shows 4‑bit and 8‑bit quantization can be deployed at scale with minimal quality loss using suitable techniques (outlier handling, SmoothQuant, QLoRA), and that 1‑bit Transformers are plausible in principle (BitNet).

The validated v2 report explicitly acknowledges and structures the remaining gaps:

- Lack of systematic work on binary KV‑caches and internal bit‑latents for long‑context decoder‑only LLMs.
- Limited understanding of task‑specific sensitivity to binarization across retrieval, classification, language modeling, and reasoning.
- Absence of standardized recipes for stable, large‑scale 1‑bit training, and the need for clear energy‑centric metrics (Joules/token, tokens/s at given power) across GPU generations.

The experimental protocol and hypotheses are designed to answer these questions empirically, with clear thresholds for success and falsification. Symbolic system models and hardware‑level analysis ensure that the planned experiments can be interpreted in terms of memory scaling, FLOP/operation throughput, and energy consumption on known GPU architectures.

---

## 10. Conclusion

This report provides a validated, NVIDIA‑grade foundation for the BinaryLLM project. It consolidates key BNN, binary embedding, and low‑bit LLM literature; formulates essential mathematical relationships (binarization operators, Hamming vs cosine distance, rate–distortion bounds); and outlines symbolic models for memory, compute, and energy on A100, H100, and Blackwell GPUs. It also specifies a precise experimental protocol and hypothesis set, tying theoretical considerations directly to measurable metrics on standard datasets.

The next phase of BinaryLLM work is to implement and evaluate the outlined BinaryLLM variants, rigorously measure their quality and efficiency relative to strong 4‑bit/8‑bit baselines, and use the resulting evidence to refine or reject the proposed hypotheses. The final goal is to distill these findings into a BinaryLLM Protocol that prescribes bit allocations, training procedures, and hardware mappings for practical deployment of energy‑efficient, binarized LLMs.

---

## Bibliography

- Courbariaux, M., Bengio, Y., & David, J.‑P. (2015). BinaryConnect: Training Deep Neural Networks with binary weights during propagations. *Advances in Neural Information Processing Systems*.
- Courbariaux, M., Hubara, I., Soudry, D., El‑Yaniv, R., & Bengio, Y. (2016). Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or −1. arXiv:1602.02830.
- Rastegari, M., Ordonez, V., Redmon, J., & Farhadi, A. (2016). XNOR‑Net: ImageNet Classification Using Binary Convolutional Neural Networks. *European Conference on Computer Vision (ECCV)*.
- Liu, Z., Luo, W., Wu, B., Yang, X., Liu, W., & Cheng, K.‑T. (2018). Bi‑Real Net: Binarizing Deep Network towards Real‑Network Performance. *European Conference on Computer Vision (ECCV)*.
- Salakhutdinov, R., & Hinton, G. (2009). Semantic Hashing. *International Journal of Approximate Reasoning*.
- Jacques, L., Laska, J. N., Boufounos, P. T., & Baraniuk, R. G. (2013). Robust 1‑bit Compressive Sensing via Binary Stable Embeddings of Sparse Vectors. *IEEE Transactions on Information Theory*.
- Norouzi, M., Punjani, A., & Fleet, D. J. (2012). Fast Exact Search in Hamming Space with Multi‑Index Hashing. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
- Qin, H., Ding, Y., Zhang, M., Yan, Q., Liu, A., Dang, Q., Liu, Z., & Liu, X. (2022). BiBERT: Accurate Fully Binarized BERT. arXiv:2203.06390.
- Zhang, Y. et al. (2020). BinaryBERT: Pushing the Limit of BERT Quantization. arXiv:2012.15701.
- Wang, H., Ma, S., Dong, L., Huang, S., Wang, H., Ma, L., Yang, F., Wang, R., Wu, Y., & Wei, F. (2023). BitNet: Scaling 1‑bit Transformers for Large Language Models. arXiv:2310.11453.
- Dettmers, T., Lewis, M., Shleifer, S., & Zettlemoyer, L. (2022). LLM.int8(): 8‑bit Matrix Multiplication for Transformers at Scale. *International Conference on Learning Representations (ICLR)*.
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *International Conference on Machine Learning (ICML)*.
- Xiao, Z., Chen, J., Yan, S., Zhang, Y., & Jin, R. (2023). SmoothQuant: Accurate and Efficient Post‑Training Quantization for Large Language Models. arXiv:2211.10438.
- Guo, Y., Zhang, A., Zhang, Y., & Chen, Z. (2020). Least Squares Binary Quantization of Neural Networks. arXiv:2001.02786.
- (FeRAM BNN) Binary‑Weighted Neural Networks Using FeRAM Array for Low‑Power AI Computing. *Nanomaterials (MDPI)*.



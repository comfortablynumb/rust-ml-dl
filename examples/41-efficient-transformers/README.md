# Efficient Transformers ⚡

**Making Transformers fast and scalable**: Reduce the O(n²) attention bottleneck to O(n) with linear attention, Flash Attention, and sparse patterns.

## Overview

Standard Transformers have a critical problem: **quadratic complexity** in sequence length. This makes them impractical for:
- Long documents (>512 tokens)
- High-resolution images (>224×224)
- Audio/video (very long sequences)
- Real-time applications

**The Problem:**
```
Standard attention: O(n²) memory and compute
For n=1024: ~1M operations
For n=4096: ~16M operations (16× worse!)
For n=16384: ~256M operations (256× worse!)
```

Efficient Transformers solve this with clever approximations and algorithmic improvements.

## The Attention Bottleneck

### Standard Self-Attention (O(n²))

```
Q, K, V = Linear projections (n × d)

Attention scores: S = QK^T / √d_k     # O(n²) memory!
Softmax: A = softmax(S)                # O(n²)
Output: O = AV                         # O(n²)
```

**Why O(n²)?**
- QK^T creates n×n attention matrix
- Every token attends to every other token
- For n=2048: 4M attention scores!

**Memory Bottleneck:**
- GPT-3 (2048 tokens): 4M floats per attention head
- 96 heads: 384M floats = 1.5GB per layer
- 96 layers: 144GB just for attention!

## Efficient Attention Approaches

### 1. Linear Attention ⚡

**Paper:** Katharopoulos et al. (2020) - "Transformers are RNNs"

**Key Insight:** Reorder the attention computation!

**Standard:**
```
Attention(Q, K, V) = softmax(QK^T) V
                      ↑
                    O(n²)
```

**Linear Attention:**
```
Attention(Q, K, V) = φ(Q) (φ(K)^T V)
                            ↑
                          O(n)
```

**The Trick:**
```
Standard: Compute QK^T first (n×n matrix), then multiply by V
Linear: Compute K^TV first (d×d matrix), then multiply by Q

Matrix dimensions:
  Q: n × d
  K: n × d
  V: n × d

Standard: (n×d @ d×n) @ n×d = n×n @ n×d = O(n²d)
Linear: n×d @ (d×n @ n×d) = n×d @ d×d = O(nd²)

When d << n: Linear is much faster!
```

**Feature Map φ:**
```
Replace softmax with kernel feature map:
  softmax(q^Tk) ≈ φ(q)^T φ(k)

Common choices:
  - ELU + 1: φ(x) = elu(x) + 1
  - ReLU: φ(x) = relu(x)
  - Random Fourier Features
```

**Benefits:**
- O(n) complexity instead of O(n²)
- Can be computed recurrently (like RNN!)
- Constant memory
- 10-100× faster for long sequences

**Drawbacks:**
- Approximation (not exact attention)
- Slightly lower accuracy (~1-2%)
- Harder to train

### 2. Flash Attention 🔥

**Paper:** Dao et al. (2022) - "FlashAttention: Fast and Memory-Efficient Exact Attention"

**Breakthrough:** Exact attention with O(n) memory through clever I/O optimization!

**The Problem:**
- GPUs have fast HBM (High Bandwidth Memory) but limited size
- Transferring n×n attention matrix to/from memory is the bottleneck
- Standard attention: Many read/write operations

**FlashAttention Algorithm:**
```
1. Divide Q, K, V into blocks (fit in SRAM)
2. Compute attention block-by-block
3. Use online softmax (streaming algorithm)
4. Never materialize full n×n matrix
5. Fuse operations (minimize memory transfers)
```

**Key Innovation: Tiled Computation**
```
Standard: Compute full QK^T, then softmax, then multiply V
Flash: For each tile:
  - Load Q_block, K_block from HBM to SRAM
  - Compute local attention scores
  - Apply softmax incrementally
  - Update output
  - Never store full attention matrix!
```

**Online Softmax:**
```
Standard softmax: Need full row for normalization
Online softmax: Update running max and sum

for each token:
  Update max_score
  Update exp_sum (with rescaling)
  Output = weighted combination

Final: Normalize once at end
```

**Benefits:**
- **2-4× faster** than standard attention
- **5-20× less memory** (no n×n matrix!)
- **Exact** (same output as standard attention)
- Enables training with 64K token sequences
- **Training speedup**: 15% faster BERT, 3× faster GPT-2

**Drawbacks:**
- Implementation complexity
- Requires custom CUDA kernels
- Not all frameworks support it yet

**Impact:**
- Used in GPT-4, Llama 2, PaLM
- Standard for training large models
- Enabled long-context models (32K, 100K tokens)

### 3. Sparse Attention Patterns 🕸️

**Idea:** Not all tokens need to attend to all other tokens - use structured sparsity!

#### 3a. Local Attention (Sliding Window)

```
Each token attends to k neighbors:
  Token i attends to [i-k/2, ..., i, ..., i+k/2]

Complexity: O(nk) where k << n
Example: k=128 gives 16× speedup for n=2048
```

**Use case:** Language modeling (local context usually sufficient)

#### 3b. Strided Attention

```
Every k-th position attends globally
Others attend locally

Pattern:
  Token 0, k, 2k, ... → Full attention (anchor tokens)
  Other tokens → Local attention
```

**Example (Sparse Transformer):**
```
For n=2048, k=128:
  Tokens 0, 128, 256, ... → Attend to all 2048
  Token 5 → Attends to [0, 1, 2, ..., 10] (local)
```

#### 3c. Random Attention

```
Each token attends to:
  - Local neighbors (always)
  - Random sample of r tokens (for global info)

Complexity: O(n(k + r))
```

**Used in:** BigBird, Longformer

#### 3d. Block-Sparse Attention

```
Divide into blocks, use sparse block patterns:
  - Diagonal blocks (local)
  - Strided blocks (global)
  - Random blocks (mix)
```

**Used in:** OpenAI's Sparse Transformer

### 4. Approximate Attention

#### Performer (Random Features)

**Paper:** Choromanski et al. (2021)

**Idea:** Approximate softmax attention with random Fourier features

```
softmax(QK^T) ≈ φ_random(Q) φ_random(K)^T

Where φ uses random projections
Complexity: O(nd²r) where r = num random features
```

**Benefits:**
- Provable approximation guarantees
- Works well in practice
- Maintains performance

#### Linformer (Low-Rank Projection)

**Paper:** Wang et al. (2020)

**Idea:** Project key and value to lower dimension

```
K_projected = K @ E  (n×d → k×d, k << n)
V_projected = V @ F

Attention with projected K, V: O(nk)
```

**Observation:** Attention matrix is often low-rank

### 5. Hierarchical Attention

**Idea:** Multi-scale attention

```
Level 1: Fine-grained (local, n tokens)
Level 2: Medium-grained (chunks, n/4 tokens)
Level 3: Coarse-grained (global, n/16 tokens)

Total complexity: O(n + n/4 + n/16) ≈ O(n)
```

**Used in:** Longformer, BigBird

## Comparison

| Method | Complexity | Memory | Exact? | Speedup (n=4096) | Use Case |
|--------|-----------|--------|--------|------------------|----------|
| Standard | O(n²) | O(n²) | ✅ Yes | 1× (baseline) | n < 512 |
| Linear Attention | O(n) | O(1) | ❌ Approx | 10-50× | Long sequences |
| Flash Attention | O(n²)* | O(n) | ✅ Yes | 2-4× | All cases |
| Sparse (Local) | O(nk) | O(nk) | ❌ No | 8-16× | Language |
| Performer | O(n) | O(n) | ❌ Approx | 5-10× | Long sequences |
| Linformer | O(nk) | O(nk) | ❌ Approx | 4-8× | Fixed length |

*Same complexity but optimized I/O

## Real-World Applications

### Long Document Understanding

```
Standard Transformer: 512 tokens max (BERT)
Efficient Transformers: 4K-128K tokens

Applications:
  - Legal document analysis
  - Scientific paper understanding
  - Book-length context
```

**Models:**
- Longformer: 4K tokens (sparse attention)
- BigBird: 4K tokens (sparse patterns)
- LongT5: 16K tokens (local + global)

### High-Resolution Vision

```
Standard ViT: 224×224 = 196 patches
Efficient ViT: 512×512 = 1024 patches

Complexity:
  Standard: 196² = 38K
  Efficient: 1024² = 1M → Use linear attention
```

### Long-Form Generation

```
GPT-3: 2048 tokens
GPT-4: 8K-32K tokens (likely Flash Attention)
Claude: 100K tokens

Use case: Long-form writing, code generation
```

### Real-Time Applications

```
Speech recognition: 10K+ audio frames
Video understanding: 1K+ frames
Real-time chat: Fast inference required

Solution: Linear attention for O(n) inference
```

## Modern Models Using Efficient Attention

| Model | Technique | Context Length | Year |
|-------|-----------|----------------|------|
| Longformer | Sparse (local+global) | 4K | 2020 |
| BigBird | Sparse (random+window+global) | 4K | 2020 |
| Performer | Random features | 64K | 2021 |
| Linformer | Low-rank projection | 4K | 2020 |
| GPT-4 | Flash Attention (rumored) | 8K-32K | 2023 |
| Llama 2 | Flash Attention | 4K | 2023 |
| Claude 2 | Unknown (likely Flash) | 100K | 2023 |

## Implementation Challenges

### Linear Attention
```python
# Challenge: Feature map choice affects quality
φ(x) = ?  # ELU+1, ReLU, RFF?

# Training stability
# Need careful initialization
```

### Flash Attention
```python
# Requires custom CUDA kernels
# Complex tiling logic
# Framework-specific

# Rust: Would need GPU programming
```

### Sparse Attention
```python
# Efficient sparse matrix ops
# Pattern definition
# Indexing complexity
```

## Best Practices

### When to Use What?

**Short sequences (n < 512):**
- Use standard attention (fast enough)
- Flash Attention for training speedup

**Medium sequences (512 < n < 4K):**
- Flash Attention (exact, fast)
- Local + Sparse for even more speed

**Long sequences (n > 4K):**
- Linear attention for inference
- Sparse patterns (Longformer-style)
- Flash Attention for training

**Memory-constrained:**
- Flash Attention (5-20× less memory)
- Linear attention (constant memory)

**Latency-critical:**
- Linear attention (O(n) scales linearly)
- Local attention with small window

## Key Takeaways

1. **Flash Attention is the default** for training (exact, 2-4× faster)
2. **Linear attention** enables 100K+ token context
3. **Sparse patterns** balance speed and quality
4. **Complexity matters**: n² → n is 100× speedup for n=100
5. **Different tasks need different patterns** (language vs vision)

## Running the Example

```bash
cargo run --package efficient-transformers
```

Demonstrates:
- Complexity comparison (O(n²) vs O(n))
- Linear attention simulation
- Sparse attention patterns
- Memory usage comparison

## References

- **Linear Attention:** Katharopoulos et al. (2020) - "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
- **Flash Attention:** Dao et al. (2022) - "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
- **Sparse Transformer:** Child et al. (2019) - "Generating Long Sequences with Sparse Transformers"
- **Longformer:** Beltagy et al. (2020) - "Longformer: The Long-Document Transformer"
- **Performer:** Choromanski et al. (2021) - "Rethinking Attention with Performers"
- **Linformer:** Wang et al. (2020) - "Linformer: Self-Attention with Linear Complexity"

## Impact

Efficient Transformers enabled:
- ✅ **Long-context models** (GPT-4 32K, Claude 100K)
- ✅ **Training speedup** (Flash Attention is standard)
- ✅ **High-resolution vision** (efficient ViT)
- ✅ **Real-time applications** (faster inference)
- ✅ **Accessible AI** (lower memory requirements)

**Without efficient attention, long-context AI would be impractical!**

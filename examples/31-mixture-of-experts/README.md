# Mixture of Experts (MoE)

Sparse activation architecture enabling trillion-parameter models.

## Overview

MoE enables massive scale by activating only a subset of parameters per input, allowing models with trillions of parameters while maintaining reasonable compute costs.

## Running

```bash
cargo run --package mixture-of-experts
```

## Core Concept

### Dense vs Sparse Activation

**Dense Model (Traditional)**
```
Every parameter activated for every input
10B params = 10B FLOPs per token
```

**MoE Model (Sparse)**
```
Different experts for different inputs
100B params, but only 10B activated per token
10× larger model with same compute!
```

## Architecture

```
Input
  ↓
Router/Gate (learns which experts to use)
  ↓
Top-k Expert Selection
  ↓
┌─────┬─────┬─────┬─────┬─────┐
│Exp 1│Exp 2│Exp 3│Exp 4│...  │  ← Only k experts activated
└─────┴─────┴─────┴─────┴─────┘
  ↓
Weighted Combination
  ↓
Output
```

### MoE in Transformer

```
Standard Layer:
  x' = x + SelfAttention(x)        ← Dense
  y  = x' + FeedForward(x')        ← Dense

MoE Layer:
  x' = x + SelfAttention(x)        ← Dense
  y  = x' + MoE_FeedForward(x')    ← Sparse (only k experts)
```

## Routing Strategies

### 1. Top-1 (Switch Transformer)

```
Select single expert with highest probability
• Most sparse: 1/N activation
• Fastest inference
• Used in: Switch Transformer (1.6T params)
```

### 2. Top-2 (Common)

```
Select two highest-probability experts
• Balance: capacity vs efficiency
• Used in: Mixtral, many production models
```

### 3. Top-k (k > 2)

```
More experts for complex inputs
• Better quality
• More compute
• Typical k=2-4
```

## Expert Specialization

Experts naturally specialize during training!

**Example discoveries (language models):**

```
Expert 1: Punctuation and grammar
Expert 2: Named entities (people, places)
Expert 3: Numbers and dates
Expert 4: Technical/scientific terms
Expert 5: Common words and phrases
Expert 6: Rare/uncommon words
```

**Token routing example:**
```
"John" → Expert 2 (names)
"runs" → Expert 5 (common verbs)
"42"   → Expert 3 (numbers)
"."    → Expert 1 (punctuation)
```

## Load Balancing

### Problem: Expert Collapse

Without constraints, routing can collapse:
```
All tokens → Expert 1
Experts 2-N never used!
No benefit from multiple experts
```

### Solutions

**1. Auxiliary Load Balance Loss**
```
L_balance = α × Σ (fraction_routed_i × probability_i)

Encourages uniform distribution
Typical α = 0.01
```

**2. Expert Capacity**
```
Capacity = (tokens_per_batch / num_experts) × capacity_factor

If expert full → token dropped or overflow
Prevents overloading popular experts
```

**3. Random Routing**
```
With probability p, route to random expert
Ensures all experts get training signal
Typical p = 0.1 during training
```

## Scaling Benefits

### Comparison

**Dense GPT-3 (175B)**
- 175B parameters
- 175B FLOPs per token
- Training cost: $5-10M

**MoE Switch-XXL (1.6T)**
- 1,600B total parameters
- ~200B FLOPs per token (top-1 routing, 8 experts)
- Training cost: ~$30M
- **8× more parameters with similar compute!**

### Key Insight

```
Dense: N params = N compute
MoE:   N params < N compute (sparse activation)
```

## Famous MoE Models

### Switch Transformer (Google, 2021)
- **1.6 trillion parameters**
- Top-1 routing (simplest)
- 7× speedup over T5-XXL (same quality)

### GLaM (Google, 2021)
- **1.2 trillion parameters**
- Top-2 routing
- Beats GPT-3 with 1/3 energy cost

### GPT-4 (OpenAI, 2023)
- **Rumored MoE architecture**
- ~8 experts, ~1.8T total params
- Not officially confirmed
- Explains high quality + reasonable latency

### Mixtral 8x7B (Mistral AI, 2023)
- **Open-source MoE**
- 47B total params, 13B active per token
- Beats GPT-3.5 on many benchmarks
- Top-2 routing from 8 experts

### DeepSeek-MoE (2024)
- 16B params, 2B active
- Fine-grained expert splitting
- State-of-the-art efficiency

## MoE vs Dense Models

| Aspect | Dense | MoE |
|--------|-------|-----|
| **Scaling** | Linear (N params = N compute) | Sublinear (N params < N compute) |
| **Max scale** | ~100B params practical | Trillion+ params possible |
| **Training** | Simpler | Complex (load balancing) |
| **Deployment** | Smaller model size | Larger model size |
| **Hardware** | Better utilization (90%+) | Lower utilization (60-80%) |
| **Communication** | Minimal | All-to-All between GPUs |

## When to Use MoE

### Use MoE When:
✅ Need very large models (> 100B params)
✅ Have multi-GPU/multi-node setup
✅ Compute budget limited
✅ Want state-of-the-art performance
✅ Diverse data (benefits from specialization)

### Use Dense When:
✅ Model size < 10B params
✅ Single GPU deployment
✅ Simpler training pipeline desired
✅ Lower latency critical
✅ Uniform data distribution

## Training Challenges

### 1. Expert Collapse
**Problem**: Router sends everything to one expert
**Solution**: Load balance loss, capacity limits

### 2. Communication Cost
**Problem**: Experts on different GPUs
**Solution**: All-to-All communication, expert parallelism

### 3. Load Imbalance
**Problem**: Some experts process 90% tokens
**Solution**: Auxiliary loss, capacity constraints

### 4. Numerical Instability
**Problem**: Routing probabilities spike
**Solution**: Gradient clipping, careful initialization

## Practical Tips

1. **Start Small**: 4-8 experts initially
2. **Monitor Usage**: Log tokens per expert, detect collapse
3. **Capacity Factor**: Start with 1.25-1.5×
4. **Balance Loss**: α=0.01 typical, adjust if needed
5. **Expert Init**: Initialize experts differently for diversity

## Hardware Considerations

### Memory
```
Total params: All experts in GPU memory
Active params: Subset used per token
MoE needs more memory than dense of same compute
```

### Compute
```
Sparse activation → Lower FLOPs
But: Routing overhead, load imbalance
GPU utilization: 60-80% (vs 90%+ dense)
```

### Communication
```
All-to-All between GPUs (expert parallelism)
Bottleneck for small batches
Requires high-bandwidth (NVLink, InfiniBand)
```

## Future Directions

1. **Conditional Computation**: Sparse attention, sparse layers
2. **Learned Routing**: Hierarchical, content-based routing
3. **Dynamic Experts**: Create/remove during training
4. **Multi-modal MoE**: Different experts for text/image/audio
5. **Edge Deployment**: Small MoE for mobile (4-8 experts)

## Historical Timeline

```
1991: Jacobs et al. - Original MoE concept
2017: Shazeer et al. - Outrageously Large Neural Networks
2021: Switch Transformer - Simplified to top-1 routing
2021: GLaM - 1.2T params, beats GPT-3
2022-23: GPT-4 (rumored MoE architecture)
2023: Mixtral 8x7B - Open-source MoE
2024: DeepSeek-MoE - Fine-grained experts
```

## Papers

- [Outrageously Large Neural Networks: The Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538) (Shazeer et al., 2017)
- [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961) (Fedus et al., 2021)
- [GLaM: Efficient Scaling of Language Models](https://arxiv.org/abs/2112.06905) (Du et al., 2021)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) (Mistral AI, 2024)

## Key Takeaways

✓ MoE enables sparse activation: Not all params for all inputs
✓ Routing decides which experts to use per input
✓ Enables trillion-parameter models with reasonable compute
✓ Experts naturally specialize (names, numbers, grammar, etc.)
✓ Load balancing critical to prevent expert collapse
✓ Used in GPT-4, Switch Transformer, Mixtral, other SOTA models
✓ Trade-off: Massive scale vs deployment complexity
✓ Future: Likely standard for large-scale models

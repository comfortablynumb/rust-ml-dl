# Advanced Optimizers 🚀

Modern optimization algorithms that power deep learning: **Adam, RMSprop, AdaGrad, and learning rate scheduling**.

## Overview

While basic gradient descent works, modern deep learning relies on sophisticated optimization algorithms that adapt learning rates, use momentum, and handle sparse gradients. This example implements the most important optimizers used in production.

## The Problem with Basic SGD

**Standard SGD:**
```
w = w - η · ∇L
```

**Issues:**
- Single learning rate for all parameters
- No adaptation to parameter importance
- Slow convergence in ravines (high curvature)
- Sensitive to learning rate choice
- Oscillates in directions with different curvatures

## Modern Optimizers

### 1. Momentum 🏃

**Idea:** Accumulate velocity from past gradients to dampen oscillations.

**Update Rule:**
```
v_t = β · v_{t-1} + ∇L
w_t = w_{t-1} - η · v_t
```

**Benefits:**
- Accelerates convergence in relevant directions
- Dampens oscillations in irrelevant directions
- Helps escape local minima
- β typically 0.9

**When to Use:**
- Deep networks with high curvature
- When gradients are noisy
- Most modern optimizers include momentum

### 2. Nesterov Accelerated Gradient (NAG) 🎯

**Idea:** Look ahead before computing gradient.

**Update Rule:**
```
v_t = β · v_{t-1} + ∇L(w - β · v_{t-1})
w_t = w_{t-1} - η · v_t
```

**Benefits:**
- More responsive corrections
- Better convergence than standard momentum
- Prevents overshooting

### 3. AdaGrad (Adaptive Gradient) 📊

**Paper:** Duchi et al., 2011

**Idea:** Adapt learning rate for each parameter based on historical gradients.

**Update Rule:**
```
G_t = G_{t-1} + (∇L)²    (element-wise)
w_t = w_{t-1} - (η / √(G_t + ε)) · ∇L
```

**Benefits:**
- Larger updates for infrequent features
- Smaller updates for frequent features
- Good for sparse data (NLP, recommender systems)
- No manual learning rate tuning per parameter

**Drawbacks:**
- Learning rate monotonically decreases
- Can stop learning too early
- Accumulates entire gradient history

**When to Use:**
- Sparse gradients (word embeddings)
- Different parameter scales
- Early stopping is acceptable

### 4. RMSprop (Root Mean Square Propagation) 🌊

**Inventor:** Geoffrey Hinton (unpublished, Coursera lecture)

**Idea:** Fix AdaGrad's monotonic learning rate decay with exponential moving average.

**Update Rule:**
```
E[g²]_t = β · E[g²]_{t-1} + (1-β) · (∇L)²
w_t = w_{t-1} - (η / √(E[g²]_t + ε)) · ∇L
```

**Benefits:**
- Adapts learning rates per parameter
- Doesn't accumulate entire history (uses decay β ≈ 0.9)
- Works well for non-stationary objectives (RNNs)
- Good for mini-batch training

**Hyperparameters:**
- η (learning rate): 0.001 (default)
- β (decay rate): 0.9
- ε (numerical stability): 1e-8

**When to Use:**
- RNNs and LSTMs (very effective!)
- Non-stationary problems
- Mini-batch training
- Before Adam existed, this was the go-to

### 5. Adam (Adaptive Moment Estimation) ⭐

**Paper:** Kingma & Ba, 2014
**Status:** Most popular optimizer in deep learning

**Idea:** Combine momentum (first moment) with RMSprop (second moment).

**Update Rule:**
```
m_t = β₁ · m_{t-1} + (1-β₁) · ∇L          (first moment: momentum)
v_t = β₂ · v_{t-1} + (1-β₂) · (∇L)²      (second moment: RMSprop)

m̂_t = m_t / (1 - β₁ᵗ)                     (bias correction)
v̂_t = v_t / (1 - β₂ᵗ)                     (bias correction)

w_t = w_{t-1} - η · m̂_t / (√v̂_t + ε)
```

**Benefits:**
- Combines best of momentum and RMSprop
- Bias correction for first few iterations
- Adaptive learning rates per parameter
- Works well with little tuning
- Handles sparse gradients well
- Default choice for most deep learning

**Hyperparameters:**
- η (learning rate): 0.001 (or 3e-4 for RL)
- β₁ (momentum decay): 0.9
- β₂ (RMSprop decay): 0.999
- ε (numerical stability): 1e-8

**Variants:**
- **AdamW**: Adam with decoupled weight decay (better regularization)
- **Adamax**: Adam with infinity norm (more stable for some problems)
- **Nadam**: Adam + Nesterov momentum

**When to Use:**
- **Default choice** for most deep learning
- CNNs, Transformers, GANs
- When you want robust performance without tuning
- Sparse gradients (NLP)

### 6. AdamW (Adam with Weight Decay) 💪

**Paper:** Loshchilov & Hutter, 2017

**Key Insight:** L2 regularization ≠ weight decay in adaptive optimizers!

**Update Rule:**
```
m_t = β₁ · m_{t-1} + (1-β₁) · ∇L
v_t = β₂ · v_{t-1} + (1-β₂) · (∇L)²
m̂_t = m_t / (1 - β₁ᵗ)
v̂_t = v_t / (1 - β₂ᵗ)

w_t = w_{t-1} - η · (m̂_t / (√v̂_t + ε) + λ · w_{t-1})
                                        ↑ decoupled weight decay
```

**Benefits:**
- Better generalization than Adam
- Proper weight decay (not L2 regularization)
- Used in BERT, GPT, Stable Diffusion
- Slightly better than Adam on most tasks

**Hyperparameters:**
- Same as Adam, plus:
- λ (weight decay): 0.01 (typical)

**When to Use:**
- **Default for Transformers** (BERT, GPT)
- When generalization is critical
- Replace Adam in most production code

## Learning Rate Schedules 📉

Fixed learning rate is suboptimal. Schedules improve convergence and generalization.

### 1. Step Decay

**Idea:** Reduce LR by factor every N epochs.

```
η_t = η₀ · γ^(epoch // step_size)
```

**Example:** η₀=0.1, γ=0.1, step_size=30
- Epochs 0-29: η = 0.1
- Epochs 30-59: η = 0.01
- Epochs 60-89: η = 0.001

**When to Use:** Simple, interpretable, good default

### 2. Exponential Decay

**Idea:** Decay by constant factor each epoch.

```
η_t = η₀ · γᵗ
```

**When to Use:** Smooth decay, less common than step decay

### 3. Cosine Annealing ⭐

**Idea:** Decay following cosine curve.

```
η_t = η_min + (η_max - η_min) · (1 + cos(πt/T)) / 2
```

**Benefits:**
- Smooth decay
- Fast early, slow late (good for fine-tuning)
- **Widely used in Transformers** (BERT, GPT)
- Can restart (SGDR: Stochastic Gradient Descent with Restarts)

**When to Use:**
- **Modern default** for Transformers
- Long training runs
- When you know total epochs

### 4. Warmup + Cosine Annealing 🔥

**Most popular for Transformers!**

**Idea:** Linear warmup, then cosine decay.

```
if t < warmup_steps:
    η_t = η_max · t / warmup_steps        (linear warmup)
else:
    η_t = cosine_annealing(t - warmup_steps)
```

**Benefits:**
- Stabilizes early training (Transformers sensitive to init)
- Used in **BERT, GPT-3, Stable Diffusion**
- Best convergence for large models

**Hyperparameters:**
- warmup_steps: 500-10000 (or 10% of total steps)
- η_max: 1e-4 to 3e-4
- η_min: 0 or η_max/10

**When to Use:**
- **Default for all Transformers**
- Large models (>100M parameters)
- When training is unstable early on

### 5. Reduce on Plateau

**Idea:** Reduce LR when validation loss plateaus.

```
if val_loss doesn't improve for patience epochs:
    η = η · factor
```

**When to Use:**
- Unknown optimal epochs
- Small datasets
- PyTorch default

### 6. One-Cycle Policy 🚴

**Paper:** Smith, 2018

**Idea:** Single cycle of LR increase then decrease.

```
Phase 1 (0-45%): Linear increase η_min → η_max
Phase 2 (45-100%): Cosine decrease η_max → η_min
```

**Benefits:**
- Faster convergence (3-10× fewer epochs)
- Better generalization
- Regularization effect

**When to Use:**
- Limited training time
- Supervised learning tasks
- Works well for ResNets

## Additional Techniques

### Gradient Clipping ✂️

**Problem:** Exploding gradients in RNNs/LSTMs

**Solution:** Clip gradient norm.

```python
if ||g|| > threshold:
    g = g · threshold / ||g||
```

**When to Use:**
- **Required for RNNs/LSTMs**
- Transformers (sometimes)
- Threshold: 1.0 for RNNs, 5.0 for Transformers

### Gradient Accumulation 📦

**Problem:** Batch size limited by GPU memory

**Solution:** Accumulate gradients over multiple mini-batches.

```python
for accumulation_steps:
    loss = forward(batch)
    loss.backward()
optimizer.step()  # After accumulation
optimizer.zero_grad()
```

**When to Use:**
- Large models (Transformers)
- Simulates larger batch size
- Stable Diffusion training

## Optimizer Comparison

| Optimizer | Speed | Memory | Tuning | Best For |
|-----------|-------|--------|--------|----------|
| SGD + Momentum | Fast | Low | Hard | ResNets, when tuned well |
| AdaGrad | Medium | Medium | Easy | Sparse features, NLP |
| RMSprop | Fast | Medium | Medium | RNNs, non-stationary |
| Adam | Fast | High | Easy | **Default choice** |
| AdamW | Fast | High | Easy | **Transformers, modern default** |

## Practical Recommendations

### For CNNs (ResNet, EfficientNet)
```
Optimizer: SGD with momentum (0.9) + Nesterov
LR: 0.1 with step decay or cosine
Weight decay: 1e-4
Batch size: 256
```

### For Transformers (BERT, GPT)
```
Optimizer: AdamW
LR: 1e-4 to 3e-4
Schedule: Warmup (10%) + Cosine decay
Weight decay: 0.01
β₁=0.9, β₂=0.999
Gradient clipping: 1.0
```

### For RNNs/LSTMs
```
Optimizer: Adam or RMSprop
LR: 0.001
Gradient clipping: 1.0-5.0 (required!)
```

### For GANs
```
Generator: Adam (lr=0.0002, β₁=0.5)
Discriminator: Adam (lr=0.0002, β₁=0.5)
```

### For Reinforcement Learning
```
Optimizer: Adam
LR: 3e-4 (standard across RL)
```

## Modern Developments

### 1. Lion (Evolved Sign Momentum) - 2023
- Discovered by Google using evolutionary algorithms
- More memory-efficient than Adam
- Comparable performance with less memory

### 2. Sophia - 2023
- Second-order optimizer (uses Hessian approximation)
- 2× faster than Adam for Transformers
- Cutting-edge research

### 3. Per-Parameter Learning Rates
- Each layer gets different LR
- Discriminative fine-tuning (ULMFiT)
- Lower layers: 1e-5, higher layers: 1e-3

## Key Takeaways

1. **Default choice:** AdamW with warmup + cosine schedule
2. **For CNNs:** SGD + momentum still competitive when tuned
3. **For Transformers:** AdamW is the standard
4. **For RNNs:** Gradient clipping is mandatory
5. **Learning rate** matters more than optimizer choice
6. **Warmup** stabilizes training for large models
7. **Cosine annealing** is the modern default schedule

## Running the Example

```bash
cargo run --package advanced-optimizers
```

This will demonstrate:
- All optimizers on the same task
- Learning rate schedules
- Gradient clipping
- Convergence comparison

## References

- **Adam:** Kingma & Ba (2014) - "Adam: A Method for Stochastic Optimization"
- **AdamW:** Loshchilov & Hutter (2017) - "Decoupled Weight Decay Regularization"
- **Cosine Annealing:** Loshchilov & Hutter (2016) - "SGDR: Stochastic Gradient Descent with Warm Restarts"
- **One-Cycle:** Smith (2018) - "A disciplined approach to neural network hyper-parameters"

## Impact

Modern optimizers are **essential** for deep learning:
- Enable training of very deep networks (100+ layers)
- Reduce training time by 10-100×
- Make hyperparameter tuning easier
- Power all modern AI (GPT, BERT, Stable Diffusion)

The shift from hand-tuned SGD to adaptive optimizers (Adam/AdamW) was a key enabler of the deep learning revolution.

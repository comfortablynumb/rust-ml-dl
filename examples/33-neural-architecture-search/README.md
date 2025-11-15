# Neural Architecture Search (NAS)

AutoML for automatically discovering optimal neural network architectures.

## Overview

NAS automates the discovery of neural network architectures, often finding better designs than human experts. It's transforming deep learning from manual art to automated science.

## Running

```bash
cargo run --package neural-architecture-search
```

## Core Problem

### Manual Design (Traditional)
```
Expert designs architecture
  ↓
Trial and error
  ↓
Months/years for breakthrough
```
❌ Requires expertise
❌ Time-consuming
❌ Limited by human bias

### Neural Architecture Search
```
Define search space
  ↓
Automated search
  ↓
Discover optimal architecture
```
✅ Systematic exploration
✅ Often beats human designs
✅ Democratizes deep learning

## Three Components

### 1. Search Space
```
Set of possible architectures

Example:
  • Operations: Conv3x3, Conv5x5, MaxPool, Identity
  • Connections: Skip, sequential
  • Depth: 10-50 layers
```

### 2. Search Strategy
```
Algorithm to explore space

Methods:
  • Random search
  • Reinforcement learning
  • Evolutionary algorithms
  • Gradient-based
```

### 3. Performance Estimation
```
How to evaluate candidates

Approaches:
  • Full training (accurate, expensive)
  • Proxy estimates (fast, noisy)
  • Weight sharing
  • Predictors
```

## Search Space Design

### Macro Search Space
```
Global network structure:
  Conv → Pool → Conv → FC

Decisions:
  • Number of layers
  • Layer types
  • Skip connections
```

### Micro Search Space (Cell-based)
```
Design reusable "cell"
Stack cells to form network

Example (NASNet):
  Cell = graph with N nodes
  Each node: operation + 2 inputs
  Network = [Cell] × K + [Reduction] + [Cell] × K
```

**Advantages of cells**:
✅ Smaller search space
✅ Transferable across tasks
✅ Scalable (stack more cells for bigger network)

## Search Strategies

### 1. Random Search
```
Baseline: Sample random architectures

while budget remaining:
    arch = random_architecture()
    score = evaluate(arch)
    if score > best_score:
        best = arch
```

✅ Simple, parallelizable
✅ Surprisingly effective!
❌ No learning from evaluations

### 2. Reinforcement Learning (NASNet)
```
Controller RNN generates architectures
Reward = validation accuracy

Training:
  1. Controller outputs architecture
  2. Train & evaluate architecture
  3. Reward = accuracy
  4. Update controller with REINFORCE
```

**Example**: NASNet (Google, 2017)
- 800 GPU days
- Beat human designs on ImageNet
- Discovered depthwise separable convolutions

✅ Learns from experience
✅ Handles complex spaces
❌ Expensive (1000s of architectures)
❌ High variance

### 3. Evolutionary Algorithms
```
Population-based search

Algorithm:
  1. Initialize population
  2. Evaluate fitness (accuracy)
  3. Select top parents
  4. Mutate/crossover → offspring
  5. Repeat
```

**Mutations**:
- Change operation (Conv3x3 → Conv5x5)
- Add/remove layer
- Change connection

✅ Parallelizable
✅ Robust
❌ Many evaluations needed

### 4. Gradient-Based (DARTS)
```
Make architecture search differentiable!

Key idea:
  Continuous relaxation of discrete choices

  output = Σ α_i × operation_i(input)
  α = softmax(architecture_params)

  Search: Optimize α with gradient descent
```

**DARTS** (Differentiable Architecture Search, 2018):
```
Bilevel optimization:
  min_α  L_val(w*(α), α)
  s.t.   w*(α) = argmin_w L_train(w, α)

Approximation:
  Alternate:
    1. Update w on train set
    2. Update α on val set
```

✅ **Fast**: 4 GPU days (vs 1000s for RL)
✅ Efficient: Gradients guide search
✅ Memory efficient
❌ Continuous ≠ discrete
❌ Performance gap possible

## Performance Estimation

### Challenge
```
1000 architectures × 100 GPU hours = 100,000 GPU hours
Need faster evaluation!
```

### Solutions

**1. Lower Fidelity**
```
Instead of:
  • 200 epochs → 10 epochs
  • Full dataset → 10% subset
  • 224×224 → 32×32 resolution

Result: Approximate ranking, 10-100× faster
```

**2. Weight Sharing (One-Shot NAS)**
```
Single "super-network" contains all architectures
Share weights between candidates
Train super-network once
Evaluate by sampling paths

Result: Seconds instead of hours!
```

**3. Learning Curve Extrapolation**
```
Train for few epochs
Predict final performance
Stop poor performers early
```

**4. Performance Predictors**
```
Train model: Architecture encoding → Predicted accuracy
Use to filter before expensive evaluation
Warm-start with evaluated architectures
```

## Famous NAS Results

### NASNet (Google, 2017)
```
Method: RL-based search
Cost: 800 GPU days
Result: Beat human designs on ImageNet
Impact: NASNet cells transferred to detection, segmentation
```

### ENAS (2018)
```
Method: Weight sharing + RL
Cost: 0.5 GPU days
Result: 1000× faster than NASNet, similar performance
```

### DARTS (2018)
```
Method: Gradient-based
Cost: 4 GPU days on CIFAR-10
Result: Competitive with NASNet/ENAS
Impact: Simple, reproducible, widely adopted
```

### EfficientNet (Google, 2019)
```
Method: NAS + compound scaling
Search: Base architecture (EfficientNet-B0)
Scale: Depth, width, resolution together
Result: SOTA ImageNet (84.3%) with fewer parameters
```

### Vision Transformer + NAS
```
AutoFormer, Evolved Transformer
Search attention patterns, MLP dimensions
Improved efficiency over vanilla ViT
```

## Discovered Patterns

**What NAS Found**:

### 1. Depthwise Separable Convolutions
```
NAS consistently selects over standard convolutions
More efficient (fewer params, similar capacity)
Now widely used: MobileNet, EfficientNet
```

### 2. Skip Connections
```
NAS re-discovers skip connections
Similar to ResNet (human-designed)
Validates: Good ideas found by both
```

### 3. Mixed Operations
```
Combination of different kernel sizes (3×3, 5×5)
Not obvious to human designers
Irregular patterns (no human bias for symmetry)
```

## NAS Variations

### Hardware-Aware NAS
```
Optimize for: Accuracy + Latency + Energy + Memory

Multi-objective:
  Maximize: Accuracy
  Minimize: Latency, energy, model size

Result: Pareto front of architectures
Example: MobileNetV3 (optimized for mobile)
```

### Transferable NAS
```
Search on proxy task (CIFAR-10)
Transfer to target task (ImageNet)
Much cheaper than direct ImageNet search
```

### Once-For-All Networks
```
Single network → Multiple architectures
Select sub-network based on constraints
Train once, deploy many configurations
```

## When to Use NAS

### Use NAS When:
✅ Significant compute budget (10+ GPUs)
✅ Need SOTA performance on specific task
✅ Specific constraints (latency, mobile, edge)
✅ Research: Discover new patterns

### Use Existing Architectures When:
✅ Limited compute (<10 GPUs)
✅ Well-solved problem
✅ Quick prototype needed
✅ Transfer learning sufficient

## Practical Approach

```
1. Start: Existing architectures (ResNet, EfficientNet, ViT)
2. If insufficient: Try architecture search
3. Method: Use efficient NAS (DARTS, weight sharing)
4. Strategy: Search on proxy, transfer to target
5. Refine: Use discovered arch as starting point
```

## Challenges

### 1. Computational Cost
```
Even efficient NAS: 4-100 GPU days
Not accessible to most researchers
Requires significant resources
```

### 2. Search Space Design
```
Good search space crucial
Requires domain expertise
Paradox: Automation needs expert input
```

### 3. Overfitting to Search Dataset
```
Architecture optimized for validation set
May not generalize to new datasets
Transfer performance can disappoint
```

### 4. Reproducibility
```
High variance across runs
Sensitive to random seed, hyperparameters
Hard to compare methods fairly
```

### 5. Evaluation Noise
```
Proxy estimates can mislead
Weight sharing biases evaluation
Best on proxy ≠ best when trained from scratch
```

## Future Directions

### 1. Faster Search
```
Better performance predictors
Zero-cost proxies (no training)
Neural architecture ranking
```

### 2. Larger Search Spaces
```
Beyond CNNs: Transformers, MoE, hybrids
Cross-modality: Vision + language + audio
Task-agnostic: Single search for multiple tasks
```

### 3. End-to-End AutoML
```
Not just architecture:
  • Data augmentation
  • Learning rate schedules
  • Optimizers
  • Training procedures
```

### 4. Neural Architecture Editing
```
Start from existing architecture
Incrementally improve through search
Interpretable modifications
```

### 5. Constitutional NAS
```
Incorporate constraints:
  • Fairness
  • Interpretability
  • Robustness
  • Privacy
Not just accuracy
```

## Papers

- [Neural Architecture Search with RL](https://arxiv.org/abs/1611.01578) (Zoph & Le, 2016)
- [Learning Transferable Architectures (NASNet)](https://arxiv.org/abs/1707.07012) (Zoph et al., 2017)
- [Efficient NAS via Parameter Sharing (ENAS)](https://arxiv.org/abs/1802.03268) (Pham et al., 2018)
- [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) (Liu et al., 2018)
- [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946) (Tan & Le, 2019)

## Key Takeaways

✓ **NAS**: Automates neural architecture discovery
✓ **Components**: Search space, strategy, performance estimation
✓ **Strategies**: Random, RL, evolution, gradient-based (DARTS)
✓ **Efficiency**: Weight sharing, low-fidelity estimates, predictors
✓ **Success**: NASNet, EfficientNet, ENAS, DARTS
✓ **Discoveries**: Depthwise separable convs, skip connections, irregular patterns
✓ **Future**: Essential AutoML tool, democratizes deep learning
✓ **Impact**: From manual art → automated science

# Contrastive Learning 🔥

**Self-supervised learning that powers modern AI**: Learn visual representations without labels using contrastive methods like SimCLR, MoCo, BYOL, and CLIP.

## Overview

Contrastive learning is revolutionizing AI by enabling models to learn from **unlabeled data** at massive scale. It's the foundation of:
- **CLIP** (powers Stable Diffusion, DALL-E prompts)
- **Self-supervised pre-training** (ImageNet without labels!)
- **Multi-modal learning** (vision + language)

## The Problem: Labeled Data is Expensive

**Traditional Supervised Learning:**
```
Need: 1M labeled images (ImageNet)
Cost: $100K+ for labels, months of work
Limitation: Labels expensive, biased, task-specific
```

**Self-Supervised Learning:**
```
Use: 1B unlabeled images (free from internet!)
Cost: Just compute
Result: Better representations than supervised
```

## Core Idea

**Learn by comparing:**
```
Similar things → Close in embedding space
Different things → Far apart
```

**Example:**
```
Image of a cat:
  + Another view of same cat  → Pull together
  - Image of a dog           → Push apart
  - Image of a car           → Push apart
```

## Key Insight: Data Augmentation as Supervision

Two augmented views of the same image = **positive pair**
Views from different images = **negative pairs**

**Augmentations:**
```python
Original image
 ├─ View 1: Crop + flip + color jitter
 └─ View 2: Different crop + rotation + blur

View 1 and View 2 should be similar (positive pair)
```

## Major Approaches

### 1. SimCLR (Simple Framework for Contrastive Learning) ⭐

**Paper:** Chen et al., Google (2020)

**Algorithm:**
```
For each image in batch:
  1. Create 2 augmented views
  2. Encode both views with CNN → embeddings
  3. Apply projection head (MLP) → representations
  4. Contrastive loss: Pull positive pairs together,
                       push negative pairs apart
```

**NT-Xent Loss (Normalized Temperature-scaled Cross Entropy):**
```
For positive pair (i, j):
  sim(i,j) = cosine_similarity(z_i, z_j)

  L = -log(exp(sim(i,j)/τ) / Σ_k exp(sim(i,k)/τ))
```

**Key Components:**
- **Large batch sizes**: 256-8192 (more negatives → better)
- **Strong augmentation**: Crop, flip, color jitter, blur
- **Projection head**: Nonlinear MLP after encoder
- **Temperature τ**: 0.1-0.5 (controls concentration)

**Results:**
- ImageNet top-1: 76.5% (linear eval) without any labels!
- Matches supervised ResNet-50 with 1% of labels
- Works better with more data (1B images → 80%+)

**Why It Works:**
- Augmentations create implicit labels
- Large batch = many negatives
- Projection head removes task-specific info
- Learned representations transfer well

### 2. MoCo (Momentum Contrast) 🔄

**Paper:** He et al., Facebook (2020)

**Problem with SimCLR:** Needs huge batches (8192) → huge memory

**Solution:** Momentum queue + momentum encoder

**Algorithm:**
```
Maintain:
  - Query encoder (trained normally)
  - Key encoder (momentum update)
  - Queue of K previous keys (65536)

For each query:
  1. Encode query with query encoder
  2. Encode positive key with key encoder
  3. Contrast against queue of negatives
  4. Update queue and key encoder (momentum)
```

**Momentum Update:**
```
θ_key ← m · θ_key + (1-m) · θ_query

m = 0.999 (very slow update)
```

**Benefits:**
- Large negatives (65K) without large batch
- Memory-efficient (queue only stores embeddings)
- More stable than SimCLR
- Better transfer learning

**Results:**
- ImageNet: 76.7% (linear eval)
- Better than SimCLR with 8× smaller batch
- Faster convergence

### 3. BYOL (Bootstrap Your Own Latent) 🎯

**Paper:** Grill et al., DeepMind (2020)

**Revolutionary Insight:** No negative pairs needed!

**Problem Solved:** Mode collapse without negatives?

**Solution:** Asymmetric architecture + momentum + stop-gradient

**Algorithm:**
```
Two networks:
  - Online network (updated by gradients)
  - Target network (momentum update)

For each image:
  1. Create 2 views
  2. Online encodes view1 → prediction
  3. Target encodes view2 → target (stop-gradient!)
  4. Minimize MSE(prediction, target)
  5. Update target with momentum
```

**Why It Doesn't Collapse:**
- Asymmetry (predictor in online only)
- Stop-gradient on target
- Momentum update
- Prediction task

**Benefits:**
- No negative pairs needed
- No large batches required
- Simpler than contrastive methods
- Better performance

**Results:**
- ImageNet: 79.6% (linear eval)
- Beats SimCLR and MoCo
- Works with batch size 256

### 4. CLIP (Contrastive Language-Image Pre-training) 🌐

**Paper:** Radford et al., OpenAI (2021)

**Breakthrough:** Joint vision-language learning at scale

**Data:** 400M (image, text) pairs from internet

**Algorithm:**
```
Batch of N (image, text) pairs:
  1. Encode images with Vision Transformer
  2. Encode texts with Text Transformer
  3. Compute N×N similarity matrix
  4. Maximize diagonal (correct pairs)
     Minimize off-diagonal (wrong pairs)
```

**Contrastive Loss:**
```
For each (image_i, text_i):
  Maximize: similarity(image_i, text_i)
  Minimize: similarity(image_i, text_j) for j≠i
            similarity(image_j, text_i) for j≠i
```

**Zero-Shot Classification:**
```
Text prompts: "a photo of a {class}"
  Classes: [dog, cat, car, ...]

Classify:
  1. Encode image
  2. Encode all text prompts
  3. Pick highest similarity
```

**Results:**
- **Zero-shot ImageNet: 76.2%** (no fine-tuning!)
- Matches ResNet-50 supervised (76.1%)
- Generalizes to any image classification task
- Foundation of Stable Diffusion, DALL-E

**Impact:**
- Unified vision and language
- Enables text-to-image generation
- Powers Stable Diffusion prompts
- Billion-dollar technology

### 5. SwAV (Swapped Assignment Views)

**Paper:** Caron et al., Facebook (2020)

**Idea:** Clustering + contrastive learning

**Benefits:**
- No negative pairs
- Works with small batches
- Multi-crop augmentation

## Comparison

| Method | Negatives | Batch Size | ImageNet Acc | Key Innovation |
|--------|-----------|------------|--------------|----------------|
| SimCLR | ✅ Many | 8192 | 76.5% | Large batch + strong aug |
| MoCo v2 | ✅ Queue | 256 | 76.7% | Momentum queue |
| BYOL | ❌ None | 256 | 79.6% | No negatives needed |
| SwAV | ❌ None | 256 | 78.5% | Clustering approach |
| CLIP | ✅ In-batch | 32K | 76.2% zero-shot | Vision-language |

## Why Contrastive Learning Works

### 1. InfoNCE Loss Maximizes Mutual Information
```
Contrastive loss ≈ lower bound on mutual information
Between views: I(view1; view2)
```

### 2. Learns Invariances
```
Invariant to:
  - Crops, rotations, color changes
  - Viewpoint, lighting
  - Non-semantic variations

Preserves:
  - Object identity
  - Semantic content
  - High-level structure
```

### 3. Emergent Properties
```
Learned representations:
  - Separate semantic classes
  - Cluster similar objects
  - Transfer to many tasks
  - Better than supervised for transfer!
```

## Data Augmentations

Critical for contrastive learning success:

**Random Crop (Important!):**
```
Crop 8%-100% of image
Forces model to recognize objects from partial views
```

**Color Jitter:**
```
Brightness, contrast, saturation, hue
Makes model color-invariant
```

**Gaussian Blur:**
```
σ ~ Uniform(0.1, 2.0)
Removes high-frequency details
```

**Random Grayscale:**
```
10% probability → grayscale
Further color invariance
```

**Composition:**
```
view = RandomCrop
       + ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
       + GaussianBlur(p=0.5)
       + RandomGrayscale(p=0.2)
       + RandomHorizontalFlip
```

## Applications

### 1. Pre-training for Computer Vision
```
Pre-train: SimCLR on ImageNet (unlabeled)
Fine-tune: Linear classifier for specific task
Result: Matches supervised with 1% of labels
```

### 2. Few-Shot Learning
```
Pre-train: Contrastive learning (millions of images)
Fine-tune: 5-10 examples per class
Result: 80%+ accuracy (vs 30% from scratch)
```

### 3. Transfer Learning
```
Pre-train: CLIP on 400M image-text pairs
Use: Zero-shot on ANY image classification
Result: Competitive with task-specific supervised
```

### 4. Medical Imaging
```
Problem: Very few labeled X-rays (expensive expert labels)
Solution: Contrastive pre-training on unlabeled scans
Result: 10-20% accuracy improvement
```

### 5. Text-to-Image (Stable Diffusion)
```
CLIP embeddings guide image generation
Text prompt → CLIP text encoder → embedding
Diffusion model generates image matching embedding
```

## Practical Training Tips

### For SimCLR
```
Batch size: 256-8192 (larger is better)
Epochs: 800-1000
Optimizer: LARS or AdamW
LR: 0.3 × batch_size / 256
Temperature: 0.5 (tune 0.1-1.0)
Augmentation: Strong (crop + jitter + blur)
```

### For MoCo
```
Batch size: 256
Queue size: 65536
Momentum: 0.999
Temperature: 0.07
```

### For BYOL
```
Batch size: 256-4096
Momentum: 0.996-0.999 (cosine schedule)
Predictor: 2-layer MLP
No temperature needed
```

## Common Pitfalls

❌ **Too weak augmentation** → Model shortcuts (e.g., chromatic aberration)
✅ Strong augmentation forces semantic learning

❌ **Too small batch** (SimCLR) → Not enough negatives
✅ Use MoCo/BYOL or increase batch size

❌ **No projection head** → Worse representations
✅ Add 2-layer MLP projection head

❌ **Wrong temperature** → Collapse or slow learning
✅ Tune τ ∈ [0.1, 0.5]

## Key Takeaways

1. **Self-supervised >> supervised** for transfer learning
2. **CLIP** is the most impactful (vision + language)
3. **Data scale matters** more than architecture
4. **Strong augmentation** is critical
5. **BYOL** simplest to implement (no negatives)
6. **Contrastive learning** is the future of pre-training

## Running the Example

```bash
cargo run --package contrastive-learning
```

This demonstrates:
- SimCLR-style contrastive loss
- Positive/negative pair sampling
- Temperature scaling effects
- Embedding similarity visualization

## References

- **SimCLR:** Chen et al. (2020) - "A Simple Framework for Contrastive Learning"
- **MoCo:** He et al. (2020) - "Momentum Contrast for Unsupervised Visual Representation Learning"
- **BYOL:** Grill et al. (2020) - "Bootstrap Your Own Latent"
- **CLIP:** Radford et al. (2021) - "Learning Transferable Visual Models From Natural Language Supervision"
- **SwAV:** Caron et al. (2020) - "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments"

## Impact

Contrastive learning enabled:
- ✅ **Learning from unlabeled internet data** (billions of images)
- ✅ **CLIP** → Text-to-image revolution (Stable Diffusion, DALL-E)
- ✅ **Better representations** than supervised learning
- ✅ **Foundation models** (GPT-4V, Flamingo)
- ✅ **Medical AI** (limited labels)

**The paradigm shift:** Supervised learning → Self-supervised pre-training + fine-tuning

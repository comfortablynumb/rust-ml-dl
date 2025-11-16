# Masked Modeling 🎭

**Self-supervised learning through reconstruction**: Learn by predicting masked portions of input. Powers BERT, GPT, and Masked Autoencoders (MAE).

## Overview

Masked modeling is one of the most successful self-supervised learning approaches:
- **BERT**: Masked Language Modeling → NLP revolution
- **MAE** (Masked Autoencoders): 75% masking → state-of-the-art vision
- **GPT**: Autoregressive masking → ChatGPT foundation

## Core Idea

**Learn by filling in the blanks:**
```
Input:  "The [MASK] sat on the [MASK]"
Task:   Predict masked words
Answer: "The cat sat on the mat"
```

**Why It Works:**
- Forces understanding of context
- Learns bidirectional relationships
- Scales to billions of unlabeled examples
- No human labels needed!

## Major Approaches

### 1. BERT (Masked Language Modeling) 📚

**Paper:** Devlin et al., Google (2018)

**Revolution:** Bidirectional pre-training for NLP

**Algorithm:**
```
1. Take text: "The cat sat on the mat"
2. Randomly mask 15% of tokens: "The [MASK] sat on the [MASK]"
3. Predict masked tokens using bidirectional Transformer
4. Loss: CrossEntropy on masked positions only
```

**Masking Strategy:**
```
For each selected token (15% of all tokens):
  - 80%: Replace with [MASK] token
  - 10%: Replace with random token
  - 10%: Keep original (unchanged)
```

**Why This Strategy?**
- 80% [MASK]: Main training signal
- 10% random: Prevents relying on [MASK] token
- 10% unchanged: Handles mismatch with fine-tuning

**Training Data:**
- BooksCorpus (800M words)
- English Wikipedia (2,500M words)
- Total: 3.3B words unlabeled

**Architecture:**
- BERT-base: 12 layers, 768 hidden, 12 heads, 110M params
- BERT-large: 24 layers, 1024 hidden, 16 heads, 340M params

**Pre-training Tasks:**
1. **Masked LM**: Predict masked tokens (main task)
2. **Next Sentence Prediction**: Is sentence B next after A? (helps with reasoning)

**Results:**
- **GLUE benchmark**: 80.5% (7.7% over previous SOTA)
- **SQuAD**: 93.2 F1 (question answering)
- **Named Entity Recognition**: 92.8 F1

**Impact:**
- Transformed NLP (pre-BERT → post-BERT era)
- Foundation for: RoBERTa, ALBERT, DistilBERT, ELECTRA
- Used in: Google Search, translation, sentiment analysis

### 2. RoBERTa (Robustly Optimized BERT) 💪

**Paper:** Liu et al., Facebook (2019)

**Improvements over BERT:**
- Remove Next Sentence Prediction (hurts performance!)
- Train longer (100K → 500K steps)
- Larger batches (256 → 8K)
- More data (16GB → 160GB text)
- Dynamic masking (different mask each epoch)

**Results:**
- Matches or beats BERT on all tasks
- GLUE: 88.5% (vs BERT: 80.5%)
- SQuAD: 94.6 F1 (vs BERT: 93.2)

**Key Insight:** More data + more compute + better training > architectural changes

### 3. MAE (Masked Autoencoders for Vision) 🖼️

**Paper:** He et al., Facebook (2021)

**Breakthrough:** BERT-style masking for images

**Algorithm:**
```
1. Divide image into patches (16×16 pixels)
2. Randomly mask 75% of patches (very high!)
3. Encode visible patches with Transformer
4. Decode to reconstruct masked patches
5. Loss: MSE on masked patches only
```

**Architecture:**
```
Input: 224×224 image → 14×14 = 196 patches

Encoder (ViT):
  - Input: 25% visible patches (49 patches)
  - 24 Transformer layers
  - No mask tokens in encoder! (efficiency)

Decoder:
  - Input: Encoded patches + mask tokens
  - 8 Transformer layers (lightweight)
  - Reconstruct: 196 patches (all)

Loss: MSE(reconstructed_pixels, original_pixels) on masked patches
```

**Why 75% Masking?**
- Low masking (15%): Too easy, trivial interpolation
- High masking (75%): Forces semantic understanding
- Images have redundancy → need aggressive masking

**Reconstruction Examples:**
```
Original image: [Full cat image]
Masked (75%):   [Only 25% visible patches]
Reconstructed:  [Complete cat image, slight blur]
```

**Results:**
- **ImageNet fine-tuning**: 87.8% top-1 (SOTA for ViT)
- **Transfer learning**: Best results on detection, segmentation
- **Training efficiency**: 3× faster than contrastive (no negatives)

**Why It Works for Vision:**
- Images have high redundancy (neighboring pixels similar)
- High masking ratio forces holistic understanding
- Asymmetric encoder-decoder (efficient)
- Pixel reconstruction = strong training signal

### 4. GPT (Autoregressive Masking) 🤖

**Idea:** Predict next token (causal masking)

**Difference from BERT:**
```
BERT: Bidirectional, mask random tokens
  "The [MASK] sat on the mat" → Can see both directions

GPT: Unidirectional, predict next token
  "The cat sat" → Predict "on"
  "The cat sat on" → Predict "the"
```

**Architecture:**
- Decoder-only Transformer
- Causal attention mask (can't see future)
- Trained on next-token prediction

**Progression:**
- GPT-1 (2018): 117M params
- GPT-2 (2019): 1.5B params
- GPT-3 (2020): 175B params
- GPT-4 (2023): ~1.7T params (rumored)

**Why Autoregressive?**
- Natural for text generation
- No special mask tokens needed
- Enables zero-shot prompting
- Foundation for ChatGPT

### 5. BEiT (BERT Pre-training of Image Transformers)

**Paper:** Bao et al., Microsoft (2021)

**Idea:** Discrete token prediction for images

**Algorithm:**
```
1. Tokenize image with dVAE (like VQ-VAE)
   Image → Discrete visual tokens
2. Mask 40% of tokens
3. Predict masked token IDs (classification)
4. Loss: CrossEntropy (like BERT for text)
```

**vs MAE:**
- BEiT: Predict discrete tokens (classification)
- MAE: Reconstruct pixels (regression)
- MAE is simpler and works better

## Comparison

| Method | Domain | Mask Ratio | Prediction Target | Use Case |
|--------|--------|-----------|-------------------|----------|
| BERT | NLP | 15% | Masked tokens | Text understanding |
| RoBERTa | NLP | 15% | Masked tokens | Improved BERT |
| GPT | NLP | Causal | Next token | Text generation |
| MAE | Vision | 75% | Masked pixels | Image understanding |
| BEiT | Vision | 40% | Discrete tokens | Image understanding |

## Why Masked Modeling Works

### 1. Self-Supervised Signal
```
Billions of unlabeled examples:
  - Text: Entire internet (trillions of tokens)
  - Images: Millions of unlabeled images

No expensive human labels needed!
```

### 2. Forces Deep Understanding
```
Can't predict masked token without understanding:
  - Syntax (grammar rules)
  - Semantics (meaning)
  - Context (surrounding information)
  - World knowledge
```

### 3. Pretext Task → Transfer Learning
```
Pre-train: Masked modeling on unlabeled data
Fine-tune: Small labeled dataset for specific task
Result: 10-100× less labeled data needed
```

### 4. Bi-directional Context (BERT/MAE)
```
Masked token sees:
  - Left context: "The cat sat"
  - Right context: "on the mat"
  - Both: Better representations than unidirectional
```

## Training Details

### BERT Pre-training
```
Data: 3.3B words (BooksCorpus + Wikipedia)
Batch size: 256 sequences
Sequence length: 512 tokens
Steps: 1M
Hardware: 64 TPU v3 chips, 4 days
Optimizer: AdamW
LR: 1e-4 with warmup
```

### MAE Pre-training
```
Data: ImageNet-1K (1.3M images, unlabeled)
Batch size: 4096
Epochs: 800
Mask ratio: 75%
Hardware: 128 TPU v3 cores
Optimizer: AdamW
LR: 1.5e-4 with warmup + cosine decay
```

## Applications

### 1. Transfer Learning (Main Use Case)
```
Pre-train: BERT on Wikipedia (unlabeled)
Fine-tune: 1K labeled examples for sentiment
Result: 85%+ accuracy (vs 60% from scratch)
```

### 2. Few-Shot Learning
```
Pre-train: MAE on ImageNet (no labels)
Fine-tune: 1% ImageNet labels
Result: 73.5% accuracy (supervised needs 100% labels for 76%)
```

### 3. Domain Adaptation
```
Pre-train: BERT on biomedical papers
Fine-tune: Clinical NER
Result: 90%+ F1 (vs 70% with general BERT)
```

### 4. Zero-Shot (GPT-3)
```
Prompt: "Translate English to French: Hello → "
Output: "Bonjour"
No fine-tuning! Just prompt engineering
```

## Modern Developments

### Scaling Laws
```
Bigger model + More data = Better performance
BERT: 110M params → RoBERTa: 355M → GPT-3: 175B
```

### Multi-Modal (FLAVA, Flamingo)
```
Masked modeling on:
  - Images (MAE-style)
  - Text (BERT-style)
  - Image-text pairs (joint)
Result: Unified vision-language model
```

### Efficient Masking (MaskGIT, MaskDINO)
```
Applications: Image generation, object detection
Faster than autoregressive (GPT)
Competitive with diffusion models
```

## Best Practices

### Masking Strategy
```
NLP: 15% (BERT)
  - Low redundancy in text
  - Higher masking → too hard

Vision: 75% (MAE)
  - High redundancy in images
  - Need aggressive masking
```

### Architecture
```
Encoder-Decoder (MAE):
  - Heavy encoder, light decoder
  - Efficient (only encode visible patches)

Encoder-Only (BERT):
  - Bidirectional Transformer
  - Same architecture for pre-train & fine-tune
```

### Data Augmentation
```
Text: None needed (masking is augmentation)
Vision: Crop, flip, color jitter (like contrastive)
```

## Key Takeaways

1. **BERT revolutionized NLP** - bidirectional pre-training
2. **MAE brought BERT to vision** - 75% masking
3. **GPT uses autoregressive masking** - next-token prediction
4. **Scales with data** - more unlabeled data = better
5. **Transfer learning** - pre-train once, fine-tune for many tasks

## Running the Example

```bash
cargo run --package masked-modeling
```

This demonstrates:
- BERT-style token masking
- Reconstruction loss calculation
- Masking strategies (random, block)
- Prediction accuracy

## References

- **BERT:** Devlin et al. (2018) - "BERT: Pre-training of Deep Bidirectional Transformers"
- **RoBERTa:** Liu et al. (2019) - "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
- **MAE:** He et al. (2021) - "Masked Autoencoders Are Scalable Vision Learners"
- **BEiT:** Bao et al. (2021) - "BEiT: BERT Pre-Training of Image Transformers"
- **GPT-3:** Brown et al. (2020) - "Language Models are Few-Shot Learners"

## Impact

Masked modeling enabled:
- ✅ **BERT → NLP revolution** (Google Search uses BERT)
- ✅ **GPT-3 → ChatGPT** (175B params, conversational AI)
- ✅ **MAE → Vision pre-training** (efficient, scalable)
- ✅ **Transfer learning** (10-100× less labeled data)
- ✅ **Foundation models** (pre-train once, use everywhere)

**The paradigm:** Masked pre-training on unlabeled data → fine-tune on labeled task-specific data

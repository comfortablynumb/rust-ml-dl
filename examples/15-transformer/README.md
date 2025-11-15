# Transformer Architecture Example

This example demonstrates the Transformer, the revolutionary architecture that powers GPT, BERT, ChatGPT, and most modern AI systems.

## Overview

Transformers use **attention mechanisms** to process sequences in parallel, eliminating the need for recurrence (RNNs) or convolution (CNNs).

**Key Paper:** "Attention is All You Need" (Vaswani et al., 2017)

## Running the Example

```bash
cargo run --package transformer
```

## Core Innovation: Attention

**Question:** How do we know which words are related?

**Example:**
```
"The animal didn't cross the street because it was too tired"
                                               ^^
```

What does "it" refer to?
- the animal ✓ (animals get tired)
- the street ✗ (streets don't get tired)

Attention mechanisms automatically learn these relationships!

## Self-Attention Mechanism

For each token, compute how much to "attend" to every other token.

### Algorithm

1. **Create Q, K, V vectors:**
   ```
   Q (Query): What am I looking for?
   K (Key): What do I contain?
   V (Value): What do I output?
   ```

2. **Compute attention scores:**
   ```
   scores = (Q · K^T) / √d_k
   ```

3. **Apply softmax:**
   ```
   weights = softmax(scores)
   ```

4. **Weighted sum:**
   ```
   output = weights · V
   ```

### Formula

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

## Multi-Head Attention

Use multiple attention mechanisms ("heads") in parallel:

```
Head 1: Syntactic relationships
Head 2: Semantic relationships
Head 3: Positional relationships
...
Head 8: Other patterns
```

**Benefits:**
- Learn different types of relationships
- More expressive representations
- Better performance

## Positional Encoding

**Problem:** Transformers have no built-in notion of order!

**Solution:** Add positional encodings to embeddings

```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**Properties:**
- Unique for each position
- Model can learn relative positions
- Works for any sequence length

## Architecture

### Encoder
```
Input → Embedding + Position
   ↓
Multi-Head Self-Attention
   ↓ (Add & Norm)
Feed-Forward Network
   ↓ (Add & Norm)
(Repeat 6-12 times)
   ↓
Encoder Output
```

### Decoder
```
Output → Embedding + Position
   ↓
Masked Self-Attention (can't see future)
   ↓ (Add & Norm)
Cross-Attention (to encoder)
   ↓ (Add & Norm)
Feed-Forward Network
   ↓ (Add & Norm)
(Repeat 6-12 times)
   ↓
Output Probabilities
```

## Key Components

### Layer Normalization
- Normalizes activations
- Stabilizes training
- Enables deeper networks

### Residual Connections
```
output = Layer(x) + x
```
- Better gradient flow
- Easier training
- Enables very deep networks

### Feed-Forward Network
```
FFN(x) = ReLU(x·W₁ + b₁)·W₂ + b₂
```
- Typically: d_model → 4·d_model → d_model
- Example: 512 → 2048 → 512

## Transformer Variants

### BERT (Bidirectional Encoder Representations)
- **Architecture:** Encoder only
- **Training:** Masked language modeling
- **Use:** Text understanding, classification, QA
- **Size:** BERT-Base (110M), BERT-Large (340M)

### GPT (Generative Pre-trained Transformer)
- **Architecture:** Decoder only
- **Training:** Next token prediction
- **Use:** Text generation, chat, code
- **Evolution:**
  - GPT-1: 117M parameters (2018)
  - GPT-2: 1.5B parameters (2019)
  - GPT-3: 175B parameters (2020)
  - GPT-4: ~1.7T parameters (2023)

### T5 (Text-to-Text Transfer Transformer)
- **Architecture:** Full encoder-decoder
- **Approach:** Everything as text-to-text
- **Use:** Translation, summarization, QA
- **Size:** Up to 11B parameters

### Vision Transformer (ViT)
- Applies transformers to images
- Treats image patches as tokens
- Competitive with CNNs
- Used in CLIP, DALL-E

### CLIP (Contrastive Language-Image Pre-training)
- Learns vision-language connections
- Powers DALL-E, Stable Diffusion
- Zero-shot image classification

## Applications

### Natural Language Processing
- Machine translation
- Text generation (GPT)
- Question answering (BERT)
- Summarization (T5, BART)
- Sentiment analysis
- Named entity recognition

### Code & Programming
- Code completion (GitHub Copilot)
- Code generation (GPT-4, Codex)
- Bug detection
- Documentation generation

### Computer Vision
- Image classification (ViT)
- Object detection (DETR)
- Image generation (DALL-E)
- Image segmentation

### Multimodal
- Image captioning
- Visual question answering
- Text-to-image (DALL-E, Stable Diffusion)
- Text-to-video

### Speech & Audio
- Speech recognition (Whisper)
- Text-to-speech
- Music generation
- Audio classification

### Science & Research
- Protein folding (AlphaFold 2)
- Drug discovery
- Scientific paper generation
- Weather forecasting

## Why Transformers Won

### vs RNNs
- **Parallelization:** Process all tokens simultaneously (100x faster)
- **Long-range dependencies:** Direct connections, no vanishing gradients
- **Scalability:** Scale with compute and data

### vs CNNs
- **Global receptive field:** See entire sequence immediately
- **Flexible attention:** Learn any relationships
- **Better for sequences:** Natural fit for text, code

### Scaling Laws
```
More parameters + More data + More compute = Better performance
```

This simple relationship drove:
- GPT-3: 175B parameters
- GPT-4: ~1.7T parameters (estimated)
- Future: 10T+ parameters

## Computational Complexity

**Self-Attention:**
- Time complexity: O(n²·d)
- Space complexity: O(n²)

**Problem:** Quadratic in sequence length!

**Example:**
- 512 tokens → 262,144 attention values
- 2048 tokens → 4,194,304 attention values

**Solutions:**
- **Sparse attention:** Longformer, BigBird
- **Linear attention:** Performer, Linear Transformer
- **Sliding window:** Local attention only
- **Flash Attention:** Optimized CUDA kernels
- **Mixture of Experts:** Conditional computation

## Training Tips

1. **Warmup learning rate**
   - Linear increase for first N steps
   - Then cosine decay

2. **Adam optimizer**
   - β₁ = 0.9, β₂ = 0.98 (original paper)
   - ε = 1e-9

3. **Gradient clipping**
   - Clip norm to 1.0
   - Prevents exploding gradients

4. **Label smoothing**
   - Use 0.1 smoothing
   - Prevents overconfidence

5. **Dropout**
   - Apply to attention (0.1)
   - Apply to embeddings (0.1)

6. **Layer normalization**
   - Pre-LN often better than Post-LN

## Famous Models & Services

### OpenAI
- GPT-3/4
- ChatGPT
- DALL-E 2/3
- Codex (GitHub Copilot)

### Google
- BERT
- T5
- PaLM, PaLM 2
- Gemini

### Meta
- LLaMA, LLaMA 2
- RoBERTa
- BART

### Anthropic
- Claude, Claude 2

### Open Source
- BLOOM (176B)
- Falcon (180B)
- Mistral (7B)

## Historical Impact

**2017:** "Attention is All You Need" published
- Introduced Transformer architecture
- Original use: Machine translation

**2018:** BERT changes NLP
- Pre-training + fine-tuning paradigm
- State-of-the-art on 11 NLP tasks

**2018-2020:** GPT evolution
- GPT-1 → GPT-2 → GPT-3
- Emergent abilities at scale

**2020+:** Transformers everywhere
- Vision (ViT)
- Speech (Whisper)
- Code (Codex)
- Science (AlphaFold 2)
- Multimodal (CLIP, DALL-E)

**2022-2023:** ChatGPT era
- Conversational AI goes mainstream
- GPT-4, Claude, LLaMA 2
- AI assistants, coding tools

## Future Directions

- **Longer context:** 100K+ tokens (Claude 2, GPT-4 Turbo)
- **Multimodal:** Unified vision-language-audio models
- **Efficient transformers:** Reduce O(n²) complexity
- **Sparse models:** Mixture of Experts (1T+ parameters)
- **Agent systems:** Autonomous AI assistants

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [BERT](https://arxiv.org/abs/1810.04805) (Devlin et al., 2018)
- [GPT-3](https://arxiv.org/abs/2005.14165) (Brown et al., 2020)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

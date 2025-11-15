# Sequence-to-Sequence (Seq2Seq) Models

Encoder-decoder architecture for variable-length sequence transformation.

## Overview

Seq2Seq models revolutionized neural machine translation by enabling variable-length input → variable-length output transformations.

## Running

```bash
cargo run --package seq2seq
```

## Core Architecture

```
Input Sequence → [ENCODER] → Context → [DECODER] → Output Sequence
                     ↓                      ↑
                Hidden States ←────────── Attention
```

### Components

**Encoder (LSTM/GRU)**
- Processes input sequence left-to-right (or bidirectional)
- Outputs hidden state at each position
- Compresses sequence into representations

**Attention Mechanism**
- Decoder "looks back" at all encoder hidden states
- Computes weighted combination based on relevance
- Eliminates bottleneck of single context vector

**Decoder (LSTM/GRU)**
- Generates output one token at a time (autoregressive)
- Uses attention context + previous output
- Continues until <END> token or max length

## Evolution

### Basic Seq2Seq (2014)

```
Encoder → Single Context Vector → Decoder
```

**Problem**: Bottleneck! All input compressed to fixed-size vector.

### Attention-Based Seq2Seq (2015)

```
Encoder → All Hidden States
              ↓
          Attention ← Decoder state
              ↓
          Context (weighted sum)
```

**Improvement**: Decoder accesses all encoder positions, no bottleneck!

## Attention Mechanism

### Attention Formula

```
1. Compute scores:  score_i = f(decoder_state, encoder_state_i)
2. Normalize:       weights = softmax(scores)
3. Context vector:  context = Σ weights_i × encoder_state_i
```

### Attention Types

**Additive (Bahdanau)**
```
score(s, h) = v^T · tanh(W_s·s + W_h·h)
```

**Multiplicative (Luong)**
```
score(s, h) = s^T · W · h
```

**Dot-Product**
```
score(s, h) = s^T · h
```

## Training: Teacher Forcing

**Problem**: Feeding wrong predictions compounds errors

**Solution**: Use ground truth (not predictions) as next input during training

```
Without Teacher Forcing (error compounds):
  Target: "Je suis bon"
  Predict "Le" → Feed "Le" → Predict "chat" (wrong trajectory!)

With Teacher Forcing (stable training):
  Target: "Je suis bon"
  Predict "Le" → Feed "Je" (truth) → Learn correctly
```

**Ratio**: Typically 80-100% during training, 0% at inference

## Decoding Strategies

### 1. Greedy Decoding

```
At each step: Pick argmax P(word | previous)
```

✅ Fast (O(n))
❌ Suboptimal (local choices)

### 2. Beam Search

```
Keep top-k hypotheses (beam) at each step
Explore multiple paths simultaneously
```

**Example** (beam_size=2):
```
Step 1: ["The", "A"]
Step 2: ["The cat", "A dog"]  (top 2 by cumulative prob)
Step 3: ["The cat sat", "A dog ran"]
...
```

**Trade-off**:
- beam=1: Greedy (fastest)
- beam=5: Good balance (typical)
- beam=10: Better quality, slower

✅ Better quality
❌ Slower (O(k·n))

### 3. Sampling (Top-k/Top-p)

```
Sample from probability distribution
```

✅ Creative, diverse outputs
❌ Less deterministic
**Use case**: Chat, story generation

## Applications

### 1. Machine Translation

```
English → French
"I love Rust" → "J'aime Rust"
```

- **Google Translate** (pre-2017): Seq2Seq + Attention
- Typical: 2-4 layer bidirectional encoder, 2-4 layer decoder
- Dataset: Millions of parallel sentence pairs

### 2. Text Summarization

```
Long Article → Short Summary
```

- Encoder: Process full document
- Decoder: Generate concise summary
- Abstractive vs extractive

### 3. Dialogue/Chatbots

```
User: "How are you?"
Bot: "I'm doing well, thanks!"
```

- Trained on conversation pairs
- Modern: Replaced by GPT-style models

### 4. Image Captioning

```
Image (via CNN) → Text Description
```

- Encoder: CNN (ResNet) → 2048-d vector
- Decoder: LSTM generates caption
- Example: "A cat sitting on a mat"

### 5. Speech Recognition

```
Audio Waveform → Text Transcription
```

- Listen, Attend and Spell (LAS)
- Encoder: Process audio frames
- Decoder: Generate text

## Seq2Seq vs Transformer

| Aspect | Seq2Seq (LSTM) | Transformer |
|--------|----------------|-------------|
| **Training** | Sequential (slow) | Parallel (100× faster) |
| **Long-range** | Difficult (vanishing gradients) | Excellent (self-attention) |
| **Memory** | O(n) | O(n²) quadratic |
| **Data needs** | Works with <1M examples | Needs >10M examples |
| **Deployment** | Lighter, streaming-friendly | Heavier, batch-oriented |

## When to Use Seq2Seq

✅ **Limited compute/memory**
✅ **Small datasets** (< 1M examples)
✅ **Streaming/online** processing
✅ **Edge deployment** (mobile, IoT)

## When to Use Transformer

✅ **Large datasets** (> 10M examples)
✅ **Offline batch** processing
✅ **SOTA performance** needed
✅ **GPU/TPU** training available

## Metrics

**BLEU (Bilingual Evaluation Understudy)**
- Precision of n-grams (1-4 grams)
- Scale: 0-100 (higher better)
  - 30-40: Understandable translation
  - 40-50: High-quality translation
  - 50-60: Near-human (multiple references)

**ROUGE (Recall-Oriented)**
- Used for summarization
- Measures recall of n-grams

## Historical Impact

```
2014: Sutskever et al. - First successful neural MT
2015: Bahdanau et al. - Attention mechanism (+10 BLEU)
2016: Google GNMT production deployment
2017: "Attention is All You Need" (Transformer)
2018+: Transformers dominate (BERT, GPT, T5)
```

## Modern Context (2024)

**Status**: Largely replaced by Transformers

**Still relevant for**:
- Streaming tasks (online speech recognition)
- Resource-constrained deployment
- Educational purposes (simpler than Transformer)
- Specific domains with limited data

## Papers

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) (Sutskever et al., 2014)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (Bahdanau et al., 2015) - Attention
- [Effective Approaches to Attention-based NMT](https://arxiv.org/abs/1508.04025) (Luong et al., 2015)
- [Google's Neural Machine Translation System](https://arxiv.org/abs/1609.08144) (Wu et al., 2016)

## Key Takeaways

✓ Enables variable-length sequence transformation
✓ Attention eliminates bottleneck, crucial for long sequences
✓ Teacher forcing critical for stable training
✓ Beam search improves quality over greedy decoding
✓ Foundation for modern NLP (led to Transformers)
✓ Still useful for streaming and embedded systems

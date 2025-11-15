# Attention Mechanisms Example

This example explains **Attention Mechanisms**, the foundational innovation powering Transformers, GPT, BERT, and modern AI.

## Overview

Attention allows models to focus on relevant parts of the input, solving the context bottleneck problem in sequence models.

## Running the Example

```bash
cargo run --package attention
```

## Core Concept

### The Problem

```
RNN/LSTM: "The cat sat on the mat" → single vector h
                                       ↑
                            Must capture everything!

Problem: Information bottleneck for long sequences
```

### The Solution: Attention

```
Compute relevance of each word dynamically:
Weights: [0.05, 0.30, 0.50, 0.10, 0.03, 0.02]
               ↑     ↑
          Focus on "cat" and "sat"!
```

## Attention Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
• Q: Query ("what am I looking for?")
• K: Keys ("what do I have?")
• V: Values ("actual information")
• √d_k: Scaling factor
```

## Key Variants

### Self-Attention

Sequence attends to itself:

```
"The animal didn't cross the street because it was too tired"
                                                    ↑
Attention weights for "it": [0.02, 0.62, ... ]
                                   ↑
                             Points to "animal"!
```

### Multi-Head Attention

```
Multiple parallel attention patterns:
• Head 1: Syntax (subject-verb)
• Head 2: Semantics (meaning)
• Head 3: Position (nearby words)
• Head 4: Long-range dependencies

Concatenate all → richer representation
```

### Cross-Attention

Attend to different sequence:

```
Translation:
Source: "I love cats" → K, V
Target: "J'aime" → Q

Decoder attends to source!
```

## Applications

- **Machine Translation**: Align source/target words
- **Image Captioning**: Attend to image regions
- **Question Answering**: Find relevant context
- **Document Classification**: Identify important sentences

## Historical Impact

**2014:** Bahdanau attention (first successful attention)
**2015:** Luong attention (simpler, faster)
**2017:** Transformer (attention-only, no RNNs)
**2018+:** Dominates NLP (BERT, GPT, T5)
**2021+:** Beyond NLP (ViT, DALL-E, multimodal)

## Further Reading

- [Neural Machine Translation](https://arxiv.org/abs/1409.0473) (Bahdanau et al., 2014)
- [Effective Approaches](https://arxiv.org/abs/1508.04025) (Luong et al., 2015)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)

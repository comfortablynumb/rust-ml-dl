# Recurrent Neural Network (RNN) Example

This example demonstrates RNNs for sequential data processing like text, time series, and speech.

## Overview

RNNs process sequences by maintaining a hidden state (memory) that persists across time steps, allowing them to handle variable-length sequences and capture temporal dependencies.

## Running the Example

```bash
cargo run --package rnn
```

## Key Concepts

### Hidden State (Memory)

The hidden state `hₜ` carries information from previous time steps:

```
hₜ = tanh(W_xh·xₜ + W_hh·hₜ₋₁ + bₕ)
```

This is what makes RNNs "recurrent" - they reuse the same weights at each step while updating the hidden state.

### Sequence Processing Patterns

1. **Many-to-One**: Sentiment analysis (sequence → single label)
2. **One-to-Many**: Image captioning (image → sequence of words)
3. **Many-to-Many (sync)**: Video frame labeling
4. **Seq2Seq**: Machine translation (encoder-decoder)

## Challenges

### Vanishing Gradients

Gradients shrink exponentially during backpropagation through time, making it hard to learn long-term dependencies.

**Solutions:**
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Gradient clipping
- Better initialization

### Exploding Gradients

Gradients grow too large, causing instability.

**Solution:**
- Gradient clipping

## RNN Variants

### Simple RNN
- Fast but limited memory
- Good for short sequences

### LSTM
- Uses gates (forget, input, output)
- Excellent long-term memory
- More parameters, slower

### GRU
- Simpler than LSTM
- Good performance
- Faster than LSTM

## Applications

**NLP:**
- Machine translation
- Text generation
- Sentiment analysis
- Named entity recognition

**Time Series:**
- Stock prediction
- Weather forecasting
- Anomaly detection

**Speech:**
- Speech recognition
- Voice synthesis

**Video:**
- Action recognition
- Video captioning

## Modern Context

While RNNs were dominant for sequence tasks, **Transformers** (2017+) have largely replaced them in NLP:

- Parallel processing (faster)
- Better long-range dependencies
- Powers GPT, BERT, ChatGPT

**RNNs still useful for:**
- Streaming/online processing
- Limited resources
- Real-time applications

## Further Reading

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Sequence Models (Coursera)](https://www.coursera.org/learn/nlp-sequence-models)

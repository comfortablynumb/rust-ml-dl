# LSTM & GRU Example

This example explains **Long Short-Term Memory (LSTM)** and **Gated Recurrent Units (GRU)**, the two most important RNN variants that revolutionized sequence modeling.

## Overview

LSTM and GRU solved the **vanishing gradient problem** that plagued vanilla RNNs, enabling networks to learn long-term dependencies in sequential data.

## Running the Example

```bash
cargo run --package lstm-gru
```

## The Vanishing Gradient Problem

### Why Vanilla RNNs Fail

```
Standard RNN update: h_t = tanh(W · h_{t-1} + U · x_t)

Problem during backpropagation:
∂L/∂h_1 = ∂L/∂h_T · (∂h_T/∂h_{T-1}) · ... · (∂h_2/∂h_1)
                     ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
            Many multiplications through time

Since |∂h_t/∂h_{t-1}| ≤ 1 (tanh derivative),
gradients exponentially decay: (0.5)^50 ≈ 0

Result: Can't learn dependencies beyond ~10 timesteps!
```

### Example Where Vanilla RNN Fails

```
"The cat, which had been sleeping on the cozy mat, was hungry"
                                                        ↑
Need to predict "was" (singular) - must remember "cat" from 11 words ago

Vanilla RNN: ❌ Gradient from "was" to "cat" vanishes
LSTM/GRU: ✅ Maintains gradient flow through gates
```

## LSTM (Long Short-Term Memory)

### Key Innovation

Instead of a single hidden state, LSTM has:
1. **Cell state (C_t)**: Long-term memory highway
2. **Hidden state (h_t)**: Short-term working memory
3. **Three gates**: Control information flow

### Architecture

```
                    Cell State (C_t)
                         │
        ┌────────────────┼────────────────┐
        │                                 │
    Forget Gate      Input Gate      Output Gate
        │                │                │
    What to          What to          What to
    forget           remember         output
```

### LSTM Equations

```
Concatenate inputs:
combined = [h_{t-1}, x_t]

1. Forget gate (sigmoid 0-1):
   f_t = σ(W_f · combined + b_f)
   "How much of old memory to keep?"

2. Input gate:
   i_t = σ(W_i · combined + b_i)
   "How much new info to store?"

   Candidate values:
   C̃_t = tanh(W_C · combined + b_C)
   "What new info to potentially add?"

3. Update cell state:
   C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
         ↑                ↑
      Keep old        Add new

4. Output gate:
   o_t = σ(W_o · combined + b_o)
   h_t = o_t ⊙ tanh(C_t)
   "What subset of memory to output?"
```

### Why LSTM Solves Vanishing Gradients

**Direct path through cell state:**

```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

∂C_t/∂C_{t-1} = f_t (just element-wise multiplication!)

No repeated matrix multiplications through time!
Gradient highway: C_1 → C_2 → ... → C_T

If f_t ≈ 1: Information flows unchanged
Can remember for 100+ timesteps!
```

## GRU (Gated Recurrent Unit)

### Motivation

LSTM works great but:
- 4 weight matrices (W_f, W_i, W_o, W_C)
- 2 state vectors (C_t, h_t)
- Many parameters → slower training

**GRU simplifies:**
- 3 weight matrices (W_z, W_r, W_h)
- 1 state vector (h_t only)
- ~25% fewer parameters
- Similar performance!

### GRU Architecture

```
Only one state: h_t (no separate cell state)

Two gates:
1. Update gate (z_t): How much to update
2. Reset gate (r_t): How much past to use
```

### GRU Equations

```
1. Update gate:
   z_t = σ(W_z · [h_{t-1}, x_t] + b_z)
   "How much to update state?"

2. Reset gate:
   r_t = σ(W_r · [h_{t-1}, x_t] + b_r)
   "How much past state to use?"

3. Candidate state:
   h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)
                     ↑
            Reset modulates past info

4. Final state (linear interpolation):
   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
         ↑                      ↑
      Keep old              Use new

   If z_t = 0: h_t = h_{t-1} (keep old state)
   If z_t = 1: h_t = h̃_t (full update)
```

## LSTM vs GRU

| Feature | LSTM | GRU |
|---------|------|-----|
| **Parameters** | ~4× input×hidden | ~3× input×hidden |
| **Speed** | Slower | ~15-25% faster |
| **Memory** | More (2 states) | Less (1 state) |
| **Gates** | 3 (forget, input, output) | 2 (update, reset) |
| **Performance** | Slightly better on complex tasks | Comparable on most tasks |
| **Training time** | Longer | Shorter |

## When to Use Which?

### Use LSTM when:
- ✅ Very long sequences (100+ steps)
- ✅ Complex temporal patterns
- ✅ Large training dataset
- ✅ Maximum accuracy needed
- ✅ Examples: machine translation, video captioning, speech synthesis

### Use GRU when:
- ✅ Moderate sequences (10-100 steps)
- ✅ Limited training data
- ✅ Faster training needed
- ✅ Resource constraints
- ✅ Examples: sentiment analysis, text classification, simple time series

### Use Transformer when:
- ✅ Lots of data and compute
- ✅ Need parallelization
- ✅ Very long-range dependencies
- ✅ State-of-the-art performance
- ✅ Examples: GPT, BERT, modern NLP

### Use Vanilla RNN when:
- ✅ Very short sequences (< 10 steps)
- ✅ Educational purposes
- ✅ Extremely simple patterns

## Advanced Variants

### Bidirectional LSTM/GRU

Process sequence in both directions:

```
Forward:  The  cat  sat  →  →  →
Backward:  ←  ←  ← sat cat The

Output: [h_forward, h_backward] concatenated

Pros:
+ Full context (sees future)
+ Better for classification tasks

Cons:
- Can't use for generation
- 2× slower
- Needs full sequence upfront

Use cases:
✅ Named entity recognition
✅ Part-of-speech tagging
✅ Sentiment classification
❌ Text generation
❌ Real-time prediction
```

### Stacked/Deep LSTM

Multiple LSTM layers:

```
Layer 3: LSTM (high-level: semantics)
          ↑
Layer 2: LSTM (mid-level: phrases)
          ↑
Layer 1: LSTM (low-level: syntax)
          ↑
Input:   tokens
```

**Guidelines:**
- 1 layer: Simple tasks (90% of cases)
- 2 layers: Most NLP tasks
- 3-4 layers: Complex tasks (QA, summarization)
- 5+ layers: Usually worse (use Transformers)

## Training Best Practices

### 1. Gradient Clipping

```python
# Essential! Even though vanishing is solved, exploding can occur
if ||gradient|| > threshold:
    gradient *= threshold / ||gradient||

Typical threshold: 1.0 to 5.0
```

### 2. Weight Initialization

```
Forget gate bias: Initialize to 1.0 or 2.0
  ↑ Start by remembering, learn to forget

Other biases: Initialize to 0
Weights: Xavier/Glorot initialization
```

### 3. Dropout

```
✅ DO apply:
   - Between LSTM layers
   - On inputs to LSTM
   - Rate: 0.2 - 0.5

❌ DON'T apply:
   - On recurrent connections!
   - Causes the network to forget
```

### 4. Optimizer Choice

```
Adam: Great default (lr = 0.001)
SGD + momentum: Better final performance (lr = 0.01)
RMSprop: Good for RNNs (lr = 0.001)

Always use learning rate decay!
```

### 5. Batch Processing

```
Problem: Variable sequence lengths

Solution 1 - Padding:
seq1: [1, 2, 3, 0, 0]  ← pad with 0
seq2: [4, 5, 6, 7, 8]
mask: [1, 1, 1, 0, 0]  ← ignore padded

Solution 2 - Packing (better):
Pack → Process → Unpack
Saves computation on padding
```

## Applications

### Natural Language Processing

**Machine Translation (pre-2017):**
- Encoder-decoder LSTM
- Google Neural Machine Translation (2016)
- 60% error reduction

**Sentiment Analysis:**
- Bidirectional LSTM
- IMDB: 89% accuracy
- Twitter: 85% accuracy

**Named Entity Recognition:**
- Bidirectional LSTM + CRF
- CoNLL-2003: 91% F1

### Speech Recognition

**Acoustic Modeling:**
- Deep LSTM (5-7 layers)
- Google Voice Search (2015)
- 49% word error reduction

**Speech Synthesis:**
- WaveNet alternatives
- Fast, high-quality TTS

### Time Series Forecasting

**Stock Prediction:**
- GRU often outperforms LSTM
- Faster, similar accuracy
- Captures momentum

**Weather Forecasting:**
- Stacked LSTM
- Multi-day predictions
- Seasonal patterns

**Energy Load:**
- Hourly electricity demand
- LSTM: 3-5% MAPE
- Better than ARIMA

### Video Analysis

**Action Recognition:**
- CNN (frame features) + LSTM (temporal)
- UCF101: 94% accuracy
- YouTube-8M: 84% GAP

**Video Captioning:**
- CNN encoder + LSTM decoder
- Generates descriptions
- MSVD dataset

### Music Generation

**Bach Chorales:**
- LSTM learns harmony
- 4-voice polyphony
- Convincing compositions

**Jazz Improvisation:**
- Learns melodic patterns
- Real-time generation

## Modern Context

### Why Transformers Replaced LSTM/GRU (mostly)

```
Transformers win on:
✅ Parallelization: Process all tokens at once
✅ Long-range dependencies: Direct attention
✅ Scalability: Bigger = better
✅ Transfer learning: Pre-training works great

LSTM/GRU advantages:
✅ O(1) memory vs O(n²) for Transformer
✅ Streaming data (online processing)
✅ Lower latency
✅ Fewer parameters
✅ Better with small data
```

### Where LSTM/GRU Still Excel

**Real-time/Streaming:**
- Live speech recognition
- Online anomaly detection
- Sensor data processing

**Resource-Constrained:**
- Mobile devices
- Edge computing
- Embedded systems

**Small Datasets:**
- Domain-specific applications
- Limited training data
- Fewer parameters to tune

**Time Series:**
- Financial data
- IoT sensors
- Industrial monitoring

## Historical Milestones

**1997:** LSTM introduced (Hochreiter & Schmidhuber)
- Solved vanishing gradients
- Enabled long-term memory

**2005-2010:** Speech recognition breakthroughs
- Alex Graves' work
- CTC loss for sequence alignment

**2014:** GRU introduced (Cho et al.)
- Simpler alternative
- Comparable performance

**2014:** Sequence-to-sequence models
- Sutskever et al.
- Machine translation revolution

**2015-2016:** Peak LSTM era
- Google Translate
- State-of-the-art NLP
- Most popular sequence model

**2017:** Transformer introduced
- "Attention is All You Need"
- Beginning of LSTM decline in NLP

**2020+:** Niche applications
- Still relevant for streaming
- Edge devices
- Time series

## Performance Comparison

### Computational Complexity

| Model | Memory | Time per step | Parallelizable |
|-------|--------|---------------|----------------|
| RNN | O(h) | O(h²) | ❌ No |
| LSTM | O(2h) | O(4h²) | ❌ No |
| GRU | O(h) | O(3h²) | ❌ No |
| Transformer | O(n×h) | O(n²×h) | ✅ Yes |

Where:
- h = hidden size
- n = sequence length

**LSTM better when:** n >> h (long sequences, small hidden)
**Transformer better when:** h >> n (short sequences, large hidden)

## Further Reading

- [LSTM paper](http://www.bioinf.jku.at/publications/older/2604.pdf) (Hochreiter & Schmidhuber, 1997)
- [GRU paper](https://arxiv.org/abs/1406.1078) (Cho et al., 2014)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) (Chris Olah)
- [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (Andrej Karpathy)
- [Empirical Evaluation of Gated RNNs](https://arxiv.org/abs/1412.3555) (Chung et al., 2014)

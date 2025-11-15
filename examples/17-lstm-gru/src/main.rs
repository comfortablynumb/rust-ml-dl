//! # LSTM & GRU: Long Short-Term Memory and Gated Recurrent Units
//!
//! This example explains LSTM and GRU, the two most important RNN variants that
//! solved the vanishing gradient problem and enabled effective long-term memory.
//!
//! ## The Problem with Vanilla RNNs
//!
//! **Vanishing Gradients:**
//! ```
//! Standard RNN: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)
//!
//! Problem:
//! • Gradients multiply through time: ∂L/∂h_1 = ∂L/∂h_T · ∂h_T/∂h_{T-1} · ... · ∂h_2/∂h_1
//! • tanh derivative ≤ 1, often much smaller
//! • After 10+ steps: gradient → 0 (vanishing)
//! • Can't learn long-term dependencies!
//! ```
//!
//! **Example Failure:**
//! ```
//! "The cat, which was sitting on the mat that was placed near the window, was hungry"
//!                                                                          ↑
//! Predict "was" - needs to remember "cat" from 15 words ago
//! Vanilla RNN: ❌ Forgets "cat"
//! LSTM/GRU: ✅ Remembers "cat"
//! ```
//!
//! ## LSTM: Long Short-Term Memory (1997)
//!
//! **Key Innovation:** Explicit memory cell with gates
//!
//! ### Architecture
//!
//! ```
//! LSTM has TWO states:
//! • Cell state (C_t): Long-term memory highway
//! • Hidden state (h_t): Short-term output
//!
//! Three gates control information flow:
//! 1. Forget gate (f_t): What to forget from memory
//! 2. Input gate (i_t): What new info to store
//! 3. Output gate (o_t): What to output
//! ```
//!
//! ### Mathematical Formulation
//!
//! ```
//! Input: x_t, previous hidden h_{t-1}, previous cell C_{t-1}
//!
//! 1. Forget gate (what to forget):
//!    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
//!
//! 2. Input gate (what to add):
//!    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
//!    C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  ← candidate values
//!
//! 3. Update cell state (memory update):
//!    C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
//!         ↑                ↑
//!    forget old       add new
//!
//! 4. Output gate (what to output):
//!    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
//!    h_t = o_t ⊙ tanh(C_t)
//!
//! Where:
//! • σ = sigmoid function (0 to 1, acts as gate)
//! • ⊙ = element-wise multiplication
//! • tanh = hyperbolic tangent (-1 to 1)
//! ```
//!
//! ### Why LSTM Works
//!
//! **Gradient Highway:**
//! ```
//! ∂C_t/∂C_{t-1} = f_t (just multiplication, no repeated squashing!)
//!
//! • Cell state has direct path: C_1 → C_2 → ... → C_T
//! • Gradients flow back unchanged (if f_t ≈ 1)
//! • Can remember info for 100+ timesteps!
//! ```
//!
//! **Intuitive Example:**
//! ```
//! Input sequence: "The cat ate the mouse. Later it was full."
//!                                                  ↑
//!                                           What does "it" refer to?
//!
//! Timestep 1: "The"
//!   f_t ≈ 0 (forget empty memory)
//!   i_t ≈ 1 (store: article detected)
//!
//! Timestep 2: "cat"
//!   f_t ≈ 0.3 (partially forget article)
//!   i_t ≈ 1 (store: subject = "cat")
//!   C_t stores: "cat is subject" ← REMEMBERED
//!
//! Timestep 3-7: "ate the mouse."
//!   f_t ≈ 0.9 (keep subject in memory!)
//!   i_t ≈ 0.5 (add action info)
//!
//! Timestep 8-9: "Later it"
//!   o_t queries memory
//!   Outputs: "cat" (subject from memory!)
//! ```
//!
//! ## GRU: Gated Recurrent Unit (2014)
//!
//! **Motivation:** LSTM works great but has many parameters. Can we simplify?
//!
//! ### Key Simplifications
//!
//! ```
//! GRU changes from LSTM:
//! 1. Merges cell state and hidden state (only h_t)
//! 2. Two gates instead of three:
//!    • Reset gate (r_t): How much past to forget
//!    • Update gate (z_t): Balance between past and new
//! 3. Fewer parameters → faster training
//! ```
//!
//! ### Mathematical Formulation
//!
//! ```
//! Input: x_t, previous hidden h_{t-1}
//!
//! 1. Update gate (how much to update):
//!    z_t = σ(W_z · [h_{t-1}, x_t] + b_z)
//!
//! 2. Reset gate (how much past to use):
//!    r_t = σ(W_r · [h_{t-1}, x_t] + b_r)
//!
//! 3. Candidate hidden state:
//!    h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)
//!          ↑
//!    reset gate modulates how much past to use
//!
//! 4. Final hidden state:
//!    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
//!          ↑                      ↑
//!      keep old              add new
//!
//! Intuition:
//! • z_t = 1: Completely replace with new info
//! • z_t = 0: Keep old hidden state
//! • z_t = 0.5: Mix 50-50
//! ```
//!
//! ## LSTM vs GRU Comparison
//!
//! | Aspect | LSTM | GRU |
//! |--------|------|-----|
//! | **Gates** | 3 (forget, input, output) | 2 (update, reset) |
//! | **States** | 2 (cell C_t, hidden h_t) | 1 (hidden h_t) |
//! | **Parameters** | ~4× input size | ~3× input size |
//! | **Speed** | Slower | ~25% faster |
//! | **Memory** | More | Less |
//! | **Performance** | Slightly better on complex tasks | Comparable on most tasks |
//! | **Use When** | Large datasets, complex patterns | Limited data, faster training |
//!
//! ## When to Use Which?
//!
//! **Use LSTM when:**
//! - Very long sequences (100+ timesteps)
//! - Complex temporal patterns
//! - Large dataset (can afford the parameters)
//! - Need maximum accuracy
//! - Examples: machine translation, speech recognition, video analysis
//!
//! **Use GRU when:**
//! - Moderate sequence length (10-50 timesteps)
//! - Limited training data
//! - Need faster training/inference
//! - Resource constraints (mobile, edge)
//! - Examples: text classification, simple time series, sentiment analysis
//!
//! **Use Vanilla RNN when:**
//! - Very short sequences (< 10 timesteps)
//! - Simple patterns
//! - Teaching/understanding basics
//!
//! **Use Transformer when:**
//! - Have lots of data and compute
//! - Need parallelization
//! - Very long sequences with complex dependencies
//! - State-of-the-art performance required
//!
//! ## Bidirectional LSTM/GRU
//!
//! **Idea:** Process sequence in both directions
//!
//! ```
//! Forward LSTM:  The cat ate  →  →  →
//! Backward LSTM:  ←  ←  ← ate cat The
//!
//! Concatenate: [h_forward, h_backward]
//!
//! Benefits:
//! • Sees full context (past + future)
//! • Better for classification, tagging
//! • Can't use for generation (no future!)
//!
//! Use cases:
//! ✅ Named entity recognition
//! ✅ Part-of-speech tagging
//! ✅ Sentiment classification
//! ❌ Text generation (no future available)
//! ❌ Real-time prediction
//! ```
//!
//! ## Stacked/Deep LSTM/GRU
//!
//! **Architecture:**
//! ```
//! Layer 3: LSTM → LSTM → LSTM (high-level features)
//!           ↑      ↑      ↑
//! Layer 2: LSTM → LSTM → LSTM (mid-level features)
//!           ↑      ↑      ↑
//! Layer 1: LSTM → LSTM → LSTM (low-level features)
//!           ↑      ↑      ↑
//! Input:   "The"  "cat"  "ate"
//!
//! Each layer learns different abstraction level:
//! • Layer 1: Syntax, word patterns
//! • Layer 2: Phrases, local context
//! • Layer 3: Semantics, global meaning
//! ```
//!
//! **Guidelines:**
//! - 1 layer: Simple tasks (sentiment, basic classification)
//! - 2 layers: Most tasks (translation, NER, tagging)
//! - 3-4 layers: Complex tasks (question answering, abstractive summarization)
//! - 5+ layers: Usually not helpful (use Transformers instead)
//!
//! ## Training Tips
//!
//! ### 1. Gradient Clipping
//! ```
//! Problem: While vanishing is solved, exploding can still occur
//!
//! Solution:
//! if ||gradient|| > threshold:
//!     gradient = gradient * (threshold / ||gradient||)
//!
//! Typical threshold: 1.0 to 5.0
//! ```
//!
//! ### 2. Initialization
//! ```
//! • Forget gate bias: Initialize to 1 or 2
//!   Reason: Start by remembering everything, learn to forget
//!
//! • Other biases: Initialize to 0
//! • Weights: Xavier/Glorot initialization
//! ```
//!
//! ### 3. Dropout
//! ```
//! ✅ Apply dropout on:
//!    • Between LSTM layers (vertical dropout)
//!    • On input to LSTM
//!
//! ❌ Don't apply dropout:
//!    • On recurrent connections (causes forgetting)
//!
//! Typical rate: 0.2 - 0.5
//! ```
//!
//! ### 4. Learning Rate
//! ```
//! Start: 0.001 (Adam) or 0.01 (SGD)
//! Schedule: Reduce by 2-10× when plateaus
//! Warmup: Helpful for large models
//! ```
//!
//! ### 5. Batch Size
//! ```
//! • Larger batch: More stable, faster training
//! • Smaller batch: Better generalization
//! • Typical: 32-128 for NLP, 64-256 for time series
//! ```
//!
//! ## Applications & Success Stories
//!
//! ### Natural Language Processing
//! ```
//! Machine Translation (pre-Transformer era):
//! • Google Neural Machine Translation (2016)
//! • 8-layer stacked LSTM
//! • Reduced translation errors by 60%
//!
//! Sentiment Analysis:
//! • IMDB review classification
//! • Bidirectional LSTM: 89% accuracy
//! • Simple CNN: 87% accuracy
//! ```
//!
//! ### Speech Recognition
//! ```
//! Google Voice Search:
//! • LSTM-based acoustic model
//! • 49% error reduction
//! • Now handles accents, noise
//! ```
//!
//! ### Time Series Forecasting
//! ```
//! Stock price prediction:
//! • GRU often outperforms LSTM
//! • Faster training on financial data
//! • Captures market momentum
//!
//! Weather forecasting:
//! • Stacked LSTM for multi-day prediction
//! • Learns seasonal patterns
//! ```
//!
//! ### Video Analysis
//! ```
//! Action recognition:
//! • CNN extracts frame features
//! • LSTM models temporal dynamics
//! • UCF101 dataset: 94% accuracy
//! ```
//!
//! ### Music Generation
//! ```
//! • LSTM learns note patterns
//! • Generates Bach-style chorales
//! • Remembers musical themes
//! ```
//!
//! ## Modern Context: LSTM/GRU vs Transformers
//!
//! **Transformers took over (2017+) because:**
//! ```
//! ✅ Parallelization: Process all tokens at once
//! ✅ Better long-range dependencies via attention
//! ✅ Scale better with data/compute
//!
//! Examples:
//! • BERT (2018): Replaced LSTM for NLP
//! • GPT series: Pure decoder transformers
//! • No recurrence needed!
//! ```
//!
//! **LSTM/GRU still useful for:**
//! ```
//! ✅ Online/streaming processing (can't wait for full sequence)
//! ✅ Limited memory (O(1) vs O(n²) for Transformer)
//! ✅ Small datasets (fewer parameters to train)
//! ✅ Real-time applications (lower latency)
//! ✅ Time series forecasting (temporal inductive bias)
//! ✅ Audio processing (continuous streams)
//!
//! Examples:
//! • IoT sensor data analysis
//! • Real-time speech recognition
//! • Anomaly detection in streams
//! • Mobile applications
//! ```
//!
//! ## Variants and Extensions
//!
//! ### Peephole LSTM
//! ```
//! Enhancement: Gates can see cell state
//!
//! f_t = σ(W_f · [C_{t-1}, h_{t-1}, x_t] + b_f)
//!                ↑
//!         Peephole connection
//!
//! Benefit: More precise timing control
//! Use: When exact timing matters (music, speech)
//! ```
//!
//! ### Coupled Input-Forget Gates
//! ```
//! Observation: Input and forget gates often opposite
//!
//! Standard:  f_t and i_t independent
//! Coupled:   i_t = 1 - f_t
//!
//! Benefit: 25% fewer parameters, similar performance
//! ```
//!
//! ### Layer Normalization
//! ```
//! Add normalization within LSTM:
//!
//! h_t = LayerNorm(LSTM(x_t, h_{t-1}))
//!
//! Benefits:
//! • Faster training
//! • Better generalization
//! • Allows higher learning rates
//! ```
//!
//! ## Implementation Considerations
//!
//! ### Computational Complexity
//! ```
//! LSTM forward pass:
//! • Memory: O(n × h) where n=sequence length, h=hidden size
//! • Time: O(n × h²) (matrix multiplications)
//! • Compared to Transformer: O(n² × h)
//!
//! LSTM is faster for: n > h (long sequences, small hidden size)
//! Transformer is faster for: h > n (short sequences, large hidden size)
//! ```
//!
//! ### Batch Processing
//! ```
//! Challenge: Sequences have different lengths
//!
//! Solutions:
//! 1. Padding + masking:
//!    Seq1: [1, 2, 3, 0, 0]  ← padded
//!    Seq2: [4, 5, 6, 7, 8]  ← full
//!    Mask: [1, 1, 1, 0, 0]  ← ignore padded
//!
//! 2. Packing (more efficient):
//!    Pack sequences → process → unpack
//!    Avoids computing on padding
//! ```
//!
//! ## Historical Impact
//!
//! **1997: LSTM Invented**
//! - Hochreiter & Schmidhuber
//! - Solved vanishing gradient problem
//! - Enabled long-term memory
//!
//! **2000-2010: Gradual Adoption**
//! - Speech recognition
//! - Handwriting recognition
//! - Limited by compute
//!
//! **2011-2014: Deep Learning Revolution**
//! - GPUs make training feasible
//! - Alex Graves' work on speech/handwriting
//! - Sequence-to-sequence models
//!
//! **2014: GRU Introduced**
//! - Cho et al.
//! - Simpler, faster alternative
//! - Similar performance
//!
//! **2015-2017: Peak LSTM Era**
//! - Google Translate uses LSTM
//! - State-of-the-art in NLP
//! - Most popular sequence model
//!
//! **2017+: Transformer Era**
//! - Attention is All You Need
//! - LSTM usage declining in NLP
//! - Still relevant for streaming/time series
//!
//! ## Code Example Pattern
//!
//! ```rust
//! // Pseudo-code for LSTM in Rust
//!
//! struct LSTMCell {
//!     W_f, W_i, W_o, W_c: Array2<f64>,  // Weight matrices
//!     b_f, b_i, b_o, b_c: Array1<f64>,  // Biases
//! }
//!
//! fn forward(x_t: Array1<f64>, h_prev: Array1<f64>, C_prev: Array1<f64>)
//!     -> (Array1<f64>, Array1<f64>)
//! {
//!     let combined = concatenate(&[h_prev, x_t]);
//!
//!     // Gates
//!     let f_t = sigmoid(W_f.dot(&combined) + b_f);
//!     let i_t = sigmoid(W_i.dot(&combined) + b_i);
//!     let o_t = sigmoid(W_o.dot(&combined) + b_o);
//!     let C_tilde = tanh(W_c.dot(&combined) + b_c);
//!
//!     // Update cell state
//!     let C_t = f_t * C_prev + i_t * C_tilde;
//!
//!     // Output
//!     let h_t = o_t * tanh(C_t);
//!
//!     (h_t, C_t)
//! }
//! ```

fn main() {
    println!("=== LSTM & GRU: Solving the Vanishing Gradient Problem ===\n");

    println!("This example explains LSTM and GRU, the architectures that made");
    println!("long-term sequence learning possible.\n");

    println!("📚 Key Concepts Covered:");
    println!("  • Vanishing gradient problem in RNNs");
    println!("  • LSTM architecture with gates and cell state");
    println!("  • GRU as a simpler alternative");
    println!("  • When to use LSTM vs GRU vs Transformers");
    println!("  • Bidirectional and stacked variants");
    println!("  • Training tips and best practices\n");

    println!("🎯 Why This Matters:");
    println!("  • LSTM enabled the first wave of deep learning for sequences");
    println!("  • Still relevant for streaming data and resource-constrained settings");
    println!("  • Foundation for understanding modern sequence models");
    println!("  • Critical for time series, speech, and online processing\n");

    println!("See the source code documentation for comprehensive explanations!");
}

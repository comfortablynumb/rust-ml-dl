//! # Recurrent Neural Network (RNN) Example
//!
//! This example demonstrates Recurrent Neural Networks, designed for sequential data
//! like text, time series, audio, and video.
//!
//! ## What is an RNN?
//!
//! Unlike feedforward networks that process inputs independently, RNNs maintain a
//! "memory" (hidden state) that persists across time steps, allowing them to process
//! sequences of any length.
//!
//! ## The Key Idea: Memory
//!
//! **Feedforward NN**: Each input is independent
//! ```
//! Input₁ → Output₁
//! Input₂ → Output₂
//! Input₃ → Output₃
//! ```
//!
//! **RNN**: Inputs are connected through hidden state
//! ```
//! Input₁ → [RNN] → Output₁
//!            ↓ h₁
//! Input₂ → [RNN] → Output₂
//!            ↓ h₂
//! Input₃ → [RNN] → Output₃
//! ```
//!
//! ## RNN Architecture
//!
//! At each time step t:
//! ```
//! hₜ = tanh(Wₓₕ·xₜ + Wₕₕ·hₜ₋₁ + bₕ)
//! yₜ = Wₕy·hₜ + by
//! ```
//!
//! Where:
//! - hₜ = hidden state at time t
//! - xₜ = input at time t
//! - yₜ = output at time t
//! - Wₓₕ = input-to-hidden weights
//! - Wₕₕ = hidden-to-hidden weights (memory!)
//! - Wₕy = hidden-to-output weights
//!
//! ## Unrolled View
//!
//! ```
//! x₁    x₂    x₃    x₄
//!  ↓     ↓     ↓     ↓
//! [RNN][RNN][RNN][RNN]
//!  ↓     ↓     ↓     ↓
//! y₁    y₂    y₃    y₄
//! ```
//!
//! The same RNN cell is reused at each time step!
//!
//! ## Types of Sequence Tasks
//!
//! ### 1. Many-to-One
//! ```
//! Sentiment Analysis:
//! "This" → "movie" → "is" → "great" → [Positive]
//! ```
//!
//! ### 2. One-to-Many
//! ```
//! Image Captioning:
//! [Image] → "A" → "cat" → "on" → "mat"
//! ```
//!
//! ### 3. Many-to-Many (Same Length)
//! ```
//! Video Classification:
//! Frame₁ → [Action₁]
//! Frame₂ → [Action₂]
//! Frame₃ → [Action₃]
//! ```
//!
//! ### 4. Many-to-Many (Different Lengths)
//! ```
//! Machine Translation:
//! "Hello" → "world" → [Encoder] → [Decoder] → "Hola" → "mundo"
//! ```
//!
//! ## Key Challenges
//!
//! ### 1. Vanishing Gradients
//! **Problem**: Gradients shrink exponentially with sequence length
//! **Effect**: Cannot learn long-term dependencies
//! **Solution**: LSTM, GRU
//!
//! ### 2. Exploding Gradients
//! **Problem**: Gradients grow exponentially
//! **Effect**: Training becomes unstable
//! **Solution**: Gradient clipping
//!
//! ## Advanced RNN Variants
//!
//! ### LSTM (Long Short-Term Memory)
//! ```
//! Introduces gates to control information flow:
//! - Forget Gate: What to forget from memory
//! - Input Gate: What new info to store
//! - Output Gate: What to output
//! ```
//!
//! **Equations:**
//! ```
//! fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)  // Forget gate
//! iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)  // Input gate
//! C̃ₜ = tanh(WC·[hₜ₋₁, xₜ] + bC)  // Candidate memory
//! Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ  // Update memory
//! oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)  // Output gate
//! hₜ = oₜ ⊙ tanh(Cₜ)  // Output
//! ```
//!
//! ### GRU (Gated Recurrent Unit)
//! ```
//! Simpler than LSTM, often works just as well:
//! - Update Gate: How much to update
//! - Reset Gate: How much of past to forget
//! ```
//!
//! **Equations:**
//! ```
//! zₜ = σ(Wz·[hₜ₋₁, xₜ])  // Update gate
//! rₜ = σ(Wr·[hₜ₋₁, xₜ])  // Reset gate
//! h̃ₜ = tanh(W·[rₜ ⊙ hₜ₋₁, xₜ])  // Candidate
//! hₜ = (1-zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ  // New hidden state
//! ```
//!
//! ## Applications
//!
//! **Natural Language Processing:**
//! - Machine translation
//! - Text generation
//! - Sentiment analysis
//! - Named entity recognition
//! - Question answering
//!
//! **Time Series:**
//! - Stock price prediction
//! - Weather forecasting
//! - Anomaly detection
//!
//! **Speech & Audio:**
//! - Speech recognition
//! - Music generation
//! - Voice assistants
//!
//! **Video:**
//! - Action recognition
//! - Video captioning
//!
//! ## Modern Alternatives
//!
//! While RNNs were dominant, they have limitations:
//!
//! **Transformers (2017+)**
//! - No sequential processing (parallel!)
//! - Better long-range dependencies
//! - Powers GPT, BERT, ChatGPT
//! - Standard for NLP now
//!
//! **When to still use RNNs:**
//! - Online/streaming processing
//! - Very long sequences
//! - Limited compute resources
//! - Real-time applications

use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Simple RNN cell
struct SimpleRNN {
    // Input to hidden weights
    w_xh: Array2<f64>,
    // Hidden to hidden weights (memory!)
    w_hh: Array2<f64>,
    // Hidden to output weights
    w_hy: Array2<f64>,
    // Biases
    b_h: Array1<f64>,
    b_y: Array1<f64>,
}

impl SimpleRNN {
    fn new(input_size: usize, hidden_size: usize, output_size: usize, rng: &mut StdRng) -> Self {
        let scale = 0.1;
        SimpleRNN {
            w_xh: Array2::random_using((input_size, hidden_size), StandardNormal, rng) * scale,
            w_hh: Array2::random_using((hidden_size, hidden_size), StandardNormal, rng) * scale,
            w_hy: Array2::random_using((hidden_size, output_size), StandardNormal, rng) * scale,
            b_h: Array1::zeros(hidden_size),
            b_y: Array1::zeros(output_size),
        }
    }

    /// Forward pass for one time step
    fn step(&self, x: &Array1<f64>, h_prev: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        // h = tanh(W_xh * x + W_hh * h_prev + b_h)
        let h = (&x.dot(&self.w_xh) + &h_prev.dot(&self.w_hh) + &self.b_h)
            .mapv(|v| v.tanh());

        // y = W_hy * h + b_y
        let y = h.dot(&self.w_hy) + &self.b_y;

        (h, y)
    }

    /// Process entire sequence
    fn forward(&self, inputs: &[Array1<f64>]) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
        let hidden_size = self.w_hh.nrows();
        let mut h = Array1::zeros(hidden_size);

        let mut hidden_states = Vec::new();
        let mut outputs = Vec::new();

        for x in inputs {
            let (h_new, y) = self.step(x, &h);
            h = h_new.clone();
            hidden_states.push(h.clone());
            outputs.push(y);
        }

        (hidden_states, outputs)
    }
}

fn main() -> anyhow::Result<()> {
    println!("=== Recurrent Neural Network (RNN) Basics ===\n");

    let mut rng = StdRng::seed_from_u64(42);

    // Create a simple RNN
    println!("1. Creating a Simple RNN\n");
    let input_size = 3;
    let hidden_size = 4;
    let output_size = 2;

    let rnn = SimpleRNN::new(input_size, hidden_size, output_size, &mut rng);

    println!("   Architecture:");
    println!("   - Input size: {}", input_size);
    println!("   - Hidden size: {} (memory capacity)", hidden_size);
    println!("   - Output size: {}\n", output_size);

    println!("   Weight matrices:");
    println!("   - W_xh: {} × {} (input to hidden)", input_size, hidden_size);
    println!("   - W_hh: {} × {} (hidden to hidden - THE MEMORY!)", hidden_size, hidden_size);
    println!("   - W_hy: {} × {} (hidden to output)", hidden_size, output_size);

    // Create a sequence
    println!("\n2. Processing a Sequence\n");

    let sequence: Vec<Array1<f64>> = vec![
        Array1::from_vec(vec![1.0, 0.0, 0.0]),
        Array1::from_vec(vec![0.0, 1.0, 0.0]),
        Array1::from_vec(vec![0.0, 0.0, 1.0]),
        Array1::from_vec(vec![1.0, 1.0, 0.0]),
    ];

    println!("   Input sequence (4 time steps):");
    for (t, x) in sequence.iter().enumerate() {
        println!("   t={}: {:?}", t, x.to_vec());
    }

    // Forward pass
    let (hidden_states, outputs) = rnn.forward(&sequence);

    println!("\n   Hidden states (RNN's memory):");
    for (t, h) in hidden_states.iter().enumerate() {
        println!("   t={}: [{:.3}, {:.3}, {:.3}, {:.3}]",
                 t, h[0], h[1], h[2], h[3]);
    }

    println!("\n   Notice: Hidden state changes at each time step!");
    println!("   This is the RNN's \"memory\" of what it has seen.");

    println!("\n   Outputs at each time step:");
    for (t, y) in outputs.iter().enumerate() {
        println!("   t={}: [{:.3}, {:.3}]", t, y[0], y[1]);
    }

    // Demonstrate sequence processing patterns
    println!("\n3. Common RNN Patterns\n");

    println!("   A) Many-to-One (e.g., Sentiment Analysis)");
    println!("      Input:  \"This\" → \"movie\" → \"is\" → \"great\"");
    println!("      Output: Only use final hidden state → \"Positive\"");
    println!("      Use case: Text classification\n");

    println!("   B) One-to-Many (e.g., Image Captioning)");
    println!("      Input:  [Image embedding]");
    println!("      Output: \"A\" → \"cat\" → \"on\" → \"mat\"");
    println!("      Use case: Caption generation\n");

    println!("   C) Many-to-Many (e.g., Named Entity Recognition)");
    println!("      Input:  \"John\" → \"lives\" → \"in\" → \"Paris\"");
    println!("      Output: PERSON → O → O → LOCATION");
    println!("      Use case: Sequence labeling\n");

    println!("   D) Seq2Seq (e.g., Translation)");
    println!("      Encoder: \"Hello\" → \"world\" → [context]");
    println!("      Decoder: [context] → \"Hola\" → \"mundo\"");
    println!("      Use case: Machine translation\n");

    // Demonstrate the vanishing gradient problem
    println!("4. The Vanishing Gradient Problem\n");

    println!("   Problem: Gradients shrink as we backpropagate through time");
    println!("   ");
    println!("   t=0    t=1    t=2    t=3    t=4");
    println!("    ↓      ↓      ↓      ↓      ↓");
    println!("   1.0 → 0.8 → 0.6 → 0.4 → 0.2  (gradient magnitude)");
    println!("   ");
    println!("   Result: Cannot learn long-term dependencies!");
    println!("   Example: \"The cat, which was very hungry, ... ate\"");
    println!("            RNN forgets \"cat\" by the time it sees \"ate\"\n");

    println!("   Solutions:");
    println!("   ✓ LSTM: Uses gates to control information flow");
    println!("   ✓ GRU: Simpler gated architecture");
    println!("   ✓ Gradient clipping: Prevents exploding gradients");
    println!("   ✓ Better initialization: He/Xavier init");

    // Compare RNN variants
    println!("\n5. RNN Variants Comparison\n");

    println!("   ┌──────────────┬─────────────┬────────────┬──────────────┐");
    println!("   │ Feature      │ Simple RNN  │ LSTM       │ GRU          │");
    println!("   ├──────────────┼─────────────┼────────────┼──────────────┤");
    println!("   │ Parameters   │ Low         │ High       │ Medium       │");
    println!("   │ Speed        │ Fast        │ Slow       │ Medium       │");
    println!("   │ Long memory  │ Poor        │ Excellent  │ Very Good    │");
    println!("   │ Training     │ Difficult   │ Stable     │ Stable       │");
    println!("   │ Use case     │ Simple seq  │ Long seq   │ Good default │");
    println!("   └──────────────┴─────────────┴────────────┴──────────────┘");

    println!("\n6. When to Use RNNs vs Transformers\n");

    println!("   Use RNNs when:");
    println!("   ✓ Processing data in real-time/streaming");
    println!("   ✓ Variable length sequences");
    println!("   ✓ Limited computational resources");
    println!("   ✓ Need sequential processing");

    println!("\n   Use Transformers when:");
    println!("   ✓ Have large datasets");
    println!("   ✓ Need long-range dependencies");
    println!("   ✓ Can parallelize computation");
    println!("   ✓ NLP tasks (state-of-the-art)");

    println!("\n7. Real-World Applications\n");

    println!("   Language:");
    println!("   - GPT-2 (2019): 1.5B parameters, text generation");
    println!("   - BERT: Bidirectional encoding for understanding");
    println!("   - Translation: Google Translate, DeepL\n");

    println!("   Speech:");
    println!("   - Speech recognition: Siri, Alexa, Google Assistant");
    println!("   - Text-to-speech: Natural voice synthesis\n");

    println!("   Time Series:");
    println!("   - Stock prediction: Financial forecasting");
    println!("   - Anomaly detection: Network security\n");

    println!("   Creative:");
    println!("   - Music generation: Compose melodies");
    println!("   - Story writing: AI authors");

    println!("\n=== Example Complete! ===");
    println!("\nKey Takeaways:");
    println!("- RNNs process sequences by maintaining hidden state");
    println!("- Hidden state acts as memory across time steps");
    println!("- Vanishing gradients limit learning long dependencies");
    println!("- LSTM/GRU solve this with gating mechanisms");
    println!("- Transformers are now preferred for many tasks");
    println!("- RNNs still useful for streaming/online processing");

    Ok(())
}

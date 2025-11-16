// ============================================================================
// Sequence-to-Sequence (Seq2Seq) Models
// ============================================================================
//
// The encoder-decoder architecture that revolutionized neural machine translation
// and sequence transformation tasks.
//
// PROBLEM SOLVED:
// ----------------
// Traditional models process fixed-length inputs → fixed-length outputs.
// But many real-world tasks need variable-length sequences:
//   • Machine translation: "Hello" (1 word) → "你好" (1 character, 2 syllables)
//   • Summarization: Long article → Short summary
//   • Chat: Question of any length → Response of any length
//
// Seq2Seq enables variable-length input → variable-length output transformation.
//
// CORE ARCHITECTURE:
// ------------------
//
// Input Sequence → [ENCODER] → Context Vector → [DECODER] → Output Sequence
//
// Encoder:
//   • Processes entire input sequence (e.g., English sentence)
//   • Compresses into fixed-size context vector (thought vector)
//   • Usually RNN/LSTM/GRU processing input left-to-right
//
// Context Vector:
//   • Fixed-size representation of entire input
//   • "Summary" of input sequence
//   • Bottleneck: All input information compressed here
//
// Decoder:
//   • Generates output sequence one token at a time
//   • Conditioned on context vector
//   • Uses previous output as input (autoregressive)
//
// Example (Translation):
// ----------------------
// Input:  "I love Rust" → Encoder → Context [0.2, -0.5, ...] → Decoder → "J'aime Rust"
//
// Step-by-step generation:
//   1. Decoder receives context + <START>
//   2. Generates "J'"
//   3. Receives context + "J'" → generates "aime"
//   4. Receives context + "J' aime" → generates "Rust"
//   5. Receives context + "J' aime Rust" → generates <END>
//
// BASIC SEQ2SEQ (2014):
// ---------------------
//
// Architecture:
// ```
// Encoder LSTM:
//   h_0 = 0
//   For each input word x_t:
//     h_t = LSTM(x_t, h_{t-1})
//   context = h_final  (last hidden state)
//
// Decoder LSTM:
//   s_0 = context  (initialize with encoder's final state)
//   For each step:
//     s_t = LSTM(y_{t-1}, s_{t-1})
//     y_t = softmax(W·s_t)  (predict next word)
// ```
//
// Limitations:
// • Bottleneck: Entire input compressed to single vector
// • Long sequences: Information loss (first words forgotten)
// • Fixed-size context: Same size for "Hi" and 100-word paragraph
//
// ATTENTION-BASED SEQ2SEQ (2015):
// --------------------------------
//
// Key Innovation: Decoder can "look back" at all encoder states!
//
// Instead of single context vector, decoder attends to all encoder outputs:
//
// ```
// Encoder:
//   h_1, h_2, ..., h_n = encode(input)  (all hidden states, not just last)
//
// Decoder at step t:
//   1. Compute attention scores:
//      score_i = attention(s_{t-1}, h_i)  (how relevant is h_i?)
//
//   2. Normalize to attention weights:
//      α_i = softmax(score)  (sum to 1)
//
//   3. Compute context vector:
//      c_t = Σ α_i · h_i  (weighted sum of encoder states)
//
//   4. Generate output:
//      s_t = LSTM([y_{t-1}, c_t], s_{t-1})  (context + previous word)
//      y_t = softmax(W·s_t)
// ```
//
// Attention Mechanisms:
// ---------------------
//
// 1. Additive (Bahdanau) Attention:
//    score(s, h) = v^T · tanh(W_s·s + W_h·h)
//    • Learnable parameters: W_s, W_h, v
//    • More expressive, slightly slower
//
// 2. Multiplicative (Luong) Attention:
//    score(s, h) = s^T · W · h
//    • Simpler, faster
//    • Foundation for Transformer attention
//
// 3. Dot-Product Attention:
//    score(s, h) = s^T · h
//    • No parameters, fastest
//    • Works if s and h same dimension
//
// Benefits of Attention:
// ----------------------
// • No bottleneck: Decoder accesses all encoder states
// • Better long sequences: Can attend to first word even after 100 steps
// • Interpretable: Attention weights show which input words influenced output
// • Performance: +10-20 BLEU points on translation
//
// Example Attention Visualization (Translation):
// ----------------------------------------------
//
// Input:  "The cat sat on the mat"
// Output: "Le chat était assis sur le tapis"
//
// When generating "chat":
//   Attention: [0.05, 0.90, 0.02, 0.01, 0.01, 0.01]  (focused on "cat")
//
// When generating "tapis":
//   Attention: [0.01, 0.01, 0.02, 0.03, 0.03, 0.90]  (focused on "mat")
//
// TRAINING: TEACHER FORCING
// --------------------------
//
// Problem: During training, decoder's predictions can be wrong.
// If we feed wrong prediction as next input → errors compound!
//
// Solution: Teacher Forcing
//   • Use ground truth (not prediction) as next input during training
//   • Example: Translating "I am good" → "Je suis bon"
//
// Wrong (without teacher forcing):
//   Target: "Je suis bon"
//   Step 1: Predict "Le" (wrong!) → Feed "Le" to next step
//   Step 2: Given "Le", predict "chat" (compounding error!)
//
// Right (with teacher forcing):
//   Target: "Je suis bon"
//   Step 1: Predict "Le" (wrong), but feed "Je" (ground truth) to next step
//   Step 2: Given "Je", predict "suis" (can learn correctly)
//
// Teacher Forcing Ratio:
//   • Training: Use teacher forcing 100% (or 50-90% for robustness)
//   • Inference: Can't use teacher forcing (no ground truth)
//
// INFERENCE: DECODING STRATEGIES
// -------------------------------
//
// 1. Greedy Decoding:
//    • At each step, pick highest-probability word
//    • Fast, but suboptimal (local choices, not global best)
//
//    Example:
//      Step 1: P("The"=0.7, "A"=0.3) → Pick "The"
//      Step 2: P("cat"=0.6, "dog"=0.4) → Pick "cat"
//
//    Problem: Maybe "A dog ..." has higher total probability than "The cat ..."!
//
// 2. Beam Search:
//    • Keep top-k hypotheses (beam) at each step
//    • Explore multiple paths simultaneously
//
//    Example (beam_size=2):
//      Step 1: Keep ["The", "A"]
//      Step 2:
//        From "The": ["The cat", "The dog"]
//        From "A":   ["A cat", "A dog"]
//        Keep top 2: ["The cat", "A dog"]  (based on cumulative log-prob)
//      Continue until <END> or max_length
//
//    Beam size trade-off:
//      • beam=1: Greedy (fast, suboptimal)
//      • beam=5: Good balance (typical)
//      • beam=10: Better quality, slower
//      • beam=100: Diminishing returns, much slower
//
// 3. Top-k / Top-p (Nucleus) Sampling:
//    • Sample from top-k or top cumulative probability p
//    • Used for creative generation (chat, story)
//    • Introduces diversity (not deterministic)
//
// Length Normalization:
// ---------------------
// Problem: Beam search favors short sequences (fewer terms to multiply)
//
// Solution: Normalize by length
//   score = (log P(y_1, ..., y_n)) / n^α
//   • α=0: No normalization
//   • α=0.6-0.7: Typical value
//   • α=1.0: Full normalization
//
// BIDIRECTIONAL ENCODER:
// ----------------------
//
// Improvement: Process input both left-to-right AND right-to-left
//
// ```
// Forward LSTM:  "I love Rust" → h_fwd = [h_1^fwd, h_2^fwd, h_3^fwd]
// Backward LSTM: "Rust love I" → h_bwd = [h_1^bwd, h_2^bwd, h_3^bwd]
// Concatenate:   h_i = [h_i^fwd; h_i^bwd]
// ```
//
// Benefits:
//   • Each position sees entire sentence context
//   • Better for ambiguous words: "bank" (river vs money)
//   • +1-2 BLEU points on translation
//
// STACKED/DEEP SEQ2SEQ:
// ---------------------
//
// Use multiple LSTM layers for more capacity:
//
// ```
// Encoder:
//   Layer 1: x → h^1
//   Layer 2: h^1 → h^2
//   Layer 3: h^2 → h^3
//
// Decoder:
//   Layer 1: [y, c] → s^1
//   Layer 2: s^1 → s^2
//   Layer 3: s^2 → s^3 → output
// ```
//
// Typical depth:
//   • Machine translation: 2-4 layers
//   • Speech recognition: 3-5 layers
//   • Diminishing returns after 4-6 layers (for RNN/LSTM)
//
// APPLICATIONS:
// -------------
//
// 1. Machine Translation:
//    • English → French, Chinese, etc.
//    • Google Translate (before Transformer): Seq2Seq + Attention
//    • Typical: 2-4 layer bidirectional encoder, 2-4 layer decoder
//    • Vocabulary: 30K-50K subword units (BPE)
//
// 2. Text Summarization:
//    • Long article → Short summary
//    • Encoder: Process full article
//    • Decoder: Generate summary
//    • Challenge: Copying vs abstractive summary
//
// 3. Dialogue/Chatbots:
//    • User message → Bot response
//    • Seq2Seq trained on conversation pairs
//    • Limitations: Generic responses ("I don't know"), no memory across turns
//    • Modern: Use Transformers (GPT, ChatGPT)
//
// 4. Image Captioning:
//    • Image → Text description
//    • Encoder: CNN (ResNet, EfficientNet)
//    • Decoder: LSTM generates caption word-by-word
//    • Example: ResNet → 2048-d vector → LSTM decoder
//
// 5. Speech Recognition:
//    • Audio waveform → Text transcription
//    • Encoder: Process audio frames
//    • Decoder: Generate text
//    • Modern: Listen, Attend and Spell (LAS) architecture
//
// 6. Code Generation:
//    • Natural language description → Code
//    • "Sort a list" → "array.sort()"
//    • Precursor to Codex, GitHub Copilot
//
// HANDLING OUT-OF-VOCABULARY (OOV):
// ----------------------------------
//
// Problem: Fixed vocabulary can't represent all words
//   • Training vocab: 50K most common words
//   • Test: "supercalifragilisticexpialidocious" → <UNK>
//
// Solutions:
//
// 1. Character-level Seq2Seq:
//    • Character as tokens: "cat" → ['c', 'a', 't']
//    • No OOV, but very long sequences
//
// 2. Subword Units (BPE, WordPiece):
//    • "unbelievable" → ["un", "believ", "able"]
//    • Balance: Shorter than char-level, no OOV
//    • Used in GPT, BERT (modern standard)
//
// 3. Copy Mechanism:
//    • Allow decoder to copy from input
//    • Useful for names, numbers: "born in 1990" → "1990"
//    • Pointer network: Attention to input, copy token
//
// TRAINING DETAILS:
// -----------------
//
// Loss Function: Cross-Entropy
//   L = -Σ_t log P(y_t | y_<t, x)
//   • Sum over all output positions
//   • Maximize log-likelihood of target sequence
//
// Optimization:
//   • Adam optimizer (lr=0.001 typical)
//   • Gradient clipping (clip_norm=5.0)
//   • Learning rate scheduling: Warmup + decay
//
// Regularization:
//   • Dropout: 0.1-0.3 on LSTM outputs
//   • Weight decay: 1e-5 to 1e-6
//   • Label smoothing: 0.1 (prevent overconfidence)
//
// Data:
//   • Parallel corpus: Millions of sentence pairs
//   • WMT datasets: 4M-40M sentence pairs
//   • Augmentation: Back-translation (translate target → source)
//
// Training Time:
//   • Small model (2-layer): 1-2 days on GPU
//   • Large model (4-layer, big vocab): 1-2 weeks on multi-GPU
//
// Evaluation Metrics:
//   • BLEU: Precision of n-grams (0-100, higher better)
//     - Good: 30-40 (understandable)
//     - Excellent: 40-50 (high quality)
//     - Human: ~50-60 (multiple references)
//   • ROUGE: Recall-based (for summarization)
//   • METEOR: Incorporates synonyms
//   • Human evaluation: Gold standard
//
// SEQ2SEQ vs TRANSFORMER:
// -----------------------
//
// Seq2Seq (LSTM):
//   ✅ Sequential processing captures order naturally
//   ✅ Less memory for short sequences
//   ✅ Simpler architecture
//   ❌ Sequential: Can't parallelize within sequence
//   ❌ Slower training (10-100× slower)
//   ❌ Attention still has quadratic cost
//   ❌ Vanishing gradients for very long sequences
//
// Transformer:
//   ✅ Fully parallelizable (train 100× faster)
//   ✅ Better long-range dependencies
//   ✅ Scalable: Bigger models → better performance
//   ❌ More memory (quadratic in sequence length)
//   ❌ Needs positional encoding (order not built-in)
//   ❌ Requires more data to train well
//
// When to use Seq2Seq:
//   • Limited compute/memory
//   • Small datasets (< 1M examples)
//   • Online/streaming processing
//   • Edge deployment (mobile, IoT)
//
// When to use Transformer:
//   • Large datasets (> 10M examples)
//   • Offline batch processing
//   • Want SOTA performance
//   • Can train on GPUs/TPUs
//
// MODERN CONTEXT (2024):
// ----------------------
//
// Status: Largely replaced by Transformers for most tasks
//   • Translation: Transformer (Google, DeepL)
//   • Chat: GPT, Claude (Transformer decoder)
//   • Summarization: BART, T5 (Transformer)
//
// Still Relevant For:
//   • Streaming/online tasks (speech recognition)
//   • Resource-constrained deployment
//   • Educational purposes (simpler than Transformer)
//   • Specific domains with limited data
//
// Historical Impact:
//   • 2014: Sutskever et al. - First successful neural MT
//   • 2015: Bahdanau et al. - Attention mechanism
//   • 2016: Google adopts for production (GNMT)
//   • 2017: Transformer paper → "Attention is All You Need"
//   • 2018+: Transformers dominate (BERT, GPT)
//
// Evolution Timeline:
//   Seq2Seq (2014) → +Attention (2015) → +Convolutional (2016) →
//   Transformer (2017) → BERT/GPT (2018-19) → GPT-3 (2020) →
//   ChatGPT/GPT-4 (2022-23)
//
// IMPLEMENTATION NOTES:
// ---------------------
//
// This example demonstrates the concepts of Seq2Seq models.
// For production use, consider:
//   • PyTorch, TensorFlow, JAX (mature implementations)
//   • Hugging Face Transformers (pre-trained models)
//   • FairSeq (Facebook's Seq2Seq library)
//
// Key Takeaways:
//   ✓ Seq2Seq enables variable-length sequence transformation
//   ✓ Attention mechanism eliminates bottleneck, enables long sequences
//   ✓ Teacher forcing critical for training stability
//   ✓ Beam search improves inference quality over greedy
//   ✓ Foundation for modern NLP, though now replaced by Transformers
//   ✓ Still useful for streaming tasks and resource-constrained settings
//
// ============================================================================

use ndarray::Array1;
use rand::Rng;

/// Conceptual Seq2Seq encoder
struct Encoder {
    hidden_size: usize,
    num_layers: usize,
}

impl Encoder {
    fn new(vocab_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        println!("Encoder Configuration:");
        println!("  Vocabulary size: {}", vocab_size);
        println!("  Hidden size: {}", hidden_size);
        println!("  Num layers: {}", num_layers);
        println!("  Type: Bidirectional LSTM\n");

        Self { hidden_size, num_layers }
    }

    /// Encode input sequence, return all hidden states (for attention)
    fn encode(&self, input_sequence: &[usize]) -> Vec<Array1<f32>> {
        // In real implementation: LSTM processes sequence
        // Returns hidden state at each position
        let seq_len = input_sequence.len();

        // Simulate encoder hidden states
        let mut rng = rand::thread_rng();
        (0..seq_len)
            .map(|_| {
                Array1::from_vec(
                    (0..self.hidden_size)
                        .map(|_| rng.gen_range(-1.0..1.0))
                        .collect()
                )
            })
            .collect()
    }
}

/// Conceptual attention mechanism
struct Attention;

impl Attention {
    /// Compute attention weights and context vector
    fn compute(
        decoder_state: &Array1<f32>,
        encoder_states: &[Array1<f32>]
    ) -> (Vec<f32>, Array1<f32>) {
        // Compute attention scores (simplified dot-product)
        let scores: Vec<f32> = encoder_states
            .iter()
            .map(|h| decoder_state.dot(h))
            .collect();

        // Softmax to get weights
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

        // Weighted sum of encoder states
        let context = encoder_states
            .iter()
            .zip(&weights)
            .fold(
                Array1::zeros(decoder_state.len()),
                |acc, (state, &weight)| acc + state * weight
            );

        (weights, context)
    }
}

/// Conceptual Seq2Seq decoder with attention
struct Decoder {
    hidden_size: usize,
    vocab_size: usize,
}

impl Decoder {
    fn new(vocab_size: usize, hidden_size: usize) -> Self {
        println!("Decoder Configuration:");
        println!("  Vocabulary size: {}", vocab_size);
        println!("  Hidden size: {}", hidden_size);
        println!("  Attention: Enabled (Bahdanau-style)\n");

        Self { hidden_size, vocab_size }
    }

    /// Greedy decoding
    fn decode_greedy(
        &self,
        encoder_states: &[Array1<f32>],
        max_length: usize
    ) -> Vec<usize> {
        let mut output = Vec::new();
        let mut state = Array1::zeros(self.hidden_size);

        for _ in 0..max_length {
            // Compute attention
            let (_, context) = Attention::compute(&state, encoder_states);

            // Simulate LSTM step and token prediction
            state = &state * 0.9 + &context * 0.1;
            let token = rand::thread_rng().gen_range(0..self.vocab_size);

            // Check for end token (simplified)
            if token == 0 {
                break;
            }

            output.push(token);
        }

        output
    }
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("Sequence-to-Sequence (Seq2Seq) Models");
    println!("{}", "=".repeat(70));
    println!();

    // Model hyperparameters
    let vocab_size = 10000;
    let hidden_size = 512;
    let num_layers = 2;

    println!("ARCHITECTURE:");
    println!("-------------");
    let encoder = Encoder::new(vocab_size, hidden_size, num_layers);
    let decoder = Decoder::new(vocab_size, hidden_size);

    // Example input sequence (token IDs)
    let input_sequence = vec![123, 456, 789, 234, 567];

    println!("ENCODING:");
    println!("---------");
    println!("Input sequence length: {}", input_sequence.len());
    let encoder_states = encoder.encode(&input_sequence);
    println!("Encoder states: {} timesteps, each dim {}\n",
             encoder_states.len(), encoder_states[0].len());

    println!("DECODING (Greedy):");
    println!("------------------");
    let output = decoder.decode_greedy(&encoder_states, 10);
    println!("Generated sequence: {:?}\n", output);

    println!("ATTENTION VISUALIZATION:");
    println!("------------------------");
    let decoder_state = Array1::from_vec(vec![0.1; hidden_size]);
    let (attention_weights, context) = Attention::compute(&decoder_state, &encoder_states);

    println!("Attention weights (decoder attends to encoder positions):");
    for (i, &weight) in attention_weights.iter().enumerate() {
        let bar_length = (weight * 50.0) as usize;
        println!("  Position {}: {:>6.3} {}", i, weight, "█".repeat(bar_length));
    }
    println!("\nContext vector norm: {:.3}", context.iter().map(|x| x*x).sum::<f32>().sqrt());

    println!("\n{}", "=".repeat(70));
    println!("KEY CONCEPTS DEMONSTRATED:");
    println!("{}", "=".repeat(70));
    println!("✓ Encoder processes input → hidden states");
    println!("✓ Attention computes weighted combination of encoder states");
    println!("✓ Decoder generates output token-by-token");
    println!("✓ Attention weights show which input positions are relevant");
    println!();
    println!("APPLICATIONS:");
    println!("  • Machine Translation (English → French)");
    println!("  • Text Summarization (Article → Summary)");
    println!("  • Dialogue Systems (Question → Answer)");
    println!("  • Image Captioning (Image → Description)");
    println!();
    println!("MODERN STATUS:");
    println!("  Largely replaced by Transformers (2017+) for most NLP tasks.");
    println!("  Still used for streaming/online tasks and embedded systems.");
    println!();
}

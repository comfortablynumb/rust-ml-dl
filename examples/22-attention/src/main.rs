//! # Attention Mechanisms: The Foundation of Modern AI
//!
//! This example explains attention mechanisms, the core innovation that powers
//! Transformers, BERT, GPT, and most modern AI systems.
//!
//! ## The Core Problem: Context Awareness
//!
//! **Traditional sequence processing:**
//! ```
//! RNN/LSTM: Process sequentially, limited context
//! "The cat sat on the mat" → h_final
//!                              ↑
//!           Single vector must capture everything!
//!
//! Problem:
//! • Information bottleneck
//! • Difficult for long sequences
//! • Can't focus on relevant parts
//! ```
//!
//! **Solution: Attention**
//! ```
//! "The cat sat on the mat"
//!   ↓   ↓   ↓   ↓   ↓   ↓
//! When translating "sat" → Look at all words, focus on relevant ones
//! Attention weights: [0.05, 0.30, 0.50, 0.10, 0.03, 0.02]
//!                           ↑    ↑
//!                         "cat" "sat" most relevant!
//! ```
//!
//! ## Intuition: Human Attention
//!
//! ```
//! Reading: "The quick brown fox jumps over the lazy dog"
//! Question: "What color is the fox?"
//!
//! Human behavior:
//! • Don't re-read entire sentence
//! • Focus attention on "brown fox"
//! • Ignore irrelevant words
//!
//! Neural attention: Same idea!
//! • Compute relevance scores for all words
//! • Focus on important parts
//! • Weighted combination
//! ```
//!
//! ## Attention Formula
//!
//! **Basic attention:**
//! ```
//! Attention(Q, K, V) = softmax(score(Q, K)) · V
//!
//! Where:
//! • Q: Query ("what am I looking for?")
//! • K: Keys ("what do I have?")
//! • V: Values ("what information to return?")
//! • score: Similarity function
//!
//! Steps:
//! 1. Compute scores: How relevant is each position?
//! 2. Softmax: Convert to probabilities (weights sum to 1)
//! 3. Weighted sum: Combine values by relevance
//! ```
//!
//! ## Attention Variants
//!
//! ### 1. Additive Attention (Bahdanau, 2014)
//!
//! ```
//! score(h_i, s_j) = v^T tanh(W_1 h_i + W_2 s_j)
//!
//! Where:
//! • h_i: Encoder hidden state
//! • s_j: Decoder hidden state
//! • W_1, W_2, v: Learnable parameters
//!
//! Used in: First attention-based machine translation
//! ```
//!
//! ### 2. Multiplicative Attention (Luong, 2015)
//!
//! ```
//! score(h_i, s_j) = h_i^T W s_j
//!
//! Simpler, faster than additive
//! Fewer parameters
//! ```
//!
//! ### 3. Scaled Dot-Product Attention (Vaswani, 2017)
//!
//! **The attention used in Transformers:**
//!
//! ```
//! Attention(Q, K, V) = softmax(QK^T / √d_k) V
//!
//! Where:
//! • Q: Query matrix (n × d_k)
//! • K: Key matrix (m × d_k)
//! • V: Value matrix (m × d_v)
//! • d_k: Dimension of keys
//! • √d_k: Scaling factor
//!
//! Why scale by √d_k?
//! • Dot products grow large for high dimensions
//! • Large values → softmax saturation
//! • Scaling keeps gradients healthy
//!
//! Matrix form:
//! Input: n queries, m key-value pairs
//! Scores: QK^T → (n × m) matrix
//! Weights: softmax → (n × m) probabilities
//! Output: Weights × V → (n × d_v)
//! ```
//!
//! ### Example Calculation
//!
//! ```
//! Sentence: "I love cats"
//! Embedding dim: 4
//!
//! Q = [[1,0,1,0]]  ← Query for "love"
//! K = [[1,0,0,1],  ← Keys for "I"
//!      [1,0,1,0],  ← "love"
//!      [0,1,1,1]]  ← "cats"
//! V = [[0.2,0.1,0.3,0.4],  ← Values for "I"
//!      [0.5,0.5,0.0,0.0],  ← "love"
//!      [0.1,0.9,0.0,0.0]]  ← "cats"
//!
//! Step 1: Compute scores
//! QK^T = [1,0,1,0] · [[1,0,0,1]^T, [1,0,1,0]^T, [0,1,1,1]^T]
//!      = [1, 2, 1]  ← Raw scores
//!
//! Step 2: Scale
//! Scaled = [1, 2, 1] / √4 = [0.5, 1.0, 0.5]
//!
//! Step 3: Softmax
//! Weights = softmax([0.5, 1.0, 0.5])
//!         = [0.21, 0.58, 0.21]  ← Attention weights!
//!
//! Step 4: Weighted sum of values
//! Output = 0.21·V[0] + 0.58·V[1] + 0.21·V[2]
//!        = [0.35, 0.42, 0.06, 0.08]
//!
//! Interpretation: "love" attends mostly to itself (0.58)
//! ```
//!
//! ## Self-Attention
//!
//! **Attend to the same sequence:**
//!
//! ```
//! Regular attention: Encoder → Decoder
//! Self-attention: Sequence attends to itself
//!
//! "The animal didn't cross the street because it was too tired"
//!                                                    ↑
//!                                            What is "it"?
//!
//! Self-attention weights for "it":
//! [0.02, 0.62, 0.05, 0.03, 0.01, 0.05, 0.01, 0.01, 0.15, 0.05]
//!        ↑                                                ↑
//!    "animal" (0.62)                               "tired" (0.15)
//!
//! → "it" = animal!
//! ```
//!
//! ### Computing Self-Attention
//!
//! ```
//! Input: Sequence of embeddings X (n × d)
//!
//! Create Q, K, V from same input:
//! Q = X W_Q  ← Query projection
//! K = X W_K  ← Key projection
//! V = X W_V  ← Value projection
//!
//! Attention(Q, K, V) = softmax(QK^T / √d_k) V
//!
//! Each position attends to all positions!
//! ```
//!
//! ## Multi-Head Attention
//!
//! **Attend to different aspects simultaneously:**
//!
//! ```
//! Single head: One attention pattern
//! Multi-head: Multiple parallel attention patterns
//!
//! head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i)
//!
//! Example with 8 heads:
//! Head 1: Syntactic relationships (subject-verb)
//! Head 2: Semantic relationships (synonyms)
//! Head 3: Positional patterns (nearby words)
//! Head 4: Long-range dependencies
//! ...and so on
//!
//! Concatenate all heads:
//! MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W_O
//!
//! Benefits:
//! • Capture different types of relationships
//! • More robust
//! • Richer representations
//! ```
//!
//! ## Masked Attention
//!
//! **Prevent looking ahead (for autoregressive models):**
//!
//! ```
//! Problem in language generation:
//! Generating "The cat sat on the"
//! Should not see "mat" yet!
//!
//! Solution: Mask future positions
//!
//! Scores before masking:
//! [[s11, s12, s13, s14],
//!  [s21, s22, s23, s24],
//!  [s31, s32, s33, s34],
//!  [s41, s42, s43, s44]]
//!
//! Apply mask (set future to -∞):
//! [[s11, -∞,  -∞,  -∞ ],
//!  [s21, s22, -∞,  -∞ ],
//!  [s31, s32, s33, -∞ ],
//!  [s41, s42, s43, s44]]
//!
//! After softmax, -∞ → 0:
//! [[1.00, 0.00, 0.00, 0.00],  ← Position 1 only sees itself
//!  [0.30, 0.70, 0.00, 0.00],  ← Position 2 sees 1,2
//!  [0.10, 0.25, 0.65, 0.00],  ← Position 3 sees 1,2,3
//!  [0.05, 0.15, 0.30, 0.50]]  ← Position 4 sees all
//!
//! Used in: GPT, decoder-only models
//! ```
//!
//! ## Cross-Attention
//!
//! **Attend to a different sequence:**
//!
//! ```
//! Encoder-Decoder attention:
//! • Q: From decoder (target sequence)
//! • K, V: From encoder (source sequence)
//!
//! Translation example:
//! Source: "I love cats"  → Encoder → K, V
//! Target: "J'aime les"   → Decoder → Q
//!
//! Decoder attends to source!
//! Aligns target words with source words
//!
//! Used in: Translation, image captioning, text-to-image
//! ```
//!
//! ## Applications
//!
//! ### 1. Machine Translation (Seq2Seq + Attention)
//!
//! ```
//! Without attention:
//! Source → Encoder → Fixed vector → Decoder → Target
//!                       ↑
//!              Information bottleneck!
//!
//! With attention:
//! Source → Encoder → All hidden states
//!                         ↓
//!                    Decoder attends to relevant parts
//!                         ↓
//!                      Target
//!
//! BLEU score improvement: +5-10 points!
//! ```
//!
//! ### 2. Image Captioning
//!
//! ```
//! CNN → Image features (grid: 7×7×512)
//!         ↓
//! Decoder generates caption word by word
//! Each word attends to different image regions
//!
//! "A dog" → Attend to dog region
//! "playing" → Attend to action area
//! "with ball" → Attend to ball
//!
//! Interpretable! Can visualize where model looks
//! ```
//!
//! ### 3. Document Classification
//!
//! ```
//! Self-attention over document:
//! • Find important sentences
//! • Context-aware representations
//! • Better than averaging
//!
//! Hierarchical attention:
//! • Word-level attention (within sentences)
//! • Sentence-level attention (within document)
//! ```
//!
//! ### 4. Question Answering
//!
//! ```
//! Context: Paragraph of text
//! Question: "What is...?"
//!
//! Cross-attention:
//! • Question attends to context
//! • Find relevant spans
//! • Extract answer
//!
//! Used in: BERT, RoBERTa for SQuAD
//! ```
//!
//! ## Attention Visualization
//!
//! **Interpreting what the model learned:**
//!
//! ```
//! Attention weights = how much each position matters
//!
//! Heatmap visualization:
//!             I    love  cats
//! I        [0.5   0.3   0.2]
//! love     [0.2   0.4   0.4]  ← "love" attends to "cats"
//! cats     [0.1   0.2   0.7]  ← "cats" attends to itself
//!
//! Patterns reveal:
//! • Syntactic structure (subject-verb-object)
//! • Semantic relationships (related concepts)
//! • Coreference (pronouns to nouns)
//! ```
//!
//! ## Computational Complexity
//!
//! ```
//! Self-attention complexity:
//!
//! Time: O(n² · d)
//! • n: Sequence length
//! • d: Dimension
//! • n² from comparing all pairs
//!
//! Memory: O(n²)
//! • Store attention matrix
//!
//! Problem for long sequences:
//! • n=512: 262K entries
//! • n=1024: 1M entries
//! • n=4096: 16M entries
//!
//! Solutions:
//! • Sparse attention (Longformer)
//! • Linear attention (Linformer)
//! • Local attention windows
//! • Compressed attention (Reformer)
//! ```
//!
//! ## Implementation Considerations
//!
//! ### Efficient Matrix Operations
//!
//! ```
//! Batch matrix multiplication:
//! • Process multiple sequences at once
//! • GPU-friendly operations
//! • Parallelizable
//!
//! Typical shapes:
//! Q: (batch, heads, seq_len, d_k)
//! K: (batch, heads, seq_len, d_k)
//! V: (batch, heads, seq_len, d_v)
//!
//! Scores: (batch, heads, seq_len, seq_len)
//! Output: (batch, heads, seq_len, d_v)
//! ```
//!
//! ### Dropout in Attention
//!
//! ```
//! Apply dropout to attention weights:
//! weights = softmax(scores)
//! weights = dropout(weights, p=0.1)
//! output = weights · V
//!
//! Benefits:
//! • Regularization
//! • Prevents over-reliance on specific positions
//! • Better generalization
//! ```
//!
//! ## Historical Impact
//!
//! **2014:** Bahdanau attention (machine translation)
//! - First successful attention mechanism
//! - Beat pure seq2seq models
//!
//! **2015:** Luong attention variations
//! - Simpler, more efficient
//! - Global vs local attention
//!
//! **2017:** Transformer ("Attention is All You Need")
//! - Self-attention only, no RNNs
//! - Multi-head attention
//! - Foundation of modern NLP
//!
//! **2018-2020:** Attention everywhere
//! - BERT: Bidirectional attention
//! - GPT: Masked attention
//! - Vision: Attention in CNNs
//!
//! **2021+:** Transformers dominate
//! - NLP: Almost all models use attention
//! - Vision: ViT, DETR
//! - Multi-modal: CLIP, DALL-E
//! - Foundation models: GPT-4, PaLM
//!
//! ## Why Attention Won
//!
//! ```
//! vs RNN/LSTM:
//! ✅ Parallelizable (no sequential dependency)
//! ✅ Better long-range dependencies
//! ✅ No vanishing gradients through layers
//! ✅ Interpretable (visualize attention)
//!
//! vs CNN:
//! ✅ Global receptive field (see entire input)
//! ✅ Position-independent
//! ✅ Dynamic weights (data-dependent)
//!
//! Trade-off:
//! ❌ O(n²) complexity
//! ❌ Need positional information (add encodings)
//! ```

fn main() {
    println!("=== Attention Mechanisms: Foundation of Modern AI ===\n");

    println!("This example explains attention mechanisms, the core innovation");
    println!("powering Transformers, GPT, BERT, and most modern AI.\n");

    println!("📚 Key Concepts Covered:");
    println!("  • Query, Key, Value framework");
    println!("  • Scaled dot-product attention");
    println!("  • Self-attention vs cross-attention");
    println!("  • Multi-head attention");
    println!("  • Masked attention for autoregressive models");
    println!("  • Attention visualization and interpretability\n");

    println!("🎯 Why This Matters:");
    println!("  • Foundation of Transformers (GPT, BERT, T5)");
    println!("  • Replaced RNNs as primary sequence model");
    println!("  • Enables parallelization and long-range dependencies");
    println!("  • Powers all modern NLP, vision transformers, multi-modal AI");
    println!("  • Most important ML innovation of the 2010s\n");

    println!("See the source code documentation for comprehensive explanations!");
}

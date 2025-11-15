//! # Transformer Architecture Example
//!
//! This example demonstrates the Transformer, the architecture that revolutionized NLP
//! and powers GPT, BERT, ChatGPT, and most modern AI systems.
//!
//! ## What is a Transformer?
//!
//! A Transformer is a neural network architecture based entirely on **attention mechanisms**,
//! eliminating the need for recurrence (RNNs) or convolution (CNNs) for sequence processing.
//!
//! **Key Innovation:** "Attention is All You Need" (Vaswani et al., 2017)
//!
//! ## Why Transformers Beat RNNs
//!
//! ```
//! RNN Problems:
//! ✗ Sequential processing (slow)
//! ✗ Vanishing gradients (long sequences)
//! ✗ Can't parallelize
//! ✗ Limited context window
//!
//! Transformer Advantages:
//! ✓ Parallel processing (fast!)
//! ✓ No vanishing gradients
//! ✓ Unlimited context (theoretically)
//! ✓ Better long-range dependencies
//! ```
//!
//! ## Core Concept: Attention
//!
//! **Question:** How do we decide which words to focus on?
//!
//! **Example:**
//! ```
//! "The animal didn't cross the street because it was too tired"
//!                                                ^^
//! What does "it" refer to?
//! - the animal? ✓ (tired)
//! - the street? ✗ (streets can't be tired)
//! ```
//!
//! Attention mechanisms automatically learn these relationships!
//!
//! ## Self-Attention Mechanism
//!
//! For each word, compute how much to "attend" to every other word.
//!
//! ### Step-by-Step:
//!
//! **1. Create Query (Q), Key (K), Value (V) vectors**
//! ```
//! Q = input × W_Q    # What am I looking for?
//! K = input × W_K    # What do I contain?
//! V = input × W_V    # What do I output?
//! ```
//!
//! **2. Compute attention scores**
//! ```
//! scores = Q · K^T / √d_k
//! ```
//! Higher score = more relevant
//!
//! **3. Apply softmax**
//! ```
//! attention_weights = softmax(scores)
//! ```
//! Converts to probabilities
//!
//! **4. Weighted sum of values**
//! ```
//! output = attention_weights · V
//! ```
//!
//! ### Mathematical Formula:
//! ```
//! Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
//! ```
//!
//! ## Multi-Head Attention
//!
//! Instead of one attention mechanism, use multiple "heads":
//!
//! ```
//! Head 1: Learns syntactic relationships
//! Head 2: Learns semantic relationships
//! Head 3: Learns positional relationships
//! ...
//! Head 8: Learns other patterns
//! ```
//!
//! Then concatenate and project:
//! ```
//! MultiHead(Q,K,V) = Concat(head₁,...,headₕ) · W_O
//! ```
//!
//! **Benefits:**
//! - Attend to different aspects simultaneously
//! - More expressive
//! - Better representations
//!
//! ## Positional Encoding
//!
//! **Problem:** Transformers have no built-in notion of order!
//! ```
//! "Dog bites man" vs "Man bites dog"
//! ```
//! Without position info, these look the same!
//!
//! **Solution:** Add positional encodings
//! ```
//! PE(pos, 2i) = sin(pos / 10000^(2i/d))
//! PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
//! ```
//!
//! Properties:
//! - Unique for each position
//! - Allows model to learn relative positions
//! - Works for any sequence length
//!
//! ## Transformer Architecture
//!
//! ### Encoder Stack
//! ```
//! Input Embedding
//!      ↓
//! Positional Encoding
//!      ↓
//! ┌──────────────────────┐
//! │ Multi-Head Attention │
//! │ (Self-Attention)     │
//! └──────────────────────┘
//!      ↓ (Add & Norm)
//! ┌──────────────────────┐
//! │ Feed-Forward Network │
//! │ (Dense → ReLU → Dense)│
//! └──────────────────────┘
//!      ↓ (Add & Norm)
//! (Repeat 6-12 times)
//!      ↓
//! Encoder Output
//! ```
//!
//! ### Decoder Stack
//! ```
//! Output Embedding
//!      ↓
//! Positional Encoding
//!      ↓
//! ┌──────────────────────┐
//! │ Masked Self-Attention│ ← Can't see future!
//! └──────────────────────┘
//!      ↓ (Add & Norm)
//! ┌──────────────────────┐
//! │ Cross-Attention      │ ← Attends to encoder
//! └──────────────────────┘
//!      ↓ (Add & Norm)
//! ┌──────────────────────┐
//! │ Feed-Forward Network │
//! └──────────────────────┘
//!      ↓ (Add & Norm)
//! (Repeat 6-12 times)
//!      ↓
//! Output Probabilities
//! ```
//!
//! ## Key Components
//!
//! ### 1. Layer Normalization
//! ```
//! Normalizes activations across features
//! Stabilizes training
//! Allows deeper networks
//! ```
//!
//! ### 2. Residual Connections
//! ```
//! output = Layer(x) + x
//!
//! Benefits:
//! - Gradient flow
//! - Easier training
//! - Deeper networks
//! ```
//!
//! ### 3. Feed-Forward Network
//! ```
//! FFN(x) = ReLU(x·W₁ + b₁)·W₂ + b₂
//!
//! Typically: d_model → 4·d_model → d_model
//! Example: 512 → 2048 → 512
//! ```
//!
//! ## Training Process
//!
//! ### Teacher Forcing (Training)
//! ```
//! Input:  "How are you"
//! Target: "Comment allez-vous"
//!
//! Decoder sees: [START] Comment allez
//! Should predict:  Comment allez vous
//! ```
//!
//! ### Autoregressive Generation (Inference)
//! ```
//! Step 1: [START] → "Comment"
//! Step 2: [START] Comment → "allez"
//! Step 3: [START] Comment allez → "vous"
//! Step 4: [START] Comment allez vous → [END]
//! ```
//!
//! ## Transformer Variants
//!
//! ### BERT (Bidirectional Encoder)
//! ```
//! Architecture: Encoder only
//! Training: Masked language modeling
//! Use: Understanding (classification, QA)
//!
//! Example: "The [MASK] sat on the mat"
//! Predicts: "cat" with high probability
//! ```
//!
//! ### GPT (Generative Pre-trained Transformer)
//! ```
//! Architecture: Decoder only
//! Training: Next token prediction
//! Use: Generation (text, code, chat)
//!
//! GPT-3: 175B parameters
//! GPT-4: ~1.7T parameters (estimated)
//! ChatGPT: Fine-tuned GPT-3.5/4
//! ```
//!
//! ### T5 (Text-to-Text)
//! ```
//! Architecture: Full encoder-decoder
//! Framing: Everything as text-to-text
//! Use: Translation, summarization, QA
//! ```
//!
//! ### Vision Transformer (ViT)
//! ```
//! Applies transformers to images
//! Treats image patches as tokens
//! Rivals CNNs for image classification
//! ```
//!
//! ### CLIP (Contrastive Language-Image Pre-training)
//! ```
//! Learns vision-language connections
//! Powers DALL-E, Stable Diffusion
//! ```
//!
//! ## Applications
//!
//! ### Natural Language Processing
//! - Machine translation (original use)
//! - Text generation (GPT)
//! - Question answering (BERT)
//! - Summarization (T5, BART)
//! - Named entity recognition
//!
//! ### Code & Programming
//! - Code completion (GitHub Copilot)
//! - Code generation (Codex, GPT-4)
//! - Bug detection
//! - Documentation generation
//!
//! ### Computer Vision
//! - Image classification (ViT)
//! - Object detection (DETR)
//! - Image generation (DALL-E)
//!
//! ### Multimodal
//! - Image captioning
//! - Visual question answering
//! - Text-to-image (DALL-E, Stable Diffusion)
//!
//! ### Speech & Audio
//! - Speech recognition (Whisper)
//! - Text-to-speech
//! - Music generation
//!
//! ### Other Domains
//! - Protein folding (AlphaFold 2)
//! - Drug discovery
//! - Time series forecasting
//! - Reinforcement learning (Decision Transformer)
//!
//! ## Computational Complexity
//!
//! **Self-Attention:**
//! ```
//! Time: O(n²·d)  where n = sequence length
//! Space: O(n²)
//! ```
//!
//! **Problem:** Quadratic in sequence length!
//!
//! **Solutions:**
//! - Sparse attention (only attend to nearby tokens)
//! - Linear attention (approximate attention)
//! - Sliding window (local attention)
//! - Flash Attention (optimized implementation)
//!
//! ## Training Tips
//!
//! 1. **Warmup learning rate**
//!    - Start low, increase linearly
//!    - Then decrease with scheduler
//!
//! 2. **Adam with β₂=0.98**
//!    - Original paper: β₁=0.9, β₂=0.98
//!
//! 3. **Gradient clipping**
//!    - Prevent exploding gradients
//!    - Typical: clip norm to 1.0
//!
//! 4. **Label smoothing**
//!    - Use 0.1 smoothing
//!    - Prevents overconfidence
//!
//! 5. **Dropout**
//!    - Apply to attention, embeddings
//!    - Typical: 0.1
//!
//! 6. **Layer normalization**
//!    - Pre-LN often works better than Post-LN
//!
//! ## Why Transformers Won
//!
//! ```
//! Parallelization:
//! • RNN: Must process sequentially
//! • Transformer: All positions in parallel
//! • Result: 100x faster training!
//!
//! Long-Range Dependencies:
//! • RNN: Gradients vanish over time
//! • Transformer: Direct connections
//! • Result: Better at long sequences!
//!
//! Scalability:
//! • Transformers scale with compute
//! • Bigger models = better performance
//! • Led to GPT-3, GPT-4, etc.
//! ```

fn main() {
    println!("=== Transformer Architecture ===\n");

    println!("This example explains the Transformer, the architecture behind modern AI.\n");

    // Explain attention
    println!("1. The Attention Mechanism\n");

    println!("   Example: 'The animal didn't cross the street because it was too tired'\n");

    println!("   Question: What does 'it' refer to?");
    println!("   - the animal? ✓ (makes sense - animals get tired)");
    println!("   - the street? ✗ (streets don't get tired)\n");

    println!("   Attention automatically learns these relationships!\n");

    println!("   Attention Scores (simplified):");
    println!("   ┌──────────┬────────┬────────┬────────┬──────┐");
    println!("   │ 'it' ↔   │ animal │ street │ cross  │ tired│");
    println!("   ├──────────┼────────┼────────┼────────┼──────┤");
    println!("   │ Score    │  0.7   │  0.05  │  0.05  │ 0.2  │");
    println!("   └──────────┴────────┴────────┴────────┴──────┘\n");

    println!("   'it' pays most attention to 'animal'!\n");

    // Self-attention computation
    println!("2. Self-Attention Computation\n");

    println!("   Step 1: Create Q, K, V matrices");
    println!("   ┌───────────────────────────────────────┐");
    println!("   │ Query  (Q): What am I looking for?    │");
    println!("   │ Key    (K): What do I contain?        │");
    println!("   │ Value  (V): What do I output?         │");
    println!("   └───────────────────────────────────────┘\n");

    println!("   Step 2: Compute attention scores");
    println!("   scores = (Q · K^T) / √d_k\n");

    println!("   Step 3: Apply softmax");
    println!("   attention_weights = softmax(scores)\n");

    println!("   Step 4: Weighted sum of values");
    println!("   output = attention_weights · V\n");

    println!("   Formula: Attention(Q,K,V) = softmax(Q·K^T/√d_k)·V\n");

    // Multi-head attention
    println!("3. Multi-Head Attention\n");

    println!("   Instead of one attention, use multiple 'heads':\n");

    println!("   Head 1: Word relationships  ('cat' → 'sat')");
    println!("   Head 2: Position info       (word 1 → word 5)");
    println!("   Head 3: Semantic similarity ('king' → 'queen')");
    println!("   Head 4: Syntax patterns     (subject → verb)");
    println!("   ... 8 heads total (typically)\n");

    println!("   Benefit: Learn different relationships simultaneously!\n");

    // Architecture
    println!("4. Complete Transformer Architecture\n");

    println!("   ┌─────────────────────────────────┐");
    println!("   │      INPUT SEQUENCE             │");
    println!("   └─────────────────────────────────┘");
    println!("                 ↓");
    println!("   ┌─────────────────────────────────┐");
    println!("   │    Word Embeddings              │");
    println!("   │  + Positional Encoding          │");
    println!("   └─────────────────────────────────┘");
    println!("                 ↓");
    println!("   ╔═════════════════════════════════╗");
    println!("   ║       ENCODER (× 6)             ║");
    println!("   ╠═════════════════════════════════╣");
    println!("   ║ Multi-Head Self-Attention       ║");
    println!("   ║         ↓ (Add & Norm)          ║");
    println!("   ║ Feed-Forward Network            ║");
    println!("   ║         ↓ (Add & Norm)          ║");
    println!("   ╚═════════════════════════════════╝");
    println!("                 ↓");
    println!("   ╔═════════════════════════════════╗");
    println!("   ║       DECODER (× 6)             ║");
    println!("   ╠═════════════════════════════════╣");
    println!("   ║ Masked Self-Attention           ║");
    println!("   ║         ↓ (Add & Norm)          ║");
    println!("   ║ Cross-Attention to Encoder      ║");
    println!("   ║         ↓ (Add & Norm)          ║");
    println!("   ║ Feed-Forward Network            ║");
    println!("   ║         ↓ (Add & Norm)          ║");
    println!("   ╚═════════════════════════════════╝");
    println!("                 ↓");
    println!("   ┌─────────────────────────────────┐");
    println!("   │   Output Probabilities          │");
    println!("   └─────────────────────────────────┘\n");

    // Variants
    println!("5. Transformer Variants\n");

    println!("   ┌─────────────┬──────────────┬────────────────────┐");
    println!("   │ Model       │ Architecture │ Use Case           │");
    println!("   ├─────────────┼──────────────┼────────────────────┤");
    println!("   │ BERT        │ Encoder only │ Understanding      │");
    println!("   │ GPT-3/4     │ Decoder only │ Generation         │");
    println!("   │ T5          │ Full         │ Text-to-text       │");
    println!("   │ BART        │ Full         │ Summarization      │");
    println!("   │ ViT         │ Encoder      │ Image vision       │");
    println!("   │ CLIP        │ Dual encoder │ Vision + Language  │");
    println!("   │ Whisper     │ Full         │ Speech recognition │");
    println!("   └─────────────┴──────────────┴────────────────────┘\n");

    // Applications
    println!("6. Real-World Applications\n");

    println!("   Language:");
    println!("   • ChatGPT, GPT-4 (conversational AI)");
    println!("   • Google Translate (translation)");
    println!("   • BERT (search, QA)\n");

    println!("   Code:");
    println!("   • GitHub Copilot (code completion)");
    println!("   • GPT-4 (code generation)");
    println!("   • Codex (programming assistant)\n");

    println!("   Vision:");
    println!("   • DALL-E 2/3 (text-to-image)");
    println!("   • Stable Diffusion (image generation)");
    println!("   • ViT (image classification)\n");

    println!("   Science:");
    println!("   • AlphaFold 2 (protein folding)");
    println!("   • Drug discovery");
    println!("   • Scientific paper generation\n");

    // Why transformers won
    println!("7. Why Transformers Dominate\n");

    println!("   vs RNNs:");
    println!("   ✓ Parallel processing (100x faster training)");
    println!("   ✓ Better long-range dependencies");
    println!("   ✓ No vanishing gradients");
    println!("   ✓ Unlimited context window (theoretically)\n");

    println!("   vs CNNs:");
    println!("   ✓ Global receptive field immediately");
    println!("   ✓ More flexible attention patterns");
    println!("   ✓ Better for sequential data\n");

    println!("   Scaling:");
    println!("   • GPT-1:   117M parameters (2018)");
    println!("   • GPT-2:   1.5B parameters (2019)");
    println!("   • GPT-3:   175B parameters (2020)");
    println!("   • GPT-4:   ~1.7T parameters (2023, estimated)");
    println!("   • More parameters → Better performance!\n");

    // Computational complexity
    println!("8. Computational Cost\n");

    println!("   Self-Attention Complexity:");
    println!("   • Time:  O(n² · d)  - Quadratic in sequence length!");
    println!("   • Space: O(n²)     - Stores attention matrix\n");

    println!("   Example:");
    println!("   • Sequence length = 512 tokens");
    println!("   • Attention matrix = 512 × 512 = 262,144 values");
    println!("   • With 8 heads × 12 layers = 25M+ values!\n");

    println!("   Solutions:");
    println!("   ✓ Sparse attention (Longformer)");
    println!("   ✓ Linear attention approximations");
    println!("   ✓ Flash Attention (optimized CUDA kernels)");
    println!("   ✓ Sliding window attention\n");

    println!("=== Example Complete! ===");
    println!("\nKey Takeaways:");
    println!("- Transformers use attention instead of recurrence");
    println!("- Self-attention: each token attends to all others");
    println!("- Multi-head attention: learn multiple relationships");
    println!("- Parallel processing enables massive scaling");
    println!("- Powers GPT, BERT, ChatGPT, DALL-E, and more");
    println!("- Revolutionized AI across all domains");
    println!("- 'Attention is All You Need' - one of the most important papers in AI");
}

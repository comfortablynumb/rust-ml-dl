//! # Autoencoder Example
//!
//! This example demonstrates autoencoders, unsupervised neural networks that learn
//! efficient representations of data through compression and reconstruction.
//!
//! ## What is an Autoencoder?
//!
//! An autoencoder is a neural network that learns to compress data into a lower-dimensional
//! representation (encoding) and then reconstruct the original data from this representation.
//!
//! ## Architecture
//!
//! ```
//! Input (784)
//!     ↓
//! ┌─────────────┐
//! │   ENCODER   │  Compresses
//! │  784 → 128  │
//! │  128 → 64   │
//! │   64 → 32   │  ← Bottleneck (latent space)
//! └─────────────┘
//!     ↓
//! ┌─────────────┐
//! │   DECODER   │  Reconstructs
//! │   32 → 64   │
//! │   64 → 128  │
//! │  128 → 784  │
//! └─────────────┘
//!     ↓
//! Output (784)
//! ```
//!
//! ## Key Components
//!
//! ### 1. Encoder
//! Compresses input into a lower-dimensional latent representation:
//! ```
//! z = encoder(x)
//! ```
//! - Input: High-dimensional data (e.g., 784 for 28×28 image)
//! - Output: Low-dimensional code (e.g., 32)
//!
//! ### 2. Latent Space (Bottleneck)
//! The compressed representation that captures essential features:
//! - Forces network to learn important patterns
//! - Dimensionality determines compression ratio
//! - Can be used for visualization, interpolation
//!
//! ### 3. Decoder
//! Reconstructs input from latent representation:
//! ```
//! x̂ = decoder(z)
//! ```
//! - Input: Latent code
//! - Output: Reconstruction of original input
//!
//! ## Loss Function
//!
//! Reconstruction loss (MSE for continuous data):
//! ```
//! L = (1/n) Σ ||x - x̂||²
//! ```
//!
//! Or binary cross-entropy for binary data:
//! ```
//! L = -Σ [x·log(x̂) + (1-x)·log(1-x̂)]
//! ```
//!
//! ## Training Process
//!
//! 1. Input data x
//! 2. Encode: z = encoder(x)
//! 3. Decode: x̂ = decoder(z)
//! 4. Compute loss: L(x, x̂)
//! 5. Backpropagate and update weights
//! 6. Repeat
//!
//! **Goal**: Minimize reconstruction error
//!
//! ## Types of Autoencoders
//!
//! ### 1. Vanilla Autoencoder
//! Basic encoder-decoder architecture described above.
//!
//! ### 2. Denoising Autoencoder (DAE)
//! ```
//! Input: x + noise → Autoencoder → Output: x (clean)
//! ```
//! - Adds noise to input
//! - Trains to reconstruct clean version
//! - Learns robust features
//! - Used for image denoising
//!
//! ### 3. Sparse Autoencoder
//! ```
//! Loss = Reconstruction + λ·Sparsity
//! ```
//! - Encourages sparse activations in latent space
//! - Only few neurons active at a time
//! - Learns interpretable features
//!
//! ### 4. Variational Autoencoder (VAE)
//! ```
//! Encoder: x → (μ, σ)
//! Sample: z ~ N(μ, σ)
//! Decoder: z → x̂
//! Loss: Reconstruction + KL divergence
//! ```
//! - Learns probability distribution
//! - Can generate new samples
//! - Smooth latent space
//!
//! ### 5. Contractive Autoencoder
//! ```
//! Loss = Reconstruction + λ·||∂h/∂x||²
//! ```
//! - Penalizes sensitivity to input changes
//! - Learns robust representations
//!
//! ## Applications
//!
//! ### 1. Dimensionality Reduction
//! Like PCA but non-linear:
//! - Compress 784D → 32D
//! - Visualize in 2D/3D
//! - Feature extraction
//!
//! ### 2. Denoising
//! Remove noise from:
//! - Images (photo restoration)
//! - Audio (noise reduction)
//! - Medical images
//!
//! ### 3. Anomaly Detection
//! ```
//! Normal data: Low reconstruction error
//! Anomalies: High reconstruction error
//! ```
//! - Fraud detection
//! - Manufacturing defects
//! - Network intrusion
//!
//! ### 4. Data Compression
//! - Image compression (JPEG-like)
//! - Video compression
//! - Lossy but learns task-specific compression
//!
//! ### 5. Generative Modeling
//! VAEs can generate new samples:
//! - Generate faces
//! - Create art
//! - Drug discovery
//!
//! ### 6. Pre-training
//! Use autoencoder to pre-train networks:
//! - Learn good initial weights
//! - Transfer learning
//! - Semi-supervised learning
//!
//! ## Advantages
//!
//! - **Unsupervised**: No labels needed
//! - **Flexible**: Works with any data type
//! - **Non-linear**: Learns complex patterns (vs PCA)
//! - **Versatile**: Many applications
//!
//! ## Limitations
//!
//! - **Harder to train**: More complex than PCA
//! - **Requires tuning**: Architecture, hyperparameters
//! - **May not generalize**: Overfits to training data
//! - **Interpretability**: Latent space can be hard to understand
//!
//! ## Comparison with PCA
//!
//! ```
//! PCA:
//! - Linear dimensionality reduction
//! - Fast, closed-form solution
//! - Guaranteed optimal (for linear)
//! - Interpretable components
//!
//! Autoencoder:
//! - Non-linear dimensionality reduction
//! - Requires training
//! - Can learn complex patterns
//! - Less interpretable
//! ```
//!
//! ## Modern Uses
//!
//! - **VAEs + Diffusion**: Stable Diffusion, DALL-E
//! - **Transformer Autoencoders**: BERT, GPT
//! - **Graph Autoencoders**: Node embeddings
//! - **Self-supervised Learning**: SimCLR, BYOL

use ndarray::Array1;

fn main() {
    println!("=== Autoencoder Basics ===\n");

    println!("This example explains autoencoder concepts and architectures.\n");

    // Demonstrate the concept
    println!("1. Autoencoder Concept\n");
    println!("   Goal: Learn to compress and reconstruct data\n");

    println!("   Example: Compress 28×28 image (784 pixels) to 32 numbers\n");

    println!("   Input:  [0.0, 0.1, 0.2, ..., 0.9] (784 values)");
    println!("      ↓ ENCODER");
    println!("   Code:   [1.2, -0.5, 0.8, ..., -1.1] (32 values) ← Compressed!");
    println!("      ↓ DECODER");
    println!("   Output: [0.0, 0.1, 0.2, ..., 0.9] (784 values)\n");

    println!("   Compression ratio: 784 / 32 = 24.5x smaller!\n");

    // Show architecture
    println!("2. Network Architecture\n");

    println!("   Layer         Size      Activation");
    println!("   ─────────────────────────────────────");
    println!("   Input         784       -");
    println!("   Encoder 1     256       ReLU");
    println!("   Encoder 2     128       ReLU");
    println!("   Bottleneck    32        -           ← Latent space");
    println!("   Decoder 1     128       ReLU");
    println!("   Decoder 2     256       ReLU");
    println!("   Output        784       Sigmoid\n");

    println!("   Parameters:");
    let params = 784*256 + 256*128 + 128*32 + 32*128 + 128*256 + 256*784;
    println!("   - Total: ~{} parameters", params);

    // Compare encoder types
    println!("\n3. Types of Autoencoders\n");

    println!("   A) Vanilla Autoencoder");
    println!("      x → Encode → z → Decode → x̂");
    println!("      Use: Basic dimensionality reduction\n");

    println!("   B) Denoising Autoencoder (DAE)");
    println!("      x + noise → Encode → z → Decode → x (clean)");
    println!("      Use: Remove noise from images/audio\n");

    println!("   C) Variational Autoencoder (VAE)");
    println!("      x → Encode → (μ, σ) → Sample z ~ N(μ,σ) → Decode → x̂");
    println!("      Use: Generate new samples\n");

    println!("   D) Sparse Autoencoder");
    println!("      x → Encode (with sparsity) → z → Decode → x̂");
    println!("      Use: Learn interpretable features\n");

    // Applications
    println!("4. Key Applications\n");

    println!("   A) Anomaly Detection");
    println!("      ┌─────────────┬──────────────────┐");
    println!("      │ Data Type   │ Reconstruction   │");
    println!("      ├─────────────┼──────────────────┤");
    println!("      │ Normal      │ Low error (good) │");
    println!("      │ Anomaly     │ High error  (⚠)  │");
    println!("      └─────────────┴──────────────────┘");
    println!("      Example: Fraud detection, defect detection\n");

    println!("   B) Denoising");
    println!("      Noisy Image → Autoencoder → Clean Image");
    println!("      Example: Photo restoration, medical imaging\n");

    println!("   C) Dimensionality Reduction");
    println!("      High-D Data → Encoder → Low-D Code");
    println!("      Example: Visualization, feature extraction\n");

    println!("   D) Generation (VAE)");
    println!("      Random z → Decoder → New Sample");
    println!("      Example: Generate faces, art, molecules\n");

    // Training process
    println!("5. Training Process\n");

    println!("   For each batch:");
    println!("   1. Input: x (original data)");
    println!("   2. Forward:");
    println!("      z = encoder(x)           # Compress");
    println!("      x̂ = decoder(z)           # Reconstruct");
    println!("   3. Loss:");
    println!("      L = MSE(x, x̂)           # Reconstruction error");
    println!("   4. Backprop:");
    println!("      Update encoder & decoder weights");
    println!("   5. Repeat until loss converges\n");

    // Latent space visualization
    println!("6. Latent Space Properties\n");

    println!("   2D Latent Space Example:");
    println!("   ");
    println!("        │   😊");
    println!("    z₂  │😊   😊     😊 = Happy faces");
    println!("        │        😢  😢 = Sad faces");
    println!("        │      😢  😢");
    println!("        └──────────── z₁");
    println!("   ");
    println!("   - Similar inputs cluster together");
    println!("   - Can interpolate between points");
    println!("   - Can sample new points for generation\n");

    // Comparison table
    println!("7. Autoencoder vs PCA\n");

    println!("   ┌─────────────────┬────────────┬──────────────┐");
    println!("   │ Feature         │ PCA        │ Autoencoder  │");
    println!("   ├─────────────────┼────────────┼──────────────┤");
    println!("   │ Type            │ Linear     │ Non-linear   │");
    println!("   │ Training        │ Closed-form│ Iterative    │");
    println!("   │ Speed           │ Fast       │ Slow         │");
    println!("   │ Flexibility     │ Low        │ High         │");
    println!("   │ Interpretability│ High       │ Low          │");
    println!("   │ Performance     │ Good       │ Better*      │");
    println!("   └─────────────────┴────────────┴──────────────┘");
    println!("   *On complex, non-linear data\n");

    // Real-world examples
    println!("8. Real-World Impact\n");

    println!("   Image Generation:");
    println!("   - VAE + Diffusion → Stable Diffusion, DALL-E");
    println!("   - Generate photorealistic images from text\n");

    println!("   Compression:");
    println!("   - Google's Raisr: Super-resolution");
    println!("   - Better than JPEG for specific domains\n");

    println!("   Science:");
    println!("   - Drug discovery: Generate new molecules");
    println!("   - Protein folding: AlphaFold uses autoencoder-like components\n");

    println!("   Security:");
    println!("   - Anomaly detection in networks");
    println!("   - Fraud detection in finance\n");

    println!("9. Tips for Training\n");

    println!("   ✓ Start simple, then add complexity");
    println!("   ✓ Match activation to data (Sigmoid for [0,1], Tanh for [-1,1])");
    println!("   ✓ Use appropriate loss (MSE or Binary Cross-Entropy)");
    println!("   ✓ Regularize bottleneck (dropout, weight decay)");
    println!("   ✓ Monitor reconstruction error on validation set");
    println!("   ✓ Visualize latent space to understand learning");

    println!("\n=== Example Complete! ===");
    println!("\nKey Takeaways:");
    println!("- Autoencoders compress data into latent representations");
    println!("- Encoder compresses, decoder reconstructs");
    println!("- Trained by minimizing reconstruction error");
    println!("- Unsupervised - no labels needed!");
    println!("- Many variants: Denoising, VAE, Sparse, etc.");
    println!("- Applications: Compression, denoising, generation, anomaly detection");
    println!("- VAEs power modern generative AI (Stable Diffusion)");
}

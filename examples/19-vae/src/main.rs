//! # VAE: Variational Autoencoder
//!
//! This example explains Variational Autoencoders (VAE), a probabilistic generative
//! model that combines deep learning with variational inference to learn latent
//! representations and generate new data.
//!
//! ## Problem: Generating New Data
//!
//! **Goal:** Learn to generate realistic new samples
//!
//! ```
//! Training data: Images of handwritten digits
//! Goal: Generate NEW digit images that look real
//!
//! Challenge:
//! • Can't just memorize training data
//! • Need to learn the "essence" of digits
//! • Must be able to sample from learned distribution
//! ```
//!
//! ## Autoencoder Limitations
//!
//! **Standard Autoencoder:**
//! ```
//! Encoder: x → z (latent code)
//! Decoder: z → x̂ (reconstruction)
//!
//! Training: Minimize ||x - x̂||²
//!
//! Problem for generation:
//! • Latent space has "holes"
//! • Only trained points work well
//! • Random sampling → garbage
//!
//! Example:
//! Training encodes "5" → z=[2.1, 3.4]
//! Random z=[2.0, 3.0] → Decoder output: nonsense ❌
//! ```
//!
//! ## VAE: The Probabilistic Solution
//!
//! **Key Innovation:** Treat latent variables as probability distributions
//!
//! ```
//! Instead of: x → z (point)
//! VAE learns: x → p(z|x) (distribution)
//!
//! Specifically: p(z|x) = N(μ(x), σ²(x))
//! Where:
//! • μ(x): Mean vector (encoder output 1)
//! • σ²(x): Variance vector (encoder output 2)
//! • N(...): Normal/Gaussian distribution
//! ```
//!
//! ## VAE Architecture
//!
//! ```
//!              Input x (e.g., 28×28 image)
//!                      ↓
//!            ┌─────────────────┐
//!            │     Encoder     │
//!            └─────────────────┘
//!                   ↓   ↓
//!                   μ   σ  ← TWO outputs
//!                   ↓   ↓
//!              ┌─────────────┐
//!              │ Reparameterize │
//!              │ z = μ + σ⊙ε  │  ε ~ N(0,1)
//!              └─────────────┘
//!                      ↓
//!                      z (latent code)
//!                      ↓
//!            ┌─────────────────┐
//!            │     Decoder     │
//!            └─────────────────┘
//!                      ↓
//!              Reconstruction x̂
//! ```
//!
//! ### Encoder Network
//! ```
//! Input: x (e.g., 784-dim flattened image)
//! Hidden: Dense(512, ReLU) → Dense(256, ReLU)
//! Outputs:
//!   μ = Dense(latent_dim)  ← Mean
//!   log_σ² = Dense(latent_dim)  ← Log variance (for numerical stability)
//!
//! Why log variance?
//! • σ must be positive
//! • log(σ²) can be any real number
//! • More stable training
//! ```
//!
//! ### Reparameterization Trick
//! ```
//! Problem: Can't backpropagate through sampling!
//!
//! Naive: z ~ N(μ, σ²)
//! ❌ Sampling is not differentiable!
//!
//! Reparameterization trick:
//! ε ~ N(0, 1)  ← Sample from standard normal
//! z = μ + σ ⊙ ε  ← Deterministic transformation
//! ✅ Backprop works! Gradients flow through μ and σ
//!
//! In code:
//! epsilon = random_normal(0, 1)
//! z = mu + sigma * epsilon
//! ```
//!
//! ### Decoder Network
//! ```
//! Input: z (latent_dim)
//! Hidden: Dense(256, ReLU) → Dense(512, ReLU)
//! Output: Dense(784, Sigmoid)  ← Reconstruction
//!
//! Output range: [0, 1] (for images)
//! ```
//!
//! ## The VAE Loss Function
//!
//! **Two components:**
//!
//! ```
//! Total Loss = Reconstruction Loss + KL Divergence
//!
//! L(x) = L_recon + β × KL
//! ```
//!
//! ### 1. Reconstruction Loss
//! ```
//! How well does x̂ match x?
//!
//! For binary images (MNIST):
//! L_recon = -Σ [x_i log(x̂_i) + (1-x_i) log(1-x̂_i)]
//!           ↑
//!    Binary cross-entropy per pixel
//!
//! For continuous images:
//! L_recon = ||x - x̂||²  (MSE)
//! ```
//!
//! ### 2. KL Divergence
//! ```
//! Kullback-Leibler Divergence:
//! Measures difference between two distributions
//!
//! KL[q(z|x) || p(z)]
//! Where:
//! • q(z|x) = N(μ, σ²): Encoder's distribution
//! • p(z) = N(0, I): Standard normal prior
//!
//! Intuition: Pull encoded distributions toward standard normal
//!
//! For Gaussian case (closed form):
//! KL = 0.5 × Σ [μ² + σ² - log(σ²) - 1]
//!
//! Why?
//! • Regularization: Prevent overfitting
//! • Continuity: Make latent space smooth
//! • Sampling: Can sample z ~ N(0,1) at test time
//! ```
//!
//! ### Why KL Divergence?
//!
//! ```
//! Without KL (just reconstruction):
//! • Encoder learns arbitrary μ, σ
//! • Latent space has "holes"
//! • Random z → bad reconstructions
//!
//! With KL divergence:
//! • Forces μ close to 0, σ close to 1
//! • Creates smooth, continuous latent space
//! • Any z ~ N(0,1) decodes to reasonable output
//!
//! Example:
//! Train on digits 0-9
//! Latent space organized:
//!
//!     0   1   2   3   4   5   6   7   8   9
//!     ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
//!   [──────────────────────────────────────]
//!              Smooth latent space
//!
//! Sample z in between → Interpolated digits!
//! ```
//!
//! ## The β-VAE Variant
//!
//! ```
//! L = L_recon + β × KL
//!
//! β=1: Standard VAE
//! β>1: β-VAE (emphasize disentanglement)
//! β<1: Emphasize reconstruction
//!
//! β=2-4: Often gives more disentangled features
//! Example: Separate latent dims for "rotation", "thickness", "digit type"
//! ```
//!
//! ## Training VAE
//!
//! ### Training Loop
//! ```
//! for batch in dataset:
//!     # Forward pass
//!     μ, log_σ² = encoder(x)
//!     σ = exp(0.5 × log_σ²)
//!     ε = random_normal(0, 1)
//!     z = μ + σ ⊙ ε
//!     x̂ = decoder(z)
//!
//!     # Compute losses
//!     L_recon = binary_crossentropy(x, x̂)
//!     KL = 0.5 × sum(μ² + σ² - log_σ² - 1)
//!     loss = L_recon + KL
//!
//!     # Backpropagation
//!     loss.backward()
//!     optimizer.step()
//! ```
//!
//! ### Hyperparameters
//! ```
//! Latent dimension: 2-512
//! • 2: Visualizable, limited capacity
//! • 10-20: Good for MNIST
//! • 128-512: Complex images (faces, etc.)
//!
//! Learning rate: 0.001 (Adam)
//! Batch size: 32-128
//! Epochs: 50-200
//! β: 1.0 (standard), 2-4 (disentangling)
//! ```
//!
//! ## Generation: Creating New Samples
//!
//! ### Random Generation
//! ```
//! # Sample from standard normal
//! z = random_normal(0, 1, size=latent_dim)
//!
//! # Decode to image
//! x_new = decoder(z)
//!
//! Result: Random but realistic sample!
//! ```
//!
//! ### Latent Space Interpolation
//! ```
//! # Encode two images
//! z1 = encoder(image1).μ  # "3"
//! z2 = encoder(image2).μ  # "8"
//!
//! # Interpolate in latent space
//! for α in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
//!     z = α×z1 + (1-α)×z2
//!     image = decoder(z)
//!     display(image)
//!
//! Result: Smooth morphing from 3 to 8!
//! ```
//!
//! ### Latent Space Arithmetic
//! ```
//! Given trained on faces:
//!
//! z_smiling = encoder(smiling_face)
//! z_neutral = encoder(neutral_face)
//! z_man = encoder(man_face)
//!
//! # "smile vector"
//! smile_vec = z_smiling - z_neutral
//!
//! # Add smile to man
//! z_smiling_man = z_man + smile_vec
//! smiling_man = decoder(z_smiling_man)
//!
//! Similar to Word2Vec: king - man + woman = queen
//! ```
//!
//! ## Conditional VAE (CVAE)
//!
//! **Add label information:**
//!
//! ```
//! Standard VAE:
//! Encoder: x → z
//! Decoder: z → x̂
//!
//! Conditional VAE:
//! Encoder: [x, y] → z  ← Concatenate input and label
//! Decoder: [z, y] → x̂  ← Concatenate latent and label
//!
//! Generation with CVAE:
//! z = random_normal(0, 1)
//! y = 5  ← Choose digit label
//! x = decoder([z, y])  ← Generate a "5"
//!
//! Control what you generate!
//! ```
//!
//! ## Applications
//!
//! ### Image Generation
//! ```
//! MNIST (digits):
//! • Latent dim: 10-20
//! • Generates realistic digits
//! • Interpolation between digits
//!
//! CelebA (faces):
//! • Latent dim: 128-256
//! • Generate new faces
//! • Control attributes (smile, glasses, age)
//! ```
//!
//! ### Anomaly Detection
//! ```
//! Train VAE on normal data
//!
//! At test time:
//! reconstruction_error = ||x - x̂||²
//!
//! If reconstruction_error > threshold:
//!     → Anomaly! (VAE can't reconstruct well)
//!
//! Use cases:
//! • Manufacturing defect detection
//! • Network intrusion detection
//! • Medical imaging abnormalities
//! ```
//!
//! ### Data Compression
//! ```
//! Encoder compresses: 784 dims → 10 dims
//! Decoder decompresses: 10 dims → 784 dims
//!
//! Similar to JPEG but learned end-to-end
//! Better for domain-specific data
//! ```
//!
//! ### Representation Learning
//! ```
//! Use μ (latent mean) as features:
//!
//! 1. Train VAE on images (unsupervised)
//! 2. Extract z = encoder(x).μ
//! 3. Train classifier on z
//!
//! Benefits:
//! • Low-dimensional features
//! • Captures data structure
//! • Works with unlabeled data
//! ```
//!
//! ### Drug Discovery
//! ```
//! Molecular VAE:
//! • Encode molecules as SMILES strings
//! • Learn latent representation
//! • Generate new molecules
//! • Interpolate between known drugs
//!
//! Result: Discover novel drug candidates
//! ```
//!
//! ## VAE Variants & Extensions
//!
//! ### β-VAE
//! ```
//! L = L_recon + β × KL  (β > 1)
//!
//! Benefits:
//! • More disentangled latent factors
//! • Separate dims for different attributes
//! • Better interpretability
//!
//! Example: One dim for rotation, one for thickness
//! ```
//!
//! ### Hierarchical VAE
//! ```
//! Multiple latent variables at different scales:
//!
//! x → z1 (low-level features)
//!   → z2 (mid-level features)
//!   → z3 (high-level features)
//!
//! Better for complex, multi-scale data
//! ```
//!
//! ### VQ-VAE (Vector Quantized)
//! ```
//! Discrete latent space (codebook):
//!
//! Instead of: z ~ continuous Gaussian
//! Use: z ∈ {e1, e2, ..., eK}  ← Discrete codes
//!
//! Benefits:
//! • Easier to model with autoregressive models
//! • Powers DALL-E
//! • Better for high-quality images
//! ```
//!
//! ### Importance Weighted VAE (IWAE)
//! ```
//! Use multiple samples to estimate loss:
//! • Tighter bound on log-likelihood
//! • Better performance
//! • More computation
//! ```
//!
//! ## VAE vs Other Generative Models
//!
//! ### VAE vs GAN
//! ```
//! VAE:
//! ✅ Stable training
//! ✅ Probabilistic framework
//! ✅ Meaningful latent space
//! ✅ Works with small data
//! ❌ Blurry outputs
//! ❌ Lower sample quality
//!
//! GAN:
//! ✅ Sharp, high-quality outputs
//! ✅ State-of-the-art images
//! ❌ Training instability
//! ❌ Mode collapse
//! ❌ Less meaningful latent space
//!
//! Use VAE when:
//! • Need stable training
//! • Want latent representation
//! • Need to encode/decode
//! • Anomaly detection
//!
//! Use GAN when:
//! • Need highest quality
//! • Only generation (no encoding)
//! • Have expertise to tune training
//! ```
//!
//! ### VAE vs Diffusion Models
//! ```
//! VAE:
//! ✅ Fast generation (one forward pass)
//! ✅ Explicit latent space
//! ✅ Can encode and decode
//! ❌ Blurry outputs
//!
//! Diffusion (Stable Diffusion, DALL-E):
//! ✅ State-of-the-art quality
//! ✅ Sharp, detailed images
//! ✅ Controllable generation
//! ❌ Slow (many steps)
//! ❌ Can't easily encode
//!
//! Hybrid: Stable Diffusion uses VAE!
//! • VAE compresses image to latent
//! • Diffusion works in latent space
//! • VAE decoder produces final image
//! ```
//!
//! ### VAE vs Standard Autoencoder
//! ```
//! Standard AE:
//! • Deterministic: x → z → x̂
//! • Can't generate (holes in latent space)
//! • Better reconstruction
//! • Use for compression, denoising
//!
//! VAE:
//! • Probabilistic: x → p(z|x) → x̂
//! • Can generate (smooth latent space)
//! • Slightly worse reconstruction
//! • Use for generation, anomaly detection
//! ```
//!
//! ## Mathematical Foundation
//!
//! ### Evidence Lower Bound (ELBO)
//! ```
//! VAE maximizes ELBO, a lower bound on log p(x):
//!
//! log p(x) ≥ ELBO = 𝔼_q[log p(x|z)] - KL[q(z|x) || p(z)]
//!                   ↑                   ↑
//!              Reconstruction      Regularization
//!
//! Where:
//! • p(x|z): Decoder (likelihood)
//! • q(z|x): Encoder (approximate posterior)
//! • p(z): Prior (standard normal)
//!
//! VAE training = maximize ELBO = minimize -ELBO
//! ```
//!
//! ### Why "Variational"?
//! ```
//! Variational inference: Approximating intractable posteriors
//!
//! True posterior: p(z|x) = p(x|z)p(z) / p(x)
//!                                       ↑
//!                           Intractable! (requires integrating over all z)
//!
//! Solution: Learn approximate q(z|x) ≈ p(z|x)
//! This is the encoder!
//!
//! "Variational" = Using variational inference
//! ```
//!
//! ## Training Tips
//!
//! ### KL Annealing
//! ```
//! Problem: KL term can dominate early, preventing learning
//!
//! Solution: Gradually increase KL weight
//! L = L_recon + γ(epoch) × KL
//!
//! γ(epoch):
//! Epochs 0-10: 0 → 0.1
//! Epochs 10-50: 0.1 → 1.0
//! Epochs 50+: 1.0
//!
//! Allows reconstruction to improve first
//! ```
//!
//! ### Free Bits
//! ```
//! Problem: Some latent dimensions collapse (not used)
//!
//! Solution: Ensure minimum KL per dimension
//! KL_free_bits = max(KL_dim, λ)
//!
//! λ = 0.5 (typical)
//! Forces each dimension to encode at least λ bits
//! ```
//!
//! ### Batch Normalization
//! ```
//! Add to encoder/decoder:
//! Dense → BatchNorm → ReLU
//!
//! Benefits:
//! • Faster training
//! • Higher learning rates
//! • Better convergence
//! ```
//!
//! ## Debugging VAE Training
//!
//! ### Problem: Blurry Reconstructions
//! ```
//! Solutions:
//! • Increase latent dimension
//! • Reduce β (less KL weight)
//! • More encoder/decoder capacity
//! • Try different loss (perceptual loss)
//! ```
//!
//! ### Problem: Posterior Collapse
//! ```
//! Symptom: KL → 0, latent not used
//!
//! Solutions:
//! • KL annealing
//! • Free bits
//! • Reduce decoder capacity
//! • Increase β gradually
//! ```
//!
//! ### Problem: Poor Generation
//! ```
//! Solutions:
//! • Check KL divergence (should be > 0)
//! • Ensure latent space is continuous
//! • Train longer
//! • Use conditional VAE for more control
//! ```
//!
//! ## Modern Impact
//!
//! **2013:** VAE introduced (Kingma & Welling)
//! - Probabilistic framework for deep generative models
//! - Reparameterization trick enabling backprop
//!
//! **2015-2017:** Extensions
//! - β-VAE for disentanglement
//! - Conditional VAE
//! - Hierarchical VAE
//!
//! **2018-2019:** Applied to complex domains
//! - VQ-VAE for images
//! - MolecularVAE for drug discovery
//! - Text VAE for language
//!
//! **2020-2021:** Hybrid models
//! - DALL-E uses VQ-VAE
//! - Stable Diffusion uses VAE encoder/decoder
//! - VAE as preprocessing for other models
//!
//! **2022+:** Still relevant
//! - Component in modern systems
//! - Anomaly detection
//! - Representation learning
//! - Fast generation when speed matters
//!
//! **Legacy:**
//! - Showed deep learning + probabilistic modeling work well together
//! - Reparameterization trick widely used
//! - Foundation for modern generative AI

fn main() {
    println!("=== Variational Autoencoder (VAE) ===\n");

    println!("This example explains VAE, a probabilistic generative model that");
    println!("learns smooth latent representations for generation and anomaly detection.\n");

    println!("📚 Key Concepts Covered:");
    println!("  • Probabilistic encoder (μ, σ)");
    println!("  • Reparameterization trick");
    println!("  • ELBO loss (reconstruction + KL divergence)");
    println!("  • Latent space interpolation");
    println!("  • Generation vs standard autoencoders");
    println!("  • β-VAE for disentanglement\n");

    println!("🎯 Why This Matters:");
    println!("  • Foundation of probabilistic deep learning");
    println!("  • Powers modern systems (Stable Diffusion uses VAE)");
    println!("  • Enables controlled generation");
    println!("  • Critical for anomaly detection");
    println!("  • Smooth latent space for interpolation\n");

    println!("See the source code documentation for comprehensive explanations!");
}

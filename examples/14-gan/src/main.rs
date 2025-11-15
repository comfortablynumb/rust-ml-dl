//! # Generative Adversarial Network (GAN) Example
//!
//! This example demonstrates GANs, one of the most exciting innovations in deep learning
//! for generating realistic data like images, text, and audio.
//!
//! ## What is a GAN?
//!
//! A GAN consists of two neural networks competing against each other:
//! - **Generator (G)**: Creates fake data to fool the discriminator
//! - **Discriminator (D)**: Tries to distinguish real data from fake data
//!
//! ```
//! Random Noise → [Generator] → Fake Data
//!                                  ↓
//! Real Data ─────────────→ [Discriminator] → Real or Fake?
//! ```
//!
//! ## The Adversarial Game
//!
//! Think of it like a counterfeiter (generator) vs detective (discriminator):
//!
//! **Generator's goal:** Create fake money so good the detective can't tell
//! **Discriminator's goal:** Detect all fake money
//!
//! As they compete:
//! - Discriminator gets better at detecting fakes
//! - Generator gets better at creating realistic fakes
//! - Eventually: Generator creates perfect fakes!
//!
//! ## Architecture
//!
//! ### Generator
//! ```
//! Random Noise (z) [100D]
//!        ↓
//!   Dense(256) + ReLU
//!        ↓
//!   Dense(512) + ReLU
//!        ↓
//!   Dense(784) + Tanh
//!        ↓
//!   Generated Image [28×28]
//! ```
//!
//! Takes random noise and generates realistic data.
//!
//! ### Discriminator
//! ```
//! Input Image [28×28 = 784]
//!        ↓
//!   Dense(512) + LeakyReLU
//!        ↓
//!   Dense(256) + LeakyReLU
//!        ↓
//!   Dense(1) + Sigmoid
//!        ↓
//!   Probability [0 = Fake, 1 = Real]
//! ```
//!
//! Binary classifier: real or fake?
//!
//! ## Training Process
//!
//! For each batch:
//!
//! **1. Train Discriminator:**
//! ```python
//! # Train on real data
//! real_loss = -log(D(real_data))
//!
//! # Train on fake data
//! fake_data = G(noise)
//! fake_loss = -log(1 - D(fake_data))
//!
//! # Total discriminator loss
//! D_loss = real_loss + fake_loss
//! ```
//!
//! **2. Train Generator:**
//! ```python
//! # Generate fake data
//! fake_data = G(noise)
//!
//! # Try to fool discriminator
//! G_loss = -log(D(fake_data))
//! ```
//!
//! ## Loss Functions
//!
//! ### Original GAN (Minimax)
//! ```
//! min_G max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]
//! ```
//!
//! - **Discriminator maximizes:** Correctly classify real vs fake
//! - **Generator minimizes:** Fool the discriminator
//!
//! ### Non-Saturating GAN (Practical)
//! ```
//! Generator Loss: -log(D(G(z)))
//! ```
//! Better gradients for generator in early training.
//!
//! ## Training Challenges
//!
//! ### 1. Mode Collapse
//! **Problem:** Generator produces limited variety
//! - Only generates a few types of outputs
//! - Ignores diversity in training data
//!
//! **Example:** Only generates "3"s, ignores other digits
//!
//! **Solutions:**
//! - Feature matching
//! - Minibatch discrimination
//! - Unrolled GANs
//!
//! ### 2. Vanishing Gradients
//! **Problem:** When D is too good, G gets no learning signal
//! - D perfectly classifies real vs fake
//! - Gradients to G vanish
//!
//! **Solutions:**
//! - Non-saturating loss
//! - Wasserstein GAN (WGAN)
//!
//! ### 3. Training Instability
//! **Problem:** Networks oscillate, don't converge
//! - D and G fight but don't reach equilibrium
//!
//! **Solutions:**
//! - Careful learning rate tuning
//! - Spectral normalization
//! - Two time-scale update rule (TTUR)
//!
//! ## GAN Variants
//!
//! ### DCGAN (Deep Convolutional GAN)
//! ```
//! - Use Conv/ConvTranspose layers
//! - BatchNorm in G and D
//! - ReLU in G, LeakyReLU in D
//! - Remove fully connected layers
//! ```
//! **Use:** High-quality image generation
//!
//! ### CGAN (Conditional GAN)
//! ```
//! Input: [noise, label]
//! Output: Image of specified class
//! ```
//! **Use:** Generate specific types of data
//!
//! ### CycleGAN
//! ```
//! Learns mapping between two domains
//! Horse ↔ Zebra, Summer ↔ Winter
//! ```
//! **Use:** Style transfer, domain translation
//!
//! ### StyleGAN
//! ```
//! Controls different levels of image style
//! - Coarse: Pose, general shape
//! - Medium: Facial features
//! - Fine: Color, micro-structure
//! ```
//! **Use:** High-quality face generation (ThisPersonDoesNotExist.com)
//!
//! ### WGAN (Wasserstein GAN)
//! ```
//! Uses Wasserstein distance
//! More stable training
//! ```
//! **Benefit:** Solves vanishing gradients
//!
//! ### Progressive GAN
//! ```
//! Starts at low resolution (4×4)
//! Progressively adds layers
//! Ends at high resolution (1024×1024)
//! ```
//! **Use:** Extremely high-quality images
//!
//! ## Applications
//!
//! ### Image Generation
//! - Generate faces (StyleGAN)
//! - Create art (DALL-E, Midjourney)
//! - Generate landscapes, objects
//!
//! ### Image-to-Image Translation
//! - Pix2Pix: Sketch → Photo
//! - CycleGAN: Summer → Winter
//! - Super-resolution: Low-res → High-res
//!
//! ### Data Augmentation
//! - Generate training data
//! - Balance imbalanced datasets
//! - Synthetic medical images
//!
//! ### Video Generation
//! - Video prediction
//! - Frame interpolation
//! - DeepFake (ethical concerns!)
//!
//! ### Drug Discovery
//! - Generate novel molecules
//! - Optimize chemical properties
//! - Protein design
//!
//! ### Text Generation
//! - SeqGAN: Generate sequences
//! - MaskGAN: Fill in missing text
//!
//! ## Evaluation Metrics
//!
//! ### Inception Score (IS)
//! ```
//! Measures quality and diversity
//! Higher is better
//! Range: [1, ∞)
//! ```
//!
//! ### Fréchet Inception Distance (FID)
//! ```
//! Compares real vs generated distributions
//! Lower is better
//! Range: [0, ∞)
//! ```
//!
//! ### Precision & Recall
//! - **Precision:** Quality (are fakes realistic?)
//! - **Recall:** Diversity (covers all modes?)
//!
//! ## Training Tips
//!
//! 1. **Normalize inputs** to [-1, 1] (use Tanh in G)
//! 2. **Use LeakyReLU** in discriminator (slope 0.2)
//! 3. **Label smoothing:** Use 0.9 instead of 1.0 for real labels
//! 4. **Train D more:** 1-5 D updates per G update
//! 5. **Use BatchNorm** in both networks
//! 6. **Avoid sparse gradients:** No max pooling, use stride
//! 7. **Monitor losses:** Both should decrease together
//! 8. **Use spectral normalization** for stability
//!
//! ## Modern State-of-the-Art
//!
//! ### Stable Diffusion (2022)
//! - Uses diffusion models (not traditional GAN)
//! - Text-to-image generation
//! - Open source, runs on consumer GPUs
//!
//! ### DALL-E 2/3 (OpenAI)
//! - Text-to-image with incredible quality
//! - Combines diffusion + transformers
//!
//! ### Midjourney
//! - Artistic image generation
//! - Commercial service
//!
//! Note: Modern generative models often use **diffusion** instead of GANs,
//! but GANs paved the way and are still widely used!

fn main() {
    println!("=== Generative Adversarial Networks (GAN) ===\n");

    println!("This example explains GAN architecture and training.\n");

    // Explain the concept
    println!("1. The GAN Game: Generator vs Discriminator\n");

    println!("   Imagine a counterfeiter (G) and detective (D):\n");

    println!("   Initial State:");
    println!("   - G creates terrible fakes (random noise)");
    println!("   - D easily spots all fakes\n");

    println!("   After Training:");
    println!("   - G creates perfect fakes");
    println!("   - D can only guess (50% accuracy)\n");

    println!("   At equilibrium: G has learned the data distribution!\n");

    // Show architecture
    println!("2. GAN Architecture\n");

    println!("   Generator Network:");
    println!("   ┌────────────────────┐");
    println!("   │ Noise z (100)      │ ← Random input");
    println!("   └────────────────────┘");
    println!("            ↓");
    println!("   ┌────────────────────┐");
    println!("   │ Dense(256) + ReLU  │");
    println!("   └────────────────────┘");
    println!("            ↓");
    println!("   ┌────────────────────┐");
    println!("   │ Dense(512) + ReLU  │");
    println!("   └────────────────────┘");
    println!("            ↓");
    println!("   ┌────────────────────┐");
    println!("   │ Dense(784) + Tanh  │");
    println!("   └────────────────────┘");
    println!("            ↓");
    println!("   Fake Image (28×28)\n");

    println!("   Discriminator Network:");
    println!("   ┌────────────────────────┐");
    println!("   │ Image (784)            │ ← Real or Fake");
    println!("   └────────────────────────┘");
    println!("            ↓");
    println!("   ┌────────────────────────┐");
    println!("   │ Dense(512) + LeakyReLU │");
    println!("   └────────────────────────┘");
    println!("            ↓");
    println!("   ┌────────────────────────┐");
    println!("   │ Dense(256) + LeakyReLU │");
    println!("   └────────────────────────┘");
    println!("            ↓");
    println!("   ┌────────────────────────┐");
    println!("   │ Dense(1) + Sigmoid     │");
    println!("   └────────────────────────┘");
    println!("            ↓");
    println!("   Probability [0=Fake, 1=Real]\n");

    // Training process
    println!("3. Training Process (Alternating)\n");

    println!("   Iteration 1:");
    println!("   ┌─────────────────────────────────────────┐");
    println!("   │ Step 1: Train Discriminator             │");
    println!("   ├─────────────────────────────────────────┤");
    println!("   │ • Feed real data → D should output 1    │");
    println!("   │ • Generate fakes → D should output 0    │");
    println!("   │ • Update D to be better at classifying  │");
    println!("   └─────────────────────────────────────────┘");
    println!("   ┌─────────────────────────────────────────┐");
    println!("   │ Step 2: Train Generator                 │");
    println!("   ├─────────────────────────────────────────┤");
    println!("   │ • Generate fakes                         │");
    println!("   │ • D evaluates them                       │");
    println!("   │ • Update G to fool D better              │");
    println!("   └─────────────────────────────────────────┘");
    println!("   Repeat 10,000+ times...\n");

    // Show loss evolution
    println!("4. Loss Evolution During Training\n");

    println!("   Early Training:");
    println!("   D_loss: High (hard to distinguish)");
    println!("   G_loss: High (D easily spots fakes)\n");

    println!("   Mid Training:");
    println!("   D_loss: Moderate (getting better)");
    println!("   G_loss: Decreasing (fakes improving)\n");

    println!("   Late Training (Equilibrium):");
    println!("   D_loss: ~0.693 (log(2), random guessing)");
    println!("   G_loss: ~0.693 (successfully fooling D)\n");

    // Common issues
    println!("5. Training Challenges\n");

    println!("   A) Mode Collapse");
    println!("   ┌────────────────────────────────────┐");
    println!("   │ Problem: G generates limited       │");
    println!("   │          variety                   │");
    println!("   │                                    │");
    println!("   │ Example: Only generates digit '3'  │");
    println!("   │          even when trained on all  │");
    println!("   │          digits 0-9                │");
    println!("   └────────────────────────────────────┘\n");

    println!("   B) Vanishing Gradients");
    println!("   ┌────────────────────────────────────┐");
    println!("   │ Problem: D becomes too good        │");
    println!("   │                                    │");
    println!("   │ Effect: G receives no learning     │");
    println!("   │         signal (gradient = 0)      │");
    println!("   │                                    │");
    println!("   │ Solution: Use non-saturating loss  │");
    println!("   └────────────────────────────────────┘\n");

    println!("   C) Training Instability");
    println!("   ┌────────────────────────────────────┐");
    println!("   │ Problem: Losses oscillate wildly   │");
    println!("   │                                    │");
    println!("   │ Solutions:                         │");
    println!("   │ • Careful learning rate tuning     │");
    println!("   │ • Spectral normalization           │");
    println!("   │ • Train D multiple times per G     │");
    println!("   └────────────────────────────────────┘\n");

    // GAN variants
    println!("6. Popular GAN Variants\n");

    println!("   ┌──────────────┬─────────────────┬───────────────────┐");
    println!("   │ Variant      │ Key Feature     │ Use Case          │");
    println!("   ├──────────────┼─────────────────┼───────────────────┤");
    println!("   │ DCGAN        │ Convolutional   │ Image generation  │");
    println!("   │ CGAN         │ Conditional     │ Class-specific    │");
    println!("   │ CycleGAN     │ Unpaired        │ Style transfer    │");
    println!("   │ StyleGAN     │ Style control   │ Faces (SOTA)      │");
    println!("   │ WGAN         │ Wasserstein     │ Stable training   │");
    println!("   │ Progressive  │ Multi-scale     │ High-res images   │");
    println!("   │ Pix2Pix      │ Paired I2I      │ Sketch→Photo      │");
    println!("   └──────────────┴─────────────────┴───────────────────┘\n");

    // Applications
    println!("7. Real-World Applications\n");

    println!("   Creative:");
    println!("   • ThisPersonDoesNotExist.com (StyleGAN)");
    println!("   • DALL-E, Midjourney (text→image)");
    println!("   • Artbreeder (genetic image breeding)\n");

    println!("   Practical:");
    println!("   • Image super-resolution (enhance quality)");
    println!("   • Data augmentation (more training data)");
    println!("   • Medical imaging (synthetic scans)\n");

    println!("   Scientific:");
    println!("   • Drug discovery (generate molecules)");
    println!("   • Material design (new materials)");
    println!("   • Protein folding assistance\n");

    println!("   Controversial:");
    println!("   • DeepFakes (face swapping)");
    println!("   • Voice cloning");
    println!("   • Misinformation risks\n");

    // Training tips
    println!("8. Training Tips for Success\n");

    println!("   ✓ Normalize inputs to [-1, 1]");
    println!("   ✓ Use Tanh activation in Generator output");
    println!("   ✓ Use LeakyReLU (α=0.2) in Discriminator");
    println!("   ✓ Use BatchNormalization in both networks");
    println!("   ✓ Label smoothing: 0.9 instead of 1.0");
    println!("   ✓ Train D more: 1-5 D steps per G step");
    println!("   ✓ Use Adam optimizer (β1=0.5, β2=0.999)");
    println!("   ✓ Monitor both losses and generated samples");
    println!("   ✓ Start with low learning rate (0.0002)");
    println!("   ✓ Use noise injection for better diversity\n");

    // Modern context
    println!("9. Modern Generative AI Landscape\n");

    println!("   GANs (2014-2020):");
    println!("   ✓ Pioneered realistic image generation");
    println!("   ✓ Fast sampling (single forward pass)");
    println!("   ✗ Difficult to train");
    println!("   ✗ Mode collapse issues\n");

    println!("   Diffusion Models (2020+):");
    println!("   ✓ Stable Diffusion, DALL-E 2/3");
    println!("   ✓ More stable training");
    println!("   ✓ Better sample quality");
    println!("   ✗ Slower sampling (many steps)\n");

    println!("   Current Trend:");
    println!("   • Diffusion models dominate image generation");
    println!("   • GANs still used for real-time applications");
    println!("   • Hybrid approaches combining both\n");

    println!("=== Example Complete! ===");
    println!("\nKey Takeaways:");
    println!("- GANs train two networks in competition");
    println!("- Generator creates fakes, Discriminator detects them");
    println!("- Training is challenging (mode collapse, instability)");
    println!("- Many variants for different tasks");
    println!("- Revolutionized generative AI");
    println!("- Modern: Diffusion models often preferred");
    println!("- Still important for real-time generation");
}

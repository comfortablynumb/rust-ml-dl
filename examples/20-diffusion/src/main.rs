//! # Diffusion Models: Denoising Diffusion Probabilistic Models
//!
//! This example explains Diffusion Models, the state-of-the-art generative models
//! that power Stable Diffusion, DALL-E 2, Midjourney, and Imagen.
//!
//! ## The Core Idea
//!
//! **Generate by learning to denoise:**
//!
//! ```
//! Traditional generative models:
//! Noise z → Generator → Image x
//!
//! Diffusion models:
//! Image x → Gradually add noise → Pure noise z
//!          ← Learn to reverse (denoise) ←
//!
//! Generation: Pure noise → Denoise step by step → Clean image
//! ```
//!
//! ### Intuitive Example
//!
//! ```
//! Start with photo of a cat
//! Step 1: Add tiny noise → slightly blurry cat
//! Step 2: Add more noise → blurrier cat
//! ...
//! Step 1000: Pure noise (no cat visible)
//!
//! Training: Learn to reverse each step
//! • Given "blurrier cat", predict "less blurry cat"
//! • Train neural network to denoise
//!
//! Generation:
//! Start: Pure random noise
//! Step 999: Denoise → Very blurry image
//! Step 998: Denoise → Less blurry
//! ...
//! Step 0: Denoise → Sharp, realistic cat!
//! ```
//!
//! ## Two Processes
//!
//! ### Forward Process (Diffusion)
//!
//! **Add noise gradually:**
//!
//! ```
//! q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
//!
//! In words:
//! • Start with clean image x_0
//! • At each step t, add Gaussian noise
//! • β_t: noise schedule (how much noise to add)
//! • After T steps (e.g., T=1000): x_T ≈ pure noise
//!
//! Example:
//! x_0: [original cat image]
//! x_1: [cat + tiny noise]
//! x_2: [cat + more noise]
//! ...
//! x_1000: [pure Gaussian noise]
//! ```
//!
//! ### Nice Property: Closed Form
//!
//! ```
//! Can jump directly to any timestep t:
//!
//! q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t) I)
//!
//! Where:
//! α_t = 1 - β_t
//! ᾱ_t = ∏_{s=1}^t α_s
//!
//! In code:
//! x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
//! where ε ~ N(0,1)
//!
//! No need to apply noise 1000 times!
//! Can directly sample x_t from x_0
//! ```
//!
//! ### Reverse Process (Denoising)
//!
//! **Learn to remove noise:**
//!
//! ```
//! p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
//!
//! Train neural network to predict:
//! • Given noisy image x_t and timestep t
//! • Predict slightly less noisy x_{t-1}
//!
//! Network μ_θ(x_t, t):
//! • Input: Noisy image x_t + timestep t
//! • Output: Denoised image x_{t-1}
//!
//! Often parameterized as predicting noise ε_θ:
//! • Instead of predicting x_{t-1}
//! • Predict the noise ε that was added
//! • More stable training!
//! ```
//!
//! ## Training Diffusion Models
//!
//! ### Training Objective
//!
//! ```
//! Simplified loss (DDPM):
//!
//! L_simple = 𝔼_t,x_0,ε [||ε - ε_θ(x_t, t)||²]
//!
//! Algorithm:
//! 1. Sample image x_0 from dataset
//! 2. Sample timestep t ~ Uniform(1, T)
//! 3. Sample noise ε ~ N(0, I)
//! 4. Compute noisy image: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
//! 5. Predict noise: ε_pred = ε_θ(x_t, t)
//! 6. Compute loss: ||ε - ε_pred||²
//! 7. Backpropagate
//!
//! Interpretation:
//! Train network to predict what noise was added!
//! ```
//!
//! ### Noise Schedule
//!
//! ```
//! β_t: How much noise to add at each step
//!
//! Common schedules:
//!
//! 1. Linear:
//! β_1 = 0.0001
//! β_T = 0.02
//! β_t = linear interpolation
//!
//! 2. Cosine (better):
//! More noise early, less at end
//! Smoother transition
//!
//! Typical: T = 1000 steps
//! ```
//!
//! ## Generation (Sampling)
//!
//! ### Sampling Algorithm (DDPM)
//!
//! ```
//! # Start with pure noise
//! x_T ~ N(0, I)
//!
//! # Denoise step by step
//! for t in [T, T-1, ..., 1]:
//!     # Predict noise
//!     ε_pred = ε_θ(x_t, t)
//!
//!     # Compute less noisy image
//!     x_{t-1} = (1/√α_t) · (x_t - ((1-α_t)/√(1-ᾱ_t)) · ε_pred)
//!
//!     # Add noise (except last step)
//!     if t > 1:
//!         z ~ N(0, I)
//!         x_{t-1} += σ_t · z
//!
//! return x_0  # Final denoised image
//!
//! Time: T forward passes (slow but high quality!)
//! ```
//!
//! ### Faster Sampling (DDIM)
//!
//! ```
//! Denoising Diffusion Implicit Models (DDIM):
//! • Skip steps! Sample at t = [1000, 900, 800, ...]
//! • 50 steps instead of 1000
//! • 20× faster!
//! • Slight quality degradation
//!
//! Deterministic option:
//! • No added noise during sampling
//! • Same noise → same image (reproducible)
//! ```
//!
//! ## Network Architecture
//!
//! ### U-Net with Time Embedding
//!
//! ```
//! Input: Noisy image x_t (e.g., 256×256×3)
//!        Timestep t (embedded as sinusoidal encoding)
//!
//! Architecture:
//!
//!     x_t (256×256)
//!        ↓
//!   Encoder (downsampling)
//!     128×128 → 64×64 → 32×32
//!        ↓
//!   Bottleneck (32×32) + Time embedding
//!        ↓
//!   Decoder (upsampling)
//!     64×64 → 128×128 → 256×256
//!        ↓
//!   Output: Predicted noise ε (256×256)
//!
//! Time embedding:
//! • Convert t to high-dim vector (like positional encoding)
//! • Add to features at each layer
//! • Network learns different behavior per timestep
//! ```
//!
//! ### Why U-Net?
//!
//! ```
//! • Skip connections preserve details
//! • Multi-scale features
//! • Proven in image-to-image tasks
//! • Same architecture as image segmentation
//! ```
//!
//! ## Conditional Generation
//!
//! ### Class-Conditional
//!
//! ```
//! Generate specific class (e.g., "dog"):
//!
//! ε_θ(x_t, t, c)  ← Add class label c
//!
//! Implementation:
//! • Embed class label
//! • Add to time embedding
//! • Network conditions on class
//!
//! Generation:
//! Choose class → Generate that class
//! ```
//!
//! ### Text-Conditional (Stable Diffusion, DALL-E)
//!
//! ```
//! Generate from text prompt: "A cat wearing a hat"
//!
//! Architecture:
//! 1. Text encoder (CLIP, T5):
//!    "A cat..." → text embedding
//!
//! 2. Cross-attention in U-Net:
//!    Image features attend to text features
//!
//! 3. Classifier-free guidance:
//!    Balance conditional and unconditional
//!
//! ε_total = ε_uncond + w · (ε_cond - ε_uncond)
//! where w = guidance scale (1.0-15.0)
//!
//! Higher w: Stronger text adherence, less diversity
//! Lower w: More diversity, weaker text match
//! ```
//!
//! ## Classifier-Free Guidance
//!
//! **Key technique for controllable generation:**
//!
//! ```
//! Problem: How to make generation follow text strongly?
//!
//! Solution: Train one model for both:
//! • Conditional: p(x|text)
//! • Unconditional: p(x)  (random dropout text 10% of time)
//!
//! At generation:
//! noise_pred = noise_uncond + guidance_scale × (noise_cond - noise_uncond)
//!
//! Guidance scale:
//! • 1.0: Unconditional (ignore text)
//! • 7.5: Default for Stable Diffusion
//! • 15.0: Very strong text adherence (may be less realistic)
//! ```
//!
//! ## Latent Diffusion (Stable Diffusion)
//!
//! **Diffusion in compressed space:**
//!
//! ```
//! Problem: Diffusion in pixel space is slow
//! • 512×512 image = 786,432 dimensions
//! • 1000 steps × huge U-Net = very slow
//!
//! Solution: Work in latent space
//!
//! 1. VAE encoder: Image (512×512) → Latent (64×64)
//!    ↓ (8× compression)
//!
//! 2. Diffusion in latent space (64×64)
//!    ↓ (8× smaller = 64× faster)
//!
//! 3. VAE decoder: Latent (64×64) → Image (512×512)
//!
//! Result: 8× faster, similar quality!
//! ```
//!
//! ### Stable Diffusion Architecture
//!
//! ```
//! Components:
//!
//! 1. CLIP Text Encoder:
//!    "A cat..." → text_embedding (77×768)
//!
//! 2. VAE Encoder:
//!    Image (512×512×3) → latent (64×64×4)
//!
//! 3. U-Net (diffusion model):
//!    latent + text_embedding → denoised_latent
//!    • Cross-attention layers
//!    • ResNet blocks
//!    • Self-attention
//!
//! 4. VAE Decoder:
//!    latent (64×64×4) → Image (512×512×3)
//!
//! Generation:
//! Text → CLIP → Diffusion (50 steps) → VAE → Image
//! Time: ~3 seconds on GPU
//! ```
//!
//! ## Applications
//!
//! ### Text-to-Image
//!
//! ```
//! Stable Diffusion, DALL-E 2, Midjourney, Imagen
//!
//! "A photo of an astronaut riding a horse"
//!   ↓
//! Photorealistic image
//!
//! Capabilities:
//! • Composition: "cat on table"
//! • Styles: "oil painting", "3D render"
//! • Attributes: "blue eyes", "wearing hat"
//! • Concepts: "in the style of Van Gogh"
//! ```
//!
//! ### Image Editing
//!
//! **Inpainting:**
//! ```
//! Mask part of image + text prompt
//! → Fill in masked region coherently
//!
//! Example:
//! Image: photo of room
//! Mask: empty wall
//! Prompt: "oil painting of mountains"
//! Result: Painting appears on wall
//! ```
//!
//! **Image-to-Image:**
//! ```
//! Start from existing image (not pure noise)
//! Add noise to step t
//! Denoise with text guidance
//! → Modified image
//!
//! Example:
//! Input: Sketch of a cat
//! Prompt: "Realistic cat photo"
//! Result: Photorealistic version of sketch
//! ```
//!
//! ### Super-Resolution
//!
//! ```
//! Low-res image → Diffusion model → High-res image
//!
//! Used in:
//! • Photography enhancement
//! • Old photo restoration
//! • Medical imaging
//! ```
//!
//! ### Video Generation
//!
//! ```
//! Extend to temporal dimension:
//! • 3D U-Net (spatial + temporal)
//! • Generate frame by frame
//! • Ensure temporal consistency
//!
//! Examples: Runway Gen-2, Pika, Sora (OpenAI)
//! ```
//!
//! ### Other Modalities
//!
//! ```
//! Audio: Generate music, speech
//! 3D: Generate 3D models
//! Molecules: Drug discovery
//! Protein: Predict protein structures
//! ```
//!
//! ## Why Diffusion Models Won
//!
//! ### vs GANs
//!
//! ```
//! Diffusion:
//! ✅ Stable training (no mode collapse)
//! ✅ Higher quality
//! ✅ Better diversity
//! ✅ Easier to scale
//! ❌ Slow generation (many steps)
//!
//! GAN:
//! ✅ Fast generation (one pass)
//! ❌ Training instability
//! ❌ Mode collapse
//! ❌ Lower diversity
//!
//! 2022+: Diffusion is the winner for images
//! ```
//!
//! ### vs VAE
//!
//! ```
//! Diffusion:
//! ✅ Sharp, high-quality outputs
//! ✅ Better for complex data
//! ❌ Slow generation
//! ❌ Hard to get latent representation
//!
//! VAE:
//! ✅ Fast generation
//! ✅ Explicit latent space
//! ❌ Blurry outputs
//!
//! Hybrid: Stable Diffusion uses VAE + Diffusion!
//! ```
//!
//! ### vs Autoregressive (GPT-style)
//!
//! ```
//! Diffusion:
//! ✅ Parallel denoising
//! ✅ Better for images
//! ✅ Continuous data
//!
//! Autoregressive:
//! ✅ Better for discrete data (text)
//! ✅ Exact likelihood
//! ❌ Sequential generation (slow for images)
//! ```
//!
//! ## Training Diffusion Models
//!
//! ### Data Requirements
//!
//! ```
//! High quality images: 10M - 1B+
//!
//! Examples:
//! • LAION-5B: 5 billion image-text pairs
//! • Used for Stable Diffusion
//! • Filtered for quality, safety
//!
//! Can train on smaller datasets:
//! • Few thousand images for specific domain
//! • Fine-tune from pretrained model
//! ```
//!
//! ### Computational Requirements
//!
//! ```
//! Training from scratch:
//! • 256×256 images: 100-500 GPU-days
//! • 512×512 images: 1000+ GPU-days
//! • Use A100 GPUs (80GB)
//!
//! Fine-tuning:
//! • 10-100 GPU-hours
//! • Consumer GPUs possible (RTX 3090)
//!
//! Inference:
//! • 512×512 image: 2-5 seconds (GPU)
//! • 50 diffusion steps
//! • Can optimize (distillation → 1 step!)
//! ```
//!
//! ### Hyperparameters
//!
//! ```
//! Diffusion steps T: 1000 (training)
//! Noise schedule: Cosine
//! Optimizer: AdamW
//! Learning rate: 1e-4
//! Batch size: 64-256
//! Image size: 256, 512, or 1024
//!
//! Sampling steps: 20-50 (inference)
//! Guidance scale: 7.5 (text-to-image)
//! ```
//!
//! ## Advanced Techniques
//!
//! ### Cascaded Diffusion
//!
//! ```
//! Generate at increasing resolutions:
//!
//! Model 1: 64×64
//! Model 2: 64×64 → 256×256 (super-resolution)
//! Model 3: 256×256 → 1024×1024 (super-resolution)
//!
//! Used in: DALL-E 2, Imagen
//! Benefit: Higher resolution, better quality
//! ```
//!
//! ### Distillation
//!
//! ```
//! Problem: 50 steps is slow
//!
//! Solution: Train smaller model to mimic in 1-4 steps
//! • Student model learns to predict 50-step output
//! • 10-50× faster
//! • Slight quality loss
//!
//! Examples: Progressive Distillation, Consistency Models
//! ```
//!
//! ### ControlNet
//!
//! ```
//! Add spatial control to Stable Diffusion:
//!
//! Inputs:
//! • Text: "A photo of a cat"
//! • Control: Edge map, depth map, pose
//!
//! Output: Image matching both text AND control
//!
//! Uses:
//! • Precise composition
//! • Preserve structure
//! • Artistic control
//! ```
//!
//! ## Practical Tips
//!
//! ### Prompt Engineering
//!
//! ```
//! Bad: "cat"
//! Good: "A professional photograph of a fluffy cat, high detail, 8k"
//!
//! Tips:
//! • Be specific
//! • Mention style ("oil painting", "3D render")
//! • Add quality terms ("highly detailed", "8k")
//! • Use negative prompts (what to avoid)
//! ```
//!
//! ### Sampling Settings
//!
//! ```
//! Steps: 20-50
//! • 20: Fast, lower quality
//! • 50: Slower, better quality
//!
//! Guidance scale: 1-15
//! • 1: Diverse, may ignore text
//! • 7.5: Balanced (default)
//! • 15: Strong text match, less diverse
//!
//! Sampler: DDIM, DPM++, Euler
//! • Different noise schedules
//! • Subtle differences
//! • DPM++ often good
//! ```
//!
//! ## Historical Impact
//!
//! **2015:** Early diffusion work (Sohl-Dickstein)
//! - Theoretical foundation
//! - Impractical to train
//!
//! **2020:** DDPM (Denoising Diffusion Probabilistic Models)
//! - Ho et al., made diffusion practical
//! - Beat GANs on some benchmarks
//!
//! **2021:** Improved Diffusion (OpenAI)
//! - Classifier guidance
//! - Higher quality than GANs
//!
//! **2021:** GLIDE (OpenAI)
//! - Text-to-image with diffusion
//! - Photorealistic results
//!
//! **2022:** DALL-E 2 (OpenAI)
//! - Cascaded diffusion
//! - Amazing text-to-image
//! - Not open source
//!
//! **2022:** Stable Diffusion (Stability AI)
//! - Open source!
//! - Latent diffusion
//! - Consumer GPUs
//! - Democratized AI art
//!
//! **2022:** Imagen (Google)
//! - Text encoder: T5
//! - Cascaded models
//! - State-of-the-art quality
//!
//! **2023:** Midjourney v5
//! - Artistic generations
//! - Commercial success
//!
//! **2024:** Sora (OpenAI)
//! - Text-to-video
//! - 1-minute videos
//! - Photorealistic
//!
//! **Legacy:**
//! - Replaced GANs as #1 generative model
//! - Enabled AI art revolution
//! - Billion-dollar industry

fn main() {
    println!("=== Diffusion Models: State-of-the-Art Generation ===\n");

    println!("This example explains Diffusion Models, the technology behind");
    println!("Stable Diffusion, DALL-E 2, Midjourney, and Imagen.\n");

    println!("📚 Key Concepts Covered:");
    println!("  • Forward diffusion (gradual noising)");
    println!("  • Reverse process (denoising)");
    println!("  • Training objective (noise prediction)");
    println!("  • Classifier-free guidance");
    println!("  • Latent diffusion (Stable Diffusion)");
    println!("  • Text-to-image generation\n");

    println!("🎯 Why This Matters:");
    println!("  • Powers modern AI art (Stable Diffusion, DALL-E, Midjourney)");
    println!("  • Replaced GANs as best generative model");
    println!("  • Enabled text-to-image revolution");
    println!("  • State-of-the-art quality and diversity");
    println!("  • Foundation of modern creative AI\n");

    println!("See the source code documentation for comprehensive explanations!");
}

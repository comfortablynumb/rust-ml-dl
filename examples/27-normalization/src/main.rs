//! # Normalization Techniques: The Secret to Training Deep Networks
//!
//! Normalization layers are critical for training deep neural networks effectively.
//! This example explains BatchNorm, LayerNorm, GroupNorm, and when to use each.
//!
//! ## The Problem: Internal Covariate Shift
//!
//! **Without normalization:**
//! ```
//! Layer 1: outputs range [0, 1]
//! Layer 2: outputs range [-100, 100]  ← Unstable!
//! Layer 3: outputs range [0.001, 0.01]
//! ...
//!
//! Problems:
//! • Gradient vanishing/exploding
//! • Slow convergence
//! • Sensitive to initialization
//! • Requires very small learning rates
//! ```
//!
//! **With normalization:**
//! ```
//! Every layer: outputs normalized to mean=0, std=1
//! • Stable gradients
//! • Faster convergence (10-100× speedup)
//! • Higher learning rates possible
//! • Less sensitive to initialization
//! ```
//!
//! ## Batch Normalization (BatchNorm, 2015)
//!
//! **The breakthrough that enabled very deep networks**
//!
//! ### How It Works
//!
//! ```
//! For each mini-batch during training:
//!
//! 1. Compute mean and variance across batch:
//!    μ_B = (1/m) Σ x_i
//!    σ²_B = (1/m) Σ (x_i - μ_B)²
//!
//! 2. Normalize:
//!    x̂_i = (x_i - μ_B) / √(σ²_B + ε)
//!
//! 3. Scale and shift (learnable):
//!    y_i = γ x̂_i + β
//!
//! Where:
//! • μ_B, σ²_B: Batch statistics
//! • ε: Small constant (1e-5) for numerical stability
//! • γ, β: Learnable parameters (scale and shift)
//! ```
//!
//! ### Why Scale and Shift (γ, β)?
//!
//! ```
//! Without γ, β: Forced to mean=0, std=1
//! Problem: Limits model expressiveness!
//!
//! Example: Sigmoid activation
//! • Linear region near 0 (normalized input)
//! • Can't use saturated regions
//!
//! With γ, β:
//! • Model can learn to undo normalization if needed
//! • Best of both worlds: stable training + expressiveness
//! ```
//!
//! ### BatchNorm for Different Tensor Shapes
//!
//! **Fully Connected Layers:**
//! ```
//! Input: (N, C)  ← Batch size N, features C
//! Normalize: Across N (batch dimension)
//! Parameters: 2C (γ and β for each feature)
//!
//! Example:
//! Input: (32, 128)  ← 32 samples, 128 features
//! → Compute μ, σ for each of 128 features across 32 samples
//! ```
//!
//! **Convolutional Layers:**
//! ```
//! Input: (N, C, H, W)  ← Batch, Channels, Height, Width
//! Normalize: Across N, H, W (keep C separate)
//! Parameters: 2C (γ and β for each channel)
//!
//! Example:
//! Input: (32, 64, 28, 28)  ← 32 images, 64 channels, 28×28
//! → For each channel: compute μ, σ across 32×28×28 values
//! → 64 channels = 128 parameters (64 γ + 64 β)
//! ```
//!
//! ### Training vs Inference
//!
//! **Training:**
//! ```
//! Use batch statistics (μ_B, σ²_B)
//! Update running average:
//! μ_running = momentum × μ_running + (1-momentum) × μ_B
//! σ²_running = momentum × σ²_running + (1-momentum) × σ²_B
//!
//! Typical momentum: 0.9 or 0.99
//! ```
//!
//! **Inference:**
//! ```
//! Use running statistics (μ_running, σ²_running)
//! Why? No batch at inference!
//! • Single image → can't compute batch statistics
//! • Need deterministic output
//!
//! y = γ × (x - μ_running)/√(σ²_running + ε) + β
//! ```
//!
//! ### Benefits
//!
//! ```
//! ✅ Faster training (10× fewer iterations)
//! ✅ Higher learning rates (10-100× larger)
//! ✅ Less sensitive to initialization
//! ✅ Regularization effect (noise from batch statistics)
//! ✅ Enables very deep networks (ResNet-152, etc.)
//! ```
//!
//! ### Limitations
//!
//! ```
//! ❌ Batch size dependency
//!    • Small batches → noisy statistics
//!    • Batch size < 8: poor performance
//!    • Training/inference discrepancy
//!
//! ❌ Not suitable for:
//!    • RNNs (variable sequence lengths)
//!    • Online learning (single sample)
//!    • Distributed training (sync required)
//! ```
//!
//! ## Layer Normalization (LayerNorm, 2016)
//!
//! **Solution for sequence models and Transformers**
//!
//! ### How It Works
//!
//! ```
//! Normalize across features (not batch!):
//!
//! For each sample independently:
//! μ = (1/C) Σ x_i
//! σ² = (1/C) Σ (x_i - μ)²
//! x̂ = (x - μ) / √(σ² + ε)
//! y = γ x̂ + β
//!
//! Key difference: Normalize within each sample
//! ```
//!
//! ### LayerNorm for Different Shapes
//!
//! **Fully Connected:**
//! ```
//! Input: (N, C)
//! Normalize: Across C for each sample
//! Parameters: 2C
//!
//! Example:
//! Input: (32, 128)
//! → Each of 32 samples normalized independently
//! → Compute μ, σ from 128 features
//! ```
//!
//! **Transformers:**
//! ```
//! Input: (N, L, D)  ← Batch, Length, Dimension
//! Normalize: Across D for each (N, L) position
//! Parameters: 2D
//!
//! Example:
//! Input: (32, 512, 768)  ← 32 seqs, 512 tokens, 768 dims
//! → Normalize 768 dims for each of 32×512 tokens
//! ```
//!
//! ### LayerNorm vs BatchNorm
//!
//! ```
//! BatchNorm:
//! • Normalize across batch dimension
//! • Requires batch size > 1
//! • Different behavior train/inference
//! • Good for CNNs
//!
//! LayerNorm:
//! • Normalize across feature dimension
//! • Works with batch size = 1
//! • Same behavior train/inference
//! • Good for RNNs, Transformers
//! ```
//!
//! ### Why LayerNorm for Transformers?
//!
//! ```
//! Transformers have:
//! • Variable sequence lengths
//! • Batch size often 1 at inference
//! • Need stable behavior regardless of batch
//!
//! LayerNorm advantages:
//! • No batch dependency
//! • Deterministic (no running stats)
//! • Works with any sequence length
//! • Used in: BERT, GPT, T5, all Transformers
//! ```
//!
//! ## Group Normalization (GroupNorm, 2018)
//!
//! **Best of both worlds for computer vision**
//!
//! ### How It Works
//!
//! ```
//! Split channels into groups, normalize within each group:
//!
//! 1. Divide C channels into G groups
//! 2. Normalize within each group
//!
//! Input: (N, C, H, W)
//! Groups: G (typically 32)
//! Channels per group: C/G
//!
//! For each group:
//!   Normalize across (C/G, H, W) for each sample
//!
//! Example:
//! Input: (32, 64, 28, 28)
//! Groups: 32
//! → 64/32 = 2 channels per group
//! → Normalize (2, 28, 28) = 1568 values per group
//! ```
//!
//! ### Special Cases
//!
//! ```
//! G = 1:  Group Normalization = Layer Normalization
//! G = C:  Group Normalization = Instance Normalization
//! G = 32: Typical choice (good performance)
//! ```
//!
//! ### Benefits
//!
//! ```
//! ✅ No batch dependency (like LayerNorm)
//! ✅ Works with batch size = 1
//! ✅ Better than LayerNorm for CNNs
//! ✅ More stable than BatchNorm with small batches
//! ✅ Good for:
//!    • Object detection (batch size 1-2)
//!    • Segmentation
//!    • Video (memory constrained)
//! ```
//!
//! ## Instance Normalization (InstanceNorm)
//!
//! **For style transfer and GANs**
//!
//! ### How It Works
//!
//! ```
//! Normalize each channel of each sample independently:
//!
//! Input: (N, C, H, W)
//! For each (n, c):
//!   Normalize across (H, W)
//!
//! Example:
//! Input: (32, 64, 28, 28)
//! → 32×64 = 2048 independent normalizations
//! → Each over 28×28 = 784 values
//! ```
//!
//! ### Use Cases
//!
//! ```
//! Style Transfer:
//! • Remove style information (color, texture)
//! • Keep content (structure)
//!
//! GANs:
//! • Stabilize training
//! • Used in StyleGAN, Pix2Pix
//! ```
//!
//! ## When to Use Which?
//!
//! ### Quick Decision Tree
//!
//! ```
//! Are you using CNNs?
//!   ├─ Yes → Large batch size (>8)?
//!   │         ├─ Yes → BatchNorm
//!   │         └─ No → GroupNorm
//!   │
//!   └─ No → Using Transformers/RNNs?
//!             └─ Yes → LayerNorm
//!
//! Style transfer or GANs?
//!   └─ InstanceNorm
//! ```
//!
//! ### Detailed Comparison Table
//!
//! | Technique | Normalize Over | Batch Dependent | Best For |
//! |-----------|----------------|-----------------|----------|
//! | **BatchNorm** | N (batch), H, W | ✅ Yes | CNNs, large batches |
//! | **LayerNorm** | C, H, W | ❌ No | Transformers, RNNs |
//! | **GroupNorm** | C/G, H, W | ❌ No | CNNs, small batches |
//! | **InstanceNorm** | H, W | ❌ No | Style transfer, GANs |
//!
//! ### Modern Recommendations (2024)
//!
//! ```
//! Computer Vision:
//! • ResNet, VGG, etc: BatchNorm
//! • Object detection (YOLO, DETR): GroupNorm
//! • Small batch training: GroupNorm
//! • Style transfer: InstanceNorm
//!
//! NLP/Transformers:
//! • BERT, GPT, T5: LayerNorm (Pre-LN or Post-LN)
//! • All modern Transformers: LayerNorm
//!
//! Hybrid:
//! • Vision Transformers (ViT): LayerNorm
//! • Convnext: LayerNorm (CNNs using LN!)
//! ```
//!
//! ## Implementation Tips
//!
//! ### Placement in Network
//!
//! ```
//! Pre-Normalization (Modern Transformers):
//! x = x + MLP(LayerNorm(x))
//!     ↑
//! Normalize BEFORE sub-layer
//!
//! Benefits:
//! • More stable training
//! • Can train deeper (100+ layers)
//! • Used in: GPT-3, BERT (modern variants)
//!
//! Post-Normalization (Original):
//! x = LayerNorm(x + MLP(x))
//!                ↑
//! Normalize AFTER residual
//!
//! Original Transformer used this, but Pre-LN is now standard
//! ```
//!
//! ### Training Considerations
//!
//! ```
//! Learning Rate:
//! • With normalization: 10-100× higher LR possible
//! • BatchNorm: Try 0.1 instead of 0.001
//! • LayerNorm: Less sensitive, start with 0.001
//!
//! Weight Initialization:
//! • Less critical with normalization
//! • But still use Xavier/He initialization
//!
//! Warmup:
//! • Still beneficial for large models
//! • Gradually increase LR over first few epochs
//! ```
//!
//! ### Common Mistakes
//!
//! ```
//! ❌ Using BatchNorm with batch size 1
//! ❌ Forgetting to switch to eval mode (BatchNorm)
//! ❌ Using BatchNorm for RNNs
//! ❌ Not updating running stats (BatchNorm)
//! ❌ Applying normalization to every layer (overkill)
//!
//! ✅ Normalize after conv/linear, before activation
//! ✅ Use eval() mode for inference
//! ✅ Match normalization to architecture type
//! ✅ Monitor running stats during training
//! ```
//!
//! ## Advanced Topics
//!
//! ### Synchronized BatchNorm
//!
//! ```
//! Problem: Distributed training with small local batches
//!
//! Solution: Sync statistics across GPUs
//! • Compute μ, σ across all GPUs
//! • Requires communication overhead
//! • Used in large-scale training
//! ```
//!
//! ### Adaptive Normalization
//!
//! ```
//! AdaIN (Adaptive Instance Normalization):
//! • Used in style transfer
//! • Modulate γ, β from style input
//! • y = γ_style × normalize(x) + β_style
//!
//! SPADE (Spatially-Adaptive Normalization):
//! • Used in image generation (GauGAN)
//! • Spatially-varying normalization
//! ```
//!
//! ### Normalization-Free Networks
//!
//! ```
//! Recent research: Train without normalization
//! • NFNets (Normalizer-Free Networks)
//! • Careful initialization + activation
//! • Match BatchNorm performance
//!
//! Benefits:
//! • Simpler architecture
//! • No batch dependency
//! • Faster inference
//!
//! Still experimental for most use cases
//! ```
//!
//! ## Historical Impact
//!
//! **2015: BatchNorm**
//! - Ioffe & Szegedy
//! - Enabled training of very deep networks
//! - ResNet (152 layers) became possible
//!
//! **2016: LayerNorm**
//! - Ba et al.
//! - Solved RNN training issues
//! - Foundation for Transformers
//!
//! **2017: Transformer**
//! - Uses LayerNorm exclusively
//! - Proved effectiveness beyond RNNs
//!
//! **2018: GroupNorm**
//! - Wu & He
//! - Better than BatchNorm for small batches
//! - Widely adopted in detection/segmentation
//!
//! **2019+: Pre-LN Transformers**
//! - Pre-normalization becomes standard
//! - Enables GPT-3 scale models
//!
//! **Legacy:**
//! - Normalization layers are now standard
//! - Every modern architecture uses some form
//! - Critical enabler of deep learning success

fn main() {
    println!("=== Normalization Techniques ===\n");

    println!("Critical techniques for training deep neural networks effectively.\n");

    println!("📚 Techniques Covered:");
    println!("  • BatchNorm: Normalize across batch (CNNs, large batches)");
    println!("  • LayerNorm: Normalize across features (Transformers, RNNs)");
    println!("  • GroupNorm: Split into groups (small batch CNNs)");
    println!("  • InstanceNorm: Per-sample normalization (style transfer)\n");

    println!("🎯 Key Benefits:");
    println!("  • 10-100× faster training");
    println!("  • Higher learning rates possible");
    println!("  • Enables very deep networks (ResNet-152, GPT-3)");
    println!("  • Less sensitive to initialization");
    println!("  • Regularization effect\n");

    println!("💡 When to Use:");
    println!("  • CNNs + large batch → BatchNorm");
    println!("  • CNNs + small batch → GroupNorm");
    println!("  • Transformers/RNNs → LayerNorm");
    println!("  • Style transfer/GANs → InstanceNorm\n");

    println!("See source code documentation for comprehensive explanations!");
}

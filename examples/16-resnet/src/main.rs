//! # Residual Network (ResNet) Example
//!
//! This example demonstrates ResNet, the architecture that made very deep networks practical
//! through skip connections, winning ImageNet 2015 and revolutionizing computer vision.
//!
//! ## The Deep Network Problem
//!
//! Before ResNet (2015):
//! ```
//! Network depth vs accuracy:
//! 20 layers:  Good accuracy  ✓
//! 56 layers:  Worse accuracy ✗
//!
//! Problem: Deeper ≠ Better!
//! ```
//!
//! **Why?** Degradation problem (not overfitting!)
//! - Deeper networks harder to optimize
//! - Gradients vanish
//! - Training error actually increases!
//!
//! ## ResNet's Solution: Skip Connections
//!
//! **Key Idea:** Instead of learning H(x), learn the residual F(x) = H(x) - x
//!
//! ```
//! Traditional Block:
//! x → [Conv → ReLU → Conv] → H(x)
//!
//! Residual Block:
//! x → [Conv → ReLU → Conv] → F(x)
//! x ──────────────────────────→ + → H(x) = F(x) + x
//!  └──── Skip Connection ────────┘
//! ```
//!
//! **Intuition:**
//! - If identity mapping is optimal, just learn F(x) = 0
//! - Easier to learn small changes than full transformations
//! - Gradients flow directly through skip connections
//!
//! ## Residual Block Architecture
//!
//! ### Basic Block (ResNet-18, ResNet-34)
//! ```
//! Input x
//!    │
//!    ├─→ Conv 3×3, 64 filters
//!    │   BatchNorm
//!    │   ReLU
//!    │   Conv 3×3, 64 filters
//!    │   BatchNorm
//!    │
//!    └─→ (identity) ──→ +
//!                       │
//!                      ReLU
//!                       │
//!                    Output
//! ```
//!
//! ### Bottleneck Block (ResNet-50, ResNet-101, ResNet-152)
//! ```
//! Input x [256 channels]
//!    │
//!    ├─→ Conv 1×1, 64 filters  ← Reduce dimensions
//!    │   BatchNorm
//!    │   ReLU
//!    │   Conv 3×3, 64 filters  ← Process
//!    │   BatchNorm
//!    │   ReLU
//!    │   Conv 1×1, 256 filters ← Expand dimensions
//!    │   BatchNorm
//!    │
//!    └─→ (identity) ──→ +
//!                       │
//!                      ReLU
//!                       │
//!                    Output
//! ```
//!
//! **Bottleneck Benefits:**
//! - Reduces parameters: 3×(3×3×256) > 1×1×64 + 3×3×64 + 1×1×256
//! - Faster computation
//! - Enables deeper networks
//!
//! ## ResNet Variants
//!
//! ### ResNet-18 (11.7M parameters)
//! ```
//! Conv1:    7×7, 64
//! Conv2_x:  3×3, 64  × 2 blocks
//! Conv3_x:  3×3, 128 × 2 blocks
//! Conv4_x:  3×3, 256 × 2 blocks
//! Conv5_x:  3×3, 512 × 2 blocks
//! AvgPool, FC(1000)
//! ```
//!
//! ### ResNet-34 (21.8M parameters)
//! Same as ResNet-18 but:
//! ```
//! Conv2_x: × 3 blocks
//! Conv3_x: × 4 blocks
//! Conv4_x: × 6 blocks
//! Conv5_x: × 3 blocks
//! ```
//!
//! ### ResNet-50 (25.6M parameters)
//! Uses bottleneck blocks:
//! ```
//! Conv2_x: × 3 bottleneck blocks
//! Conv3_x: × 4 bottleneck blocks
//! Conv4_x: × 6 bottleneck blocks
//! Conv5_x: × 3 bottleneck blocks
//! ```
//!
//! ### ResNet-101 (44.5M parameters)
//! ```
//! Conv4_x: × 23 bottleneck blocks (!)
//! ```
//!
//! ### ResNet-152 (60.2M parameters)
//! ```
//! Conv3_x: × 8 blocks
//! Conv4_x: × 36 blocks (!)
//! ```
//!
//! ## Why Skip Connections Work
//!
//! ### 1. Gradient Flow
//! ```
//! Without skip connections:
//! ∂L/∂x₀ = ∂L/∂xₙ × ∂xₙ/∂xₙ₋₁ × ... × ∂x₁/∂x₀
//!
//! Problem: Product of many terms < 1 → vanishing!
//!
//! With skip connections:
//! ∂L/∂x₀ = ∂L/∂xₙ × (1 + ∂F/∂x)
//!
//! Benefit: Always have gradient of at least 1!
//! ```
//!
//! ### 2. Identity Mapping
//! ```
//! If optimal H(x) = x, just need:
//! - F(x) = 0 (easy!)
//! vs
//! - Learn full identity (hard)
//! ```
//!
//! ### 3. Ensemble Effect
//! ```
//! Network learns multiple paths:
//! Input → Output (8 possible paths in 3-block ResNet!)
//!
//! Acts like ensemble of shallow networks
//! ```
//!
//! ## Projection Shortcuts
//!
//! When dimensions change (stride > 1 or channels change):
//!
//! ### Option A: Zero Padding
//! ```
//! Cheap, no extra parameters
//! Pad with zeros to match dimensions
//! ```
//!
//! ### Option B: Projection (1×1 Conv)
//! ```
//! x → 1×1 Conv → matches output dimensions
//!
//! Adds parameters but more flexible
//! Used in all ResNet variants
//! ```
//!
//! ## Batch Normalization
//!
//! **Critical for ResNet:**
//! ```
//! x → Conv → BatchNorm → ReLU
//! ```
//!
//! **Benefits:**
//! - Normalizes activations
//! - Reduces internal covariate shift
//! - Allows higher learning rates
//! - Acts as regularization
//! - Essential for very deep networks
//!
//! **Formula:**
//! ```
//! BN(x) = γ × (x - μ)/σ + β
//! ```
//! Where γ, β are learnable parameters
//!
//! ## ImageNet Results (2015)
//!
//! ```
//! Model         Top-5 Error   Layers
//! ──────────────────────────────────
//! VGG-16        7.3%          16
//! GoogLeNet     6.7%          22
//! ResNet-34     5.7%          34   ← Deeper!
//! ResNet-50     5.25%         50
//! ResNet-101    4.6%          101  ← 100+ layers!
//! ResNet-152    3.57%         152  ← Winner!
//! ```
//!
//! **Breakthrough:** 152 layers trained successfully!
//!
//! ## Applications
//!
//! ### Computer Vision
//! - Image classification (ImageNet)
//! - Object detection (Faster R-CNN, Mask R-CNN)
//! - Semantic segmentation (FCN, U-Net variants)
//! - Face recognition
//! - Medical imaging
//!
//! ### Transfer Learning
//! ```
//! 1. Pre-train on ImageNet (millions of images)
//! 2. Fine-tune on your dataset (thousands of images)
//! 3. Often better than training from scratch!
//! ```
//!
//! ### Feature Extraction
//! ```
//! Remove final FC layer
//! Use Conv5_x output as features
//! Train classifier on top
//! ```
//!
//! ## Variants & Extensions
//!
//! ### ResNeXt
//! ```
//! Adds "cardinality" dimension
//! Multiple parallel paths instead of one
//! Better accuracy with same complexity
//! ```
//!
//! ### Wide ResNet (WRN)
//! ```
//! Fewer layers, wider (more filters)
//! Example: WRN-28-10
//! - 28 layers
//! - 10× wider than standard
//! ```
//!
//! ### DenseNet
//! ```
//! Every layer connects to every other
//! x_l = H([x_0, x_1, ..., x_{l-1}])
//! Even stronger gradient flow!
//! ```
//!
//! ### EfficientNet
//! ```
//! Compound scaling (depth + width + resolution)
//! State-of-the-art efficiency
//! ResNet-like blocks with MBConv
//! ```
//!
//! ### Vision Transformer (ViT)
//! ```
//! Eventually surpassed ResNets
//! But uses similar residual connections!
//! ```
//!
//! ## Training Tips
//!
//! ### 1. Learning Rate Schedule
//! ```
//! Start: 0.1
//! Divide by 10 every 30 epochs
//! Or: Cosine annealing
//! ```
//!
//! ### 2. Data Augmentation
//! ```
//! - Random crop
//! - Random horizontal flip
//! - Color jitter
//! - Mixup / CutMix
//! ```
//!
//! ### 3. Regularization
//! ```
//! - Weight decay: 1e-4
//! - Dropout: Usually NOT needed (BatchNorm is enough)
//! - Label smoothing: 0.1
//! ```
//!
//! ### 4. Optimization
//! ```
//! - SGD with momentum (0.9)
//! - Batch size: 256
//! - Epochs: 90-120
//! - Warmup: First 5 epochs
//! ```
//!
//! ## Key Innovations
//!
//! 1. **Skip Connections**
//!    - Identity shortcuts
//!    - Enable gradient flow
//!    - Allow 100+ layers
//!
//! 2. **Batch Normalization**
//!    - After every conv layer
//!    - Stabilizes training
//!    - Reduces dependence on initialization
//!
//! 3. **Bottleneck Design**
//!    - 1×1 conv for dimension reduction
//!    - Efficient computation
//!    - More layers, fewer parameters
//!
//! 4. **No Dropout Needed**
//!    - BatchNorm provides regularization
//!    - Simpler architecture
//!
//! ## Impact on Deep Learning
//!
//! **Before ResNet (2015):**
//! - Deepest networks: ~30 layers
//! - Deeper networks performed worse
//! - Limited by vanishing gradients
//!
//! **After ResNet:**
//! - 100+ layer networks practical
//! - Skip connections everywhere
//! - Enabled modern architectures
//! - Influenced transformers, GANs, etc.
//!
//! **Citation Impact:**
//! - 100,000+ citations
//! - One of most influential papers in AI
//! - CVPR 2016 Best Paper Award

fn main() {
    println!("=== Residual Networks (ResNet) ===\n");

    println!("This example explains ResNet and skip connections.\n");

    // The problem
    println!("1. The Deep Network Problem (Before ResNet)\n");

    println!("   Experiment: Train networks of different depths");
    println!("   ┌────────────┬───────────────┬────────────────┐");
    println!("   │ Layers     │ Train Error   │ Test Error     │");
    println!("   ├────────────┼───────────────┼────────────────┤");
    println!("   │ 20         │ 8.8%          │ 7.1%           │");
    println!("   │ 56         │ 10.2%         │ 8.5%   ✗ Worse!│");
    println!("   └────────────┴───────────────┴────────────────┘\n");

    println!("   Problem: Deeper network has HIGHER training error!");
    println!("   - Not overfitting (test error also worse)");
    println!("   - Degradation problem: optimization difficulty\n");

    // The solution
    println!("2. ResNet's Solution: Skip Connections\n");

    println!("   Traditional Block:");
    println!("   ┌─────────────────────────────────┐");
    println!("   │ x → Conv → ReLU → Conv → H(x)   │");
    println!("   └─────────────────────────────────┘\n");

    println!("   Residual Block:");
    println!("   ┌─────────────────────────────────┐");
    println!("   │ x → Conv → ReLU → Conv → F(x)   │");
    println!("   │ x ─────────────────────→ + → H(x)│");
    println!("   │         Skip                     │");
    println!("   └─────────────────────────────────┘");
    println!("   H(x) = F(x) + x\n");

    println!("   Key Insight:");
    println!("   - Learn residual F(x) = H(x) - x");
    println!("   - If identity is optimal, just set F(x) = 0");
    println!("   - Easier than learning full H(x) = x!\n");

    // Architecture
    println!("3. Residual Block Types\n");

    println!("   A) Basic Block (ResNet-18/34):");
    println!("   ┌──────────────────────┐");
    println!("   │ 3×3 Conv, 64         │");
    println!("   │ BatchNorm            │");
    println!("   │ ReLU                 │");
    println!("   │ 3×3 Conv, 64         │");
    println!("   │ BatchNorm            │");
    println!("   ├──────────────────────┤");
    println!("   │ + (skip connection)  │");
    println!("   ├──────────────────────┤");
    println!("   │ ReLU                 │");
    println!("   └──────────────────────┘\n");

    println!("   B) Bottleneck Block (ResNet-50/101/152):");
    println!("   ┌──────────────────────┐");
    println!("   │ 1×1 Conv, 64      ← Reduce");
    println!("   │ BatchNorm            │");
    println!("   │ ReLU                 │");
    println!("   │ 3×3 Conv, 64      ← Process");
    println!("   │ BatchNorm            │");
    println!("   │ ReLU                 │");
    println!("   │ 1×1 Conv, 256     ← Expand");
    println!("   │ BatchNorm            │");
    println!("   ├──────────────────────┤");
    println!("   │ + (skip connection)  │");
    println!("   ├──────────────────────┤");
    println!("   │ ReLU                 │");
    println!("   └──────────────────────┘\n");

    // ResNet variants
    println!("4. ResNet Variants\n");

    println!("   ┌────────────┬────────┬────────────┬────────────┐");
    println!("   │ Model      │ Blocks │ Parameters │ Top-5 Error│");
    println!("   ├────────────┼────────┼────────────┼────────────┤");
    println!("   │ ResNet-18  │ Basic  │ 11.7M      │ 10.2%      │");
    println!("   │ ResNet-34  │ Basic  │ 21.8M      │ 8.6%       │");
    println!("   │ ResNet-50  │ Bottl. │ 25.6M      │ 5.25%      │");
    println!("   │ ResNet-101 │ Bottl. │ 44.5M      │ 4.6%       │");
    println!("   │ ResNet-152 │ Bottl. │ 60.2M      │ 3.57%  ✓   │");
    println!("   └────────────┴────────┴────────────┴────────────┘\n");

    println!("   ResNet-152: Winner of ImageNet 2015!\n");

    // Why it works
    println!("5. Why Skip Connections Work\n");

    println!("   A) Gradient Flow");
    println!("   ┌──────────────────────────────────────────┐");
    println!("   │ Without skips:                           │");
    println!("   │ grad = ∂L/∂x₁₀₀ × ∂x₁₀₀/∂x₉₉ × ... × ∂x₁/∂x₀│");
    println!("   │ Problem: Many terms < 1 → vanishing!     │");
    println!("   └──────────────────────────────────────────┘");
    println!("   ┌──────────────────────────────────────────┐");
    println!("   │ With skips:                              │");
    println!("   │ grad = ∂L/∂xₙ × (1 + ∂F/∂x)              │");
    println!("   │ Benefit: Always have term of 1!          │");
    println!("   └──────────────────────────────────────────┘\n");

    println!("   B) Easier Optimization");
    println!("   - If identity is optimal: F(x) = 0");
    println!("   - Easier to push weights to zero");
    println!("   - Than to learn full identity mapping\n");

    println!("   C) Ensemble Effect");
    println!("   - Multiple paths through network");
    println!("   - Like ensemble of shallow networks");
    println!("   - More robust representations\n");

    // Applications
    println!("6. Applications\n");

    println!("   Computer Vision:");
    println!("   • Classification (ImageNet, CIFAR)");
    println!("   • Object Detection (Faster R-CNN, YOLO)");
    println!("   • Segmentation (Mask R-CNN, DeepLab)");
    println!("   • Face Recognition (FaceNet variants)\n");

    println!("   Transfer Learning:");
    println!("   1. Pre-train on ImageNet (1.2M images)");
    println!("   2. Remove last layer");
    println!("   3. Fine-tune on your data (1K images)");
    println!("   4. Often better than training from scratch!\n");

    // Modern variants
    println!("7. Modern Variants & Extensions\n");

    println!("   ResNeXt:");
    println!("   • Adds 'cardinality' (parallel paths)");
    println!("   • Better accuracy, same complexity\n");

    println!("   Wide ResNet (WRN):");
    println!("   • Fewer layers, more filters");
    println!("   • Sometimes better than going deeper\n");

    println!("   DenseNet:");
    println!("   • Each layer connects to all previous");
    println!("   • Even stronger gradient flow\n");

    println!("   EfficientNet:");
    println!("   • Compound scaling (depth × width × resolution)");
    println!("   • State-of-the-art efficiency\n");

    println!("   Vision Transformer:");
    println!("   • Replaced CNNs for many tasks");
    println!("   • Still uses residual connections!\n");

    // Historical impact
    println!("8. Historical Impact\n");

    println!("   Before ResNet (2015):");
    println!("   ✗ Networks limited to ~30 layers");
    println!("   ✗ Deeper = worse performance");
    println!("   ✗ Vanishing gradients\n");

    println!("   After ResNet:");
    println!("   ✓ 100+ layer networks practical");
    println!("   ✓ Skip connections standard everywhere");
    println!("   ✓ Enabled modern deep learning");
    println!("   ✓ Influenced transformers, GANs, etc.\n");

    println!("   Recognition:");
    println!("   • CVPR 2016 Best Paper Award");
    println!("   • 100,000+ citations");
    println!("   • One of most influential AI papers ever\n");

    println!("=== Example Complete! ===");
    println!("\nKey Takeaways:");
    println!("- Skip connections solve degradation problem");
    println!("- Enable training of very deep networks (100+ layers)");
    println!("- Gradient flows directly through shortcuts");
    println!("- Identity mapping easier to learn than full transformation");
    println!("- Batch normalization crucial for stability");
    println!("- Won ImageNet 2015, revolutionized computer vision");
    println!("- Skip connections now standard in most architectures");
}

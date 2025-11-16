//! # Vision Transformer (ViT): Transformers for Computer Vision
//!
//! Applies pure Transformer architecture to images, replacing CNNs.
//! Paper: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
//!
//! ## Core Idea: Images as Sequences
//!
//! Traditional: CNN processes images with convolutions
//! ViT: Split image into patches, treat as sequence of tokens
//!
//! ```
//! Image (224×224×3)
//!   ↓
//! Split into patches (16×16 each) → 196 patches
//!   ↓
//! Linear embedding → 196 tokens
//!   ↓
//! Add position embeddings
//!   ↓
//! Transformer Encoder (same as BERT!)
//!   ↓
//! Classification head
//! ```
//!
//! ## Architecture Details
//!
//! ### 1. Patch Embedding
//! ```
//! Input: 224×224×3 image
//! Patch size: 16×16
//! Number of patches: (224/16)² = 196
//!
//! Each patch: 16×16×3 = 768 values
//! Linear projection: 768 → d_model (e.g., 768)
//!
//! Result: 196 patch embeddings
//! ```
//!
//! ### 2. Position Embeddings
//! ```
//! Learnable 1D position embeddings
//! Added to patch embeddings
//!
//! pos_embed: (196, 768) learned parameters
//! patches + pos_embed → position-aware tokens
//! ```
//!
//! ### 3. Class Token
//! ```
//! Prepend learnable [CLS] token (like BERT)
//! Final sequence: [CLS] + 196 patches = 197 tokens
//!
//! [CLS] token output → classification
//! ```
//!
//! ### 4. Transformer Encoder
//! ```
//! Standard Transformer encoder blocks:
//! • Multi-head self-attention
//! • MLP (Feed-forward)
//! • Layer normalization
//! • Residual connections
//!
//! ViT-Base: 12 layers
//! ViT-Large: 24 layers
//! ViT-Huge: 32 layers
//! ```
//!
//! ## Why ViT Works
//!
//! **Inductive biases:**
//! ```
//! CNN: Built-in locality, translation equivariance
//! ViT: Minimal inductive bias, learns from data
//!
//! Consequence:
//! • Needs more data than CNNs
//! • But scales better!
//!
//! ImageNet (1.3M): CNN > ViT
//! JFT-300M (300M): ViT > CNN
//! ```
//!
//! **Global receptive field:**
//! ```
//! CNN: Local receptive field, grows with depth
//! ViT: Global from layer 1 (self-attention sees all patches)
//!
//! Better for:
//! • Long-range dependencies
//! • Global context
//! ```
//!
//! ## Variants
//!
//! **DeiT (Data-efficient ViT):**
//! - Distillation from CNNs
//! - Works with ImageNet-scale data
//! - Achieves ViT performance with less data
//!
//! **Swin Transformer:**
//! - Hierarchical architecture
//! - Shifted windows (local self-attention)
//! - Better for dense predictions (segmentation, detection)
//!
//! **BEiT:**
//! - Self-supervised pre-training (masked image modeling)
//! - Like BERT for images
//!
//! ## Applications
//!
//! - Image classification: ImageNet, CIFAR
//! - Object detection: DETR (Detection Transformer)
//! - Segmentation: SegFormer, Mask2Former
//! - Multi-modal: CLIP (vision + language)
//! - Generation: DALL-E uses ViT components
//!
//! ## Training Tips
//!
//! ```
//! Data: >10M images for best results
//! Augmentation: RandAugment, Mixup, CutMix
//! Regularization: Dropout, stochastic depth
//! Optimizer: AdamW
//! Learning rate: Warmup + cosine decay
//! ```

fn main() {
    println!("=== Vision Transformer (ViT) ===\n");
    println!("Transformers for computer vision: Images as sequences of patches.\n");
    println!("Key Innovation: Replace CNNs with pure Transformer architecture");
    println!("Impact: State-of-the-art on ImageNet with sufficient data");
    println!("Used in: CLIP, DALL-E, Stable Diffusion");
}

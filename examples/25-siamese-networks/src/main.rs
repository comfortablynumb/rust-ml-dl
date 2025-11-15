//! # Siamese Networks: Similarity Learning
//!
//! Learn similarity between inputs using twin networks
//! Applications: Face verification, signature verification, one-shot learning
//!
//! ## The Similarity Learning Problem
//!
//! **Goal:** Determine if two inputs are similar
//! ```
//! Classification: "What is this?" → Cat, Dog, Car
//! Similarity: "Are these the same?" → Yes/No
//!
//! Challenge: Classes not known at training time!
//! Example: Face verification for new person
//! ```
//!
//! ## Siamese Architecture
//!
//! ```
//! Image 1 →  Network → Embedding 1
//!              ↓           ↓
//!         (Shared weights)  Distance
//!              ↓           ↓
//! Image 2 →  Network → Embedding 2
//!
//! If distance < threshold: Similar
//! If distance > threshold: Different
//! ```
//!
//! ## Key Innovation: Weight Sharing
//!
//! ```
//! Both images processed by SAME network
//! • Learn general feature extractor
//! • Embeddings in same space
//! • Can compare any pair
//!
//! Benefits:
//! • Half the parameters
//! • Consistent embeddings
//! • Generalizes to new classes
//! ```
//!
//! ## Loss Functions
//!
//! ### 1. Contrastive Loss
//! ```
//! L = (1-Y) * 0.5 * D² + Y * 0.5 * max(margin - D, 0)²
//!
//! Where:
//! • Y=0: Similar pair → minimize distance D
//! • Y=1: Dissimilar pair → maximize distance (up to margin)
//! • margin: Minimum separation for dissimilar pairs
//!
//! Intuition:
//! Pull similar pairs together
//! Push dissimilar pairs apart (up to margin)
//! ```
//!
//! ### 2. Triplet Loss
//! ```
//! Anchor, Positive, Negative triplets
//!
//! L = max(D(anchor, positive) - D(anchor, negative) + margin, 0)
//!
//! Goal: 
//! D(anchor, positive) + margin < D(anchor, negative)
//!
//! Example:
//! Anchor: Photo of person A
//! Positive: Another photo of person A
//! Negative: Photo of person B
//! ```
//!
//! ## Training Process
//!
//! ```
//! 1. Sample pairs/triplets
//! 2. Forward pass through twin networks
//! 3. Compute embeddings
//! 4. Calculate distance (Euclidean, cosine)
//! 5. Compute loss
//! 6. Backpropagate (weights shared!)
//! 7. Update network
//! ```
//!
//! ## Applications
//!
//! ### Face Verification
//! ```
//! Training: Learn face embeddings
//! Test: Compare new face to database
//! Use: Phone unlock, security systems
//!
//! FaceNet (Google): Triplet loss + deep CNN
//! 99.63% accuracy on LFW dataset
//! ```
//!
//! ### One-Shot Learning
//! ```
//! Problem: Learn from single example
//!
//! Traditional: Need many examples per class
//! Siamese: Learn similarity, not classes
//!
//! Application: Character recognition with few examples
//! Omniglot dataset: 1623 characters, 20 examples each
//! ```
//!
//! ### Signature Verification
//! ```
//! Determine if signature is genuine
//! Compare to stored signature embedding
//! Robust to variations in writing
//! ```
//!
//! ### Image Retrieval
//! ```
//! Find similar images in database
//! Compute embedding for query
//! Find nearest neighbors in embedding space
//! Use: Google Images, Pinterest
//! ```
//!
//! ## Distance Metrics
//!
//! ### Euclidean Distance
//! ```
//! D(x, y) = ||x - y||₂ = √(Σ(x_i - y_i)²)
//!
//! Simple, intuitive
//! Sensitive to magnitude
//! ```
//!
//! ### Cosine Similarity
//! ```
//! Similarity(x, y) = (x · y) / (||x|| · ||y||)
//! Distance = 1 - Similarity
//!
//! Measures angle, not magnitude
//! Good for high-dimensional embeddings
//! ```
//!
//! ## Variants
//!
//! **DeepFace (Facebook):**
//! - 3D face alignment
//! - Deep CNN (9 layers)
//! - 97.35% accuracy
//!
//! **FaceNet (Google):**
//! - Triplet loss
//! - Inception-style CNN
//! - 128-dimensional embeddings
//!
//! **Matching Networks:**
//! - Attention mechanism
//! - Few-shot learning
//! - Meta-learning approach
//!
//! ## Modern Context
//!
//! **Contrastive Learning (SimCLR, MoCo):**
//! - Self-supervised learning
//! - Learn representations without labels
//! - State-of-the-art pre-training
//!
//! **CLIP (OpenAI):**
//! - Dual encoder (image + text)
//! - Contrastive learning at scale
//! - Powers Stable Diffusion, DALL-E

fn main() {
    println!("=== Siamese Networks: Similarity Learning ===\n");
    println!("Twin networks with shared weights for learning similarity");
    println!("Applications: Face verification, one-shot learning, image retrieval");
    println!("Modern use: Contrastive learning (SimCLR, CLIP)");
}

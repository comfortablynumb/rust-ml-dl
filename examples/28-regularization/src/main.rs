//! # Regularization & Dropout: Preventing Overfitting
//!
//! Comprehensive guide to regularization techniques that prevent overfitting
//! and improve model generalization.
//!
//! ## The Overfitting Problem
//!
//! ```
//! Training: 99% accuracy ✓
//! Test: 60% accuracy ❌
//!
//! Model memorized training data but doesn't generalize!
//!
//! Signs of overfitting:
//! • Training loss decreases, validation loss increases
//! • Large gap between train and test performance
//! • Model too complex for dataset size
//! • High variance (sensitive to training data)
//! ```
//!
//! ## Weight Regularization (L1 & L2)
//!
//! **Add penalty to loss function for large weights**
//!
//! ### L2 Regularization (Ridge, Weight Decay)
//!
//! ```
//! Modified loss:
//! L_total = L_data + λ × Σ w²
//!
//! Where:
//! • L_data: Original loss (cross-entropy, MSE, etc.)
//! • λ: Regularization strength (0.0001 - 0.01)
//! • Σ w²: Sum of squared weights
//!
//! Effect on gradients:
//! ∂L_total/∂w = ∂L_data/∂w + 2λw
//!
//! Update rule:
//! w = w - lr × (∂L_data/∂w + 2λw)
//!   = (1 - 2λ×lr) × w - lr × ∂L_data/∂w
//!       ↑
//!   Weight decay!
//! ```
//!
//! **Why it works:**
//! ```
//! • Prevents weights from growing too large
//! • Prefers simpler models (Occam's razor)
//! • Smooths decision boundaries
//! • Distributes weights more evenly
//!
//! Example:
//! Without L2: [0, 0, 100, 0, 0]  ← Relies on single feature
//! With L2: [10, 15, 20, 12, 8]  ← Uses multiple features
//! ```
//!
//! **Typical values:**
//! ```
//! λ = 0.0001: Light regularization
//! λ = 0.001: Medium (common default)
//! λ = 0.01: Strong regularization
//! ```
//!
//! ### L1 Regularization (Lasso)
//!
//! ```
//! Modified loss:
//! L_total = L_data + λ × Σ |w|
//!
//! Gradient:
//! ∂L_total/∂w = ∂L_data/∂w + λ × sign(w)
//!
//! Effect:
//! • Pushes weights exactly to zero
//! • Creates sparse models
//! • Feature selection (some weights become 0)
//! ```
//!
//! **L1 vs L2:**
//! ```
//! L1:
//! ✅ Sparse solutions (feature selection)
//! ✅ Interpretable (fewer non-zero weights)
//! ❌ Not differentiable at 0
//!
//! L2:
//! ✅ Smooth optimization
//! ✅ Better gradient properties
//! ✅ More commonly used in deep learning
//! ❌ Doesn't create sparsity
//!
//! Elastic Net: L1 + L2 (best of both)
//! L_total = L_data + λ₁ Σ |w| + λ₂ Σ w²
//! ```
//!
//! ## Dropout (2014)
//!
//! **Randomly drop neurons during training**
//!
//! ### How It Works
//!
//! ```
//! Training:
//! For each training sample:
//!   For each neuron:
//!     With probability p: Set output to 0
//!     With probability 1-p: Keep active
//!
//! Typical p = 0.5 (drop 50% of neurons)
//!
//! Example forward pass:
//! Before dropout: [0.5, 0.8, 0.3, 0.9, 0.2]
//! After dropout:  [0.0, 0.8, 0.0, 0.9, 0.2]  ← Random!
//!                  ↑        ↑
//!              Dropped   Dropped
//! ```
//!
//! ### Inverted Dropout
//!
//! ```
//! Training:
//! mask = random(0,1) > p
//! output = (input * mask) / (1-p)
//!          ↑               ↑
//!       Random drop    Scale up to maintain expected value
//!
//! Inference:
//! output = input  ← No dropout, no scaling
//!
//! Example with p=0.5:
//! Training: [1.0, 0, 2.0, 0] / 0.5 = [2.0, 0, 4.0, 0]
//! Inference: [1.0, 1.0, 2.0, 2.0]  ← All neurons active
//!
//! Expected value matches!
//! ```
//!
//! ### Why Dropout Works
//!
//! ```
//! 1. Ensemble Effect:
//!    Each forward pass = different sub-network
//!    Training 1000 mini-batches = 1000 different networks!
//!    Inference = averaging all networks
//!
//! 2. Co-adaptation Prevention:
//!    Without dropout: Neurons rely on specific other neurons
//!    With dropout: Each neuron must be independently useful
//!    → More robust features
//!
//! 3. Noise Injection:
//!    Acts as regularization
//!    Forces network to learn redundant representations
//! ```
//!
//! ### Dropout Rates by Layer
//!
//! ```
//! Input layer: 0.1-0.2 (light dropout)
//! Hidden layers: 0.3-0.5 (moderate to heavy)
//! Output layer: 0.0 (no dropout)
//!
//! Rule of thumb:
//! • Larger layers: Higher dropout (0.5)
//! • Smaller layers: Lower dropout (0.2)
//! • CNNs: Lower dropout (0.1-0.3)
//! • Fully connected: Higher dropout (0.5)
//! ```
//!
//! ## DropConnect
//!
//! ```
//! Dropout: Drop neurons (activations)
//! DropConnect: Drop weights
//!
//! Training:
//! For each forward pass:
//!   Randomly drop weight connections
//!   M ~ Bernoulli(1-p)
//!   y = σ((W ⊙ M) × x)
//!
//! Effect:
//! • More fine-grained than dropout
//! • Can be more effective
//! • Computationally more expensive
//! • Less commonly used than dropout
//! ```
//!
//! ## Early Stopping
//!
//! **Stop training when validation performance degrades**
//!
//! ```
//! Algorithm:
//! best_val_loss = ∞
//! patience_counter = 0
//! patience = 10  ← How many epochs to wait
//!
//! For each epoch:
//!   Train on training set
//!   Evaluate on validation set
//!   
//!   if val_loss < best_val_loss:
//!     best_val_loss = val_loss
//!     save_model()  ← Checkpoint
//!     patience_counter = 0
//!   else:
//!     patience_counter += 1
//!   
//!   if patience_counter >= patience:
//!     break  ← Stop training
//!
//! Load best model from checkpoint
//! ```
//!
//! **Benefits:**
//! ```
//! ✅ Simple and effective
//! ✅ No hyperparameters to tune (just patience)
//! ✅ Works with any model
//! ✅ Prevents overfitting automatically
//! ```
//!
//! **Best Practices:**
//! ```
//! • Patience: 5-20 epochs (longer for large models)
//! • Always save best model
//! • Monitor validation loss, not accuracy
//! • Use separate validation set
//! • Combine with other regularization
//! ```
//!
//! ## Data Augmentation
//!
//! **Create more training data through transformations**
//!
//! ### Image Augmentation
//!
//! ```
//! Geometric:
//! • Random crop (224×224 from 256×256)
//! • Horizontal flip (p=0.5)
//! • Rotation (±15°)
//! • Scaling (0.8-1.2×)
//! • Translation (shift)
//! • Shear
//!
//! Photometric:
//! • Brightness adjustment (±20%)
//! • Contrast adjustment (±20%)
//! • Saturation adjustment
//! • Hue adjustment
//! • Gaussian noise
//! • Gaussian blur
//!
//! Advanced:
//! • Cutout (random patches set to 0)
//! • Mixup (blend two images)
//! • CutMix (paste patch from another image)
//! • AutoAugment (learned augmentation policy)
//! • RandAugment (random augmentation chain)
//! ```
//!
//! ### Text Augmentation
//!
//! ```
//! • Synonym replacement (word → similar word)
//! • Random insertion (add words)
//! • Random swap (swap word positions)
//! • Random deletion (remove words)
//! • Back-translation (translate → translate back)
//! ```
//!
//! ### Time Series Augmentation
//!
//! ```
//! • Jittering (add noise)
//! • Scaling (multiply by constant)
//! • Time warping (speed up/slow down)
//! • Window slicing (different subsequences)
//! • Magnitude warping
//! ```
//!
//! **Benefits:**
//! ```
//! ✅ Increases effective dataset size
//! ✅ Improves generalization
//! ✅ Reduces overfitting
//! ✅ Makes model robust to variations
//! ✅ Often better than more data
//! ```
//!
//! ## Combining Regularization Techniques
//!
//! ### Typical Configurations
//!
//! **Image Classification (ResNet-style):**
//! ```
//! • L2 weight decay: 0.0001
//! • Dropout: None (BatchNorm provides regularization)
//! • Data augmentation: Heavy (crop, flip, color jitter)
//! • Early stopping: Yes (patience=10)
//! ```
//!
//! **NLP (BERT-style):**
//! ```
//! • L2 weight decay: 0.01
//! • Dropout: 0.1 (after attention, embeddings)
//! • Data augmentation: Minimal
//! • Early stopping: Yes
//! • Label smoothing: 0.1
//! ```
//!
//! **Small Dataset:**
//! ```
//! • L2 weight decay: 0.001-0.01 (strong)
//! • Dropout: 0.5 (heavy)
//! • Data augmentation: Very heavy
//! • Early stopping: Yes (patience=20)
//! ```
//!
//! **Large Dataset:**
//! ```
//! • L2 weight decay: 0.0001 (light)
//! • Dropout: 0.0-0.2 (light or none)
//! • Data augmentation: Moderate
//! • Early stopping: Optional
//! ```
//!
//! ## Less Common Regularization Techniques
//!
//! ### Label Smoothing
//!
//! ```
//! Standard: Hard targets [0, 0, 1, 0]
//! Smoothed: Soft targets [0.05, 0.05, 0.85, 0.05]
//!
//! Formula:
//! y_smooth = (1-ε) × y + ε/K
//!
//! Where ε = 0.1 (typical), K = number of classes
//!
//! Benefits:
//! • Prevents overconfident predictions
//! • Better calibration
//! • Improves generalization
//! ```
//!
//! ### Stochastic Depth
//!
//! ```
//! Randomly drop entire layers during training
//!
//! For ResNet:
//! x = x + Block(x) with probability p
//! x = x with probability 1-p
//!
//! Benefits:
//! • Train deeper networks
//! • Implicit ensemble
//! • Used in very deep ResNets (100+)
//! ```
//!
//! ### Batch Normalization as Regularization
//!
//! ```
//! BatchNorm has regularization effect:
//! • Noise from batch statistics
//! • Acts like dropout
//! • Often sufficient for CNNs
//!
//! With BatchNorm:
//! • Can reduce/remove dropout
//! • Still use weight decay
//! ```
//!
//! ## Practical Tips
//!
//! ### How Much Regularization?
//!
//! ```
//! Too little:
//! • Training loss → 0
//! • Validation loss stays high
//! • Large train-val gap
//!
//! Too much:
//! • Training loss stays high
//! • Model underfits
//! • Low capacity
//!
//! Just right:
//! • Training loss: Low but not zero
//! • Validation loss: Close to training
//! • Small train-val gap (< 5%)
//! ```
//!
//! ### Tuning Strategy
//!
//! ```
//! 1. Start without regularization
//!    → Establish baseline, check for overfitting
//!
//! 2. Add weight decay (0.0001)
//!    → Usually helps, minimal tuning needed
//!
//! 3. Add dropout if still overfitting (0.3-0.5)
//!    → Start moderate, increase if needed
//!
//! 4. Add data augmentation
//!    → Often biggest impact
//!
//! 5. Implement early stopping
//!    → Always beneficial
//!
//! 6. Fine-tune hyperparameters
//!    → Grid search or manual tuning
//! ```
//!
//! ### Common Mistakes
//!
//! ```
//! ❌ Using dropout at inference
//! ❌ Regularizing bias terms (usually not needed)
//! ❌ Same dropout rate for all layers
//! ❌ No validation set for early stopping
//! ❌ Too aggressive regularization
//! ❌ Forgetting to turn off dropout for eval()
//!
//! ✅ Different dropout rates by layer type
//! ✅ Monitor train vs validation gap
//! ✅ Start with light regularization
//! ✅ Always use validation set
//! ✅ Combine multiple techniques
//! ```

fn main() {
    println!("=== Regularization & Dropout ===\n");

    println!("Techniques to prevent overfitting and improve generalization.\n");

    println!("📚 Techniques Covered:");
    println!("  • L1/L2 Regularization: Penalize large weights");
    println!("  • Dropout: Randomly drop neurons (0.3-0.5)");
    println!("  • Early Stopping: Stop when validation degrades");
    println!("  • Data Augmentation: Create more training data");
    println!("  • Label Smoothing: Prevent overconfidence\n");

    println!("🎯 Key Benefits:");
    println!("  • Prevents overfitting");
    println!("  • Improves test performance");
    println!("  • More robust models");
    println!("  • Better generalization\n");

    println!("💡 Typical Configuration:");
    println!("  • L2 weight decay: 0.0001-0.001");
    println!("  • Dropout: 0.3-0.5 (fully connected), 0.1-0.2 (CNNs)");
    println!("  • Early stopping: patience=10-20");
    println!("  • Data augmentation: Heavy for images\n");

    println!("See source code documentation for comprehensive explanations!");
}

//! # U-Net: Convolutional Networks for Biomedical Image Segmentation
//!
//! This example explains U-Net, the architecture that revolutionized semantic
//! segmentation, especially in medical imaging with limited training data.
//!
//! ## Semantic Segmentation: The Task
//!
//! **Goal:** Classify every pixel in an image
//!
//! ```
//! Image Classification (CNN):
//! Input: 256×256×3 image
//! Output: Single label ("cat")
//!
//! Object Detection (R-CNN, YOLO):
//! Input: 256×256×3 image
//! Output: Bounding boxes + labels
//!
//! Semantic Segmentation (U-Net):
//! Input: 256×256×3 image
//! Output: 256×256 pixel-wise labels
//!         ↑ Every pixel classified!
//!
//! Example output:
//! [0,0,0,1,1,1,0,0]  ← 0=background, 1=tumor
//! [0,0,1,1,1,1,1,0]
//! [0,1,1,1,1,1,1,0]
//! [0,0,1,1,1,1,0,0]
//! ```
//!
//! **Applications:**
//! - Medical: Tumor detection, organ segmentation, cell counting
//! - Autonomous driving: Lane detection, pedestrian segmentation
//! - Satellite: Land use classification, building detection
//! - Photography: Background removal, portrait mode
//!
//! ## The Challenge
//!
//! **Problem 1: Resolution**
//! ```
//! Classification CNN:
//! 224×224 → 112×112 → 56×56 → 28×28 → 14×14 → 7×7 → 1×1 (class)
//!           ↓
//!        Lose spatial detail (OK for classification!)
//!
//! Segmentation needs:
//! 224×224 → ... → 224×224 (pixel-wise predictions)
//!           ↓
//!        Must preserve AND recover spatial detail!
//! ```
//!
//! **Problem 2: Limited Data**
//! ```
//! ImageNet: 1.2M images
//! Medical dataset: Often < 100 images!
//!
//! Need architecture that works with small datasets
//! ```
//!
//! ## U-Net Architecture: The Solution
//!
//! **Key Innovation:** Symmetric encoder-decoder with skip connections
//!
//! ### The "U" Shape
//!
//! ```
//!                    Input (572×572×1)
//!                           ↓
//!         Contracting Path (Encoder)
//!                           ↓
//!     568×568×64  →  280×280×128  →  136×136×256
//!         ↓              ↓               ↓
//!     284×284×64  →  140×140×128  →   68×68×256
//!         ↓              ↓               ↓ Bottleneck
//!                    28×28×512
//!                        ↓
//!         Expanding Path (Decoder)
//!                        ↓
//!     52×52×256   ←  100×100×128  ←  196×196×64
//!         ↑              ↑               ↑
//!     104×104×256 ←  200×200×128  ←  392×392×64
//!         ↑              ↑               ↑
//!                 Skip Connections
//!                        ↓
//!                Output (388×388×2)
//! ```
//!
//! ### Three Core Components
//!
//! **1. Contracting Path (Encoder):**
//! ```
//! Purpose: Capture context, extract features
//!
//! Each step:
//! • Two 3×3 convolutions (ReLU)
//! • 2×2 max pooling (stride 2, downsample)
//! • Double feature channels
//!
//! Example:
//! 572×572×1 → [conv, conv] → 568×568×64
//!           → [max pool]   → 284×284×64
//!           → [conv, conv] → 280×280×128
//!           → [max pool]   → 140×140×128
//!           ...
//!
//! Captures: Edges → Textures → Parts → Objects
//! ```
//!
//! **2. Bottleneck:**
//! ```
//! Smallest spatial size, highest channels
//! Example: 28×28×512
//!
//! Contains: Most abstract/semantic features
//! ```
//!
//! **3. Expanding Path (Decoder):**
//! ```
//! Purpose: Localization, recover spatial detail
//!
//! Each step:
//! • 2×2 upconvolution (stride 2, upsample)
//! • Concatenate with cropped encoder feature map ← KEY!
//! • Two 3×3 convolutions (ReLU)
//! • Halve feature channels
//!
//! Example:
//! 28×28×512 → [upconv]     → 56×56×256
//!           → [concat]     → 56×56×512 (256+256 from encoder)
//!           → [conv, conv] → 52×52×256
//!           → [upconv]     → 104×104×128
//!           ...
//! ```
//!
//! ## Skip Connections: The Secret Sauce
//!
//! **Why Skip Connections?**
//! ```
//! Without skips:
//! Encoder → Bottleneck → Decoder
//!    ↓
//! High-resolution details LOST in bottleneck
//! Decoder struggles to recover precise boundaries
//!
//! With skips:
//! Encoder ───────────→ Decoder (concatenate)
//!    ↓                    ↑
//! Bottleneck ────────────┘
//!    ↓
//! Decoder gets both:
//! • Semantic info from bottleneck (what)
//! • Spatial details from encoder (where)
//! ```
//!
//! **Concrete Example:**
//! ```
//! Task: Segment tumor boundary
//!
//! Encoder features at 100×100:
//! • Exact pixel locations of edges
//! • Fine-grained texture
//! • Precise boundaries
//!
//! Bottleneck features at 28×28:
//! • "This is a tumor" (semantic)
//! • Approximate location
//! • Missing fine details
//!
//! Decoder at 100×100 receives:
//! • Bottleneck: "tumor here" (upsampled)
//! • Encoder skip: "exact boundary is HERE"
//! → Accurate segmentation!
//! ```
//!
//! ## Mathematical Details
//!
//! ### Convolution Layers
//! ```
//! Standard pattern:
//! Conv(3×3, ReLU, no padding) → Conv(3×3, ReLU, no padding)
//!
//! Example:
//! Input: 572×572×1
//! After conv1: 570×570×64 (lost 2 pixels per side)
//! After conv2: 568×568×64 (lost 2 more pixels)
//!
//! Note: Original U-Net uses no padding
//! Modern versions: Use padding to preserve size
//! ```
//!
//! ### Downsampling (Encoder)
//! ```
//! MaxPool(2×2, stride=2):
//! 568×568×64 → 284×284×64
//!
//! Effect:
//! • Halve spatial dimensions
//! • Keep all channels
//! • Reduce parameters
//! • Increase receptive field
//! ```
//!
//! ### Upsampling (Decoder)
//! ```
//! Method 1 - Transpose Convolution (Upconv):
//! 28×28×512 → 56×56×256
//!
//! Learnable upsampling, but can cause checkerboard artifacts
//!
//! Method 2 - Bilinear Interpolation + Conv:
//! 28×28×512 → [interpolate] → 56×56×512
//!           → [conv 1×1]    → 56×56×256
//!
//! Smoother, no artifacts, commonly used now
//! ```
//!
//! ### Skip Connection Concatenation
//! ```
//! Encoder feature: 136×136×256
//! Decoder feature: 100×100×128 (after upconv)
//!
//! Problem: Size mismatch!
//!
//! Solution: Crop encoder feature
//! 136×136×256 → [crop center] → 100×100×256
//!
//! Then concatenate:
//! [100×100×128, 100×100×256] → 100×100×384
//!
//! Modern alternative: Use padding to match sizes
//! ```
//!
//! ### Output Layer
//! ```
//! Final conv: 1×1 convolution
//! 388×388×64 → 388×388×num_classes
//!
//! For binary segmentation:
//! 388×388×64 → 388×388×1 → sigmoid → probabilities
//!
//! For multi-class:
//! 388×388×64 → 388×388×C → softmax → class probabilities
//! ```
//!
//! ## Training U-Net
//!
//! ### Loss Functions
//!
//! **1. Pixel-wise Cross-Entropy:**
//! ```
//! Standard choice for segmentation
//!
//! L = -(1/N) Σ [y_i log(ŷ_i) + (1-y_i) log(1-ŷ_i)]
//!
//! Where:
//! • N = number of pixels
//! • y_i = ground truth for pixel i
//! • ŷ_i = predicted probability for pixel i
//!
//! Problem: Imbalanced classes (99% background, 1% tumor)
//! ```
//!
//! **2. Weighted Cross-Entropy:**
//! ```
//! Add weight map to handle:
//! • Class imbalance
//! • Separation of touching objects
//!
//! L = -(1/N) Σ w_i [y_i log(ŷ_i) + (1-y_i) log(1-ŷ_i)]
//!
//! Weight map w_i higher:
//! • On rare class pixels
//! • At boundaries between objects
//! ```
//!
//! **3. Dice Loss:**
//! ```
//! Based on Dice coefficient (overlap measure)
//!
//! Dice = 2×|X ∩ Y| / (|X| + |Y|)
//!
//! Dice Loss = 1 - Dice
//!
//! Benefits:
//! • Handles class imbalance naturally
//! • Focus on overlap, not individual pixels
//! • Popular in medical imaging
//!
//! Often combined: BCE + Dice
//! ```
//!
//! ### Data Augmentation
//!
//! **Critical for small datasets!**
//!
//! ```
//! Original U-Net paper used:
//! • Elastic deformations (simulate tissue variation)
//! • Random rotations (0-180°)
//! • Random shifts
//! • Random scaling
//!
//! Modern additions:
//! • Color jittering (brightness, contrast)
//! • Gaussian noise
//! • Blur
//! • Random crops
//! • Horizontal/vertical flips
//!
//! Can train on < 30 images with heavy augmentation!
//! ```
//!
//! ### Optimization
//!
//! ```
//! Optimizer: Adam or SGD with momentum
//! Learning rate: 0.0001 - 0.001
//! Batch size: 1-8 (often limited by GPU memory)
//! Epochs: 100-500 (early stopping on validation)
//!
//! Learning rate schedule:
//! • ReduceLROnPlateau: Halve LR when validation plateaus
//! • Cosine annealing
//! • Step decay
//! ```
//!
//! ## U-Net Variants
//!
//! ### U-Net++ (Nested U-Net)
//! ```
//! Enhancement: Dense skip connections at multiple scales
//!
//!     Encoder
//!        ↓↘
//!        ↓ ↘→ Dense connections
//!        ↓   ↘
//!    Bottleneck
//!        ↑   ↗
//!        ↑ ↗→ Multiple paths
//!        ↑↗
//!     Decoder
//!
//! Benefits:
//! • Better gradient flow
//! • Multi-scale feature fusion
//! • Slight accuracy improvement
//! ```
//!
//! ### Attention U-Net
//! ```
//! Add attention gates to skip connections:
//!
//! Encoder feature → Attention Gate → Weighted feature
//!                        ↑
//!                   Decoder feature
//!
//! Attention gate:
//! • Highlights relevant regions
//! • Suppresses irrelevant features
//! • Learns where to focus
//!
//! Benefits:
//! • Better for complex images
//! • Handles variable object sizes
//! • Popular in medical imaging
//! ```
//!
//! ### 3D U-Net
//! ```
//! Extend to volumetric data (CT, MRI scans):
//!
//! 2D: Conv(3×3) → MaxPool(2×2) → Upconv(2×2)
//! 3D: Conv(3×3×3) → MaxPool(2×2×2) → Upconv(2×2×2)
//!
//! Input: 128×128×128×1 (3D volume)
//! Output: 128×128×128×C (volumetric segmentation)
//!
//! Applications:
//! • Organ segmentation in CT
//! • Brain tumor in MRI
//! • Video object segmentation
//! ```
//!
//! ### Residual U-Net
//! ```
//! Replace plain convolutions with ResNet blocks:
//!
//! Instead of: Conv → Conv
//! Use:        ResBlock (conv + skip connection)
//!
//! Benefits:
//! • Easier to train deeper networks
//! • Better gradient flow
//! • Slight performance gain
//! ```
//!
//! ## Applications & Success Stories
//!
//! ### Medical Imaging
//!
//! **Cell Segmentation (Original paper 2015):**
//! ```
//! Dataset: Neuronal structures in EM images
//! Training: 30 images (512×512)
//! Result: Won ISBI 2015 challenge
//! Dice score: 0.92 (vs 0.88 previous best)
//!
//! Key: Heavy data augmentation!
//! ```
//!
//! **Tumor Detection:**
//! ```
//! Brain tumors (MRI):
//! • 4 classes: healthy, edema, enhancing, non-enhancing
//! • Dice: 0.88-0.91
//! • Helps radiologists find small tumors
//!
//! Lung nodules (CT):
//! • Cancer screening
//! • Dice: 0.85-0.90
//! • Reduces false positives
//! ```
//!
//! **Organ Segmentation:**
//! ```
//! Liver, kidney, spleen from CT:
//! • Automate surgical planning
//! • Dice: 0.94-0.96
//! • Save radiologist hours of manual work
//! ```
//!
//! ### Autonomous Driving
//!
//! **Road Scene Segmentation:**
//! ```
//! Cityscapes dataset:
//! • 19 classes: road, car, person, etc.
//! • Real-time processing needed
//! • mIoU: 70-80% (IoU = Intersection over Union)
//!
//! Use: Lane keeping, obstacle avoidance
//! ```
//!
//! ### Satellite Imagery
//!
//! **Land Use Classification:**
//! ```
//! Classes: urban, forest, water, agriculture
//! • Large-scale mapping
//! • Environmental monitoring
//! • Urban planning
//! ```
//!
//! **Building Detection:**
//! ```
//! Segment buildings from aerial photos
//! • Disaster response
//! • Infrastructure mapping
//! • F1 score: 0.85-0.90
//! ```
//!
//! ### Photography & Art
//!
//! **Portrait Segmentation:**
//! ```
//! Separate person from background:
//! • Portrait mode blur
//! • Background replacement
//! • Real-time on mobile (optimized U-Net)
//! ```
//!
//! **Image Inpainting:**
//! ```
//! Remove objects, fill in background
//! • Photoshop-style content-aware fill
//! • Old photo restoration
//! ```
//!
//! ## Performance Metrics
//!
//! ### IoU (Intersection over Union)
//! ```
//! IoU = Area of Overlap / Area of Union
//!
//! Example:
//! Ground truth: ■■■□
//! Prediction:    □■■■
//! Overlap:       □■■□ (2 pixels)
//! Union:        ■■■■ (4 pixels)
//! IoU = 2/4 = 0.5
//!
//! Range: 0 (no overlap) to 1 (perfect)
//! Good: > 0.7, Great: > 0.85
//! ```
//!
//! ### Dice Coefficient
//! ```
//! Dice = 2×|Overlap| / (|Prediction| + |Truth|)
//!
//! Example:
//! Ground truth: 10 pixels
//! Prediction: 12 pixels
//! Overlap: 8 pixels
//! Dice = 2×8/(10+12) = 16/22 = 0.73
//!
//! Similar to IoU, often used in medical imaging
//! ```
//!
//! ### Pixel Accuracy
//! ```
//! Accuracy = Correct pixels / Total pixels
//!
//! Problem: Misleading with imbalance
//! Example: 95% background, 5% tumor
//! Predict all background: 95% accuracy! ❌
//!
//! Use IoU or Dice instead for segmentation
//! ```
//!
//! ## Modern Developments
//!
//! ### Transformers for Segmentation
//! ```
//! SegFormer, Swin-UNet (2021-2022):
//! • Replace CNN encoder with Vision Transformer
//! • Better long-range dependencies
//! • State-of-the-art results
//!
//! But:
//! • Need more data than U-Net
//! • Slower inference
//! • Higher memory
//!
//! U-Net still preferred for:
//! • Medical imaging (small datasets)
//! • Real-time applications
//! • Resource-constrained settings
//! ```
//!
//! ### Efficient U-Net
//! ```
//! MobileNet-UNet:
//! • Replace encoder with MobileNetV2
//! • Depthwise separable convolutions
//! • 10× fewer parameters
//! • 3× faster inference
//! • Minimal accuracy loss
//!
//! Use: Mobile apps, edge devices
//! ```
//!
//! ### Foundation Models
//! ```
//! Segment Anything (SAM, Meta 2023):
//! • U-Net-like architecture
//! • Trained on 1B masks
//! • Zero-shot segmentation
//! • Click → instant segmentation
//!
//! Revolution: No training needed!
//! ```
//!
//! ## Implementation Tips
//!
//! ### Memory Management
//! ```
//! U-Net is memory-hungry:
//! • Stores features at every level for skips
//! • Large input images → huge memory
//!
//! Solutions:
//! 1. Reduce batch size (even 1 works!)
//! 2. Use smaller input patches
//! 3. Mixed precision training (FP16)
//! 4. Gradient checkpointing
//! 5. Use smaller base channels (32 instead of 64)
//! ```
//!
//! ### Input Size Considerations
//! ```
//! Original: 572×572 → 388×388 (size reduction)
//! Modern: Use padding → same size input/output
//!
//! Patch-based for large images:
//! 2048×2048 image → 256×256 patches
//! Segment each patch → stitch together
//! Add overlap to avoid boundary artifacts
//! ```
//!
//! ### Batch Normalization
//! ```
//! Add after each convolution:
//! Conv → BatchNorm → ReLU
//!
//! Benefits:
//! • Faster training
//! • Higher learning rates possible
//! • Better generalization
//! • Less sensitive to initialization
//!
//! Modern standard in U-Net variants
//! ```
//!
//! ## Historical Impact
//!
//! **2015:** U-Net paper published
//! - Won ISBI cell tracking challenge
//! - 30 training images → state-of-the-art
//! - Showed power of augmentation + architecture
//!
//! **2016-2018:** Rapid adoption
//! - Became standard for medical segmentation
//! - Kaggle competitions
//! - 10,000+ citations
//!
//! **2019-2020:** Variants flourish
//! - U-Net++, Attention U-Net, 3D U-Net
//! - Applied beyond medical imaging
//! - Autonomous driving, satellite imagery
//!
//! **2021+:** Still relevant
//! - 40,000+ citations (most cited segmentation paper)
//! - Benchmark for new methods
//! - Foundation for modern architectures
//! - SAM (2023) uses U-Net-like design
//!
//! **Legacy:**
//! - Proof that architecture matters
//! - Skip connections now standard everywhere
//! - Encoder-decoder paradigm ubiquitous

fn main() {
    println!("=== U-Net: Semantic Segmentation Architecture ===\n");

    println!("This example explains U-Net, the architecture that revolutionized");
    println!("image segmentation, especially with limited training data.\n");

    println!("📚 Key Concepts Covered:");
    println!("  • Semantic segmentation vs classification");
    println!("  • Encoder-decoder architecture");
    println!("  • Skip connections for spatial detail");
    println!("  • Training with small medical datasets");
    println!("  • Data augmentation strategies");
    println!("  • Dice loss and IoU metrics\n");

    println!("🎯 Why This Matters:");
    println!("  • Revolutionized medical image analysis");
    println!("  • Enabled segmentation with < 30 training images");
    println!("  • Standard architecture for pixel-wise prediction");
    println!("  • Applied to autonomous driving, satellite imagery, photography");
    println!("  • Foundation for modern segmentation models\n");

    println!("See the source code documentation for comprehensive explanations!");
}

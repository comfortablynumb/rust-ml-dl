//! # Transfer Learning & Fine-Tuning: The Most Practical Deep Learning Workflow
//!
//! Transfer learning is how practitioners actually use deep learning in production.
//! Train on large dataset (ImageNet, BERT), then adapt to your specific task.
//!
//! ## The Core Idea
//!
//! ```
//! Traditional: Train from scratch on your data
//! • Need millions of samples
//! • Requires weeks of training
//! • Often impossible for small datasets
//!
//! Transfer Learning: Start from pre-trained model
//! • Uses knowledge from large dataset (ImageNet, Wikipedia)
//! • Fine-tune on your data (hundreds to thousands of samples)
//! • Train in hours instead of weeks
//! • Often better performance!
//! ```
//!
//! ## Why Transfer Learning Works
//!
//! ### Feature Hierarchy
//!
//! ```
//! Deep networks learn hierarchical features:
//!
//! Layer 1 (Early): Edges, colors, textures
//! • Universal! Same for all vision tasks
//! • Horizontal/vertical edges
//! • Color blobs
//!
//! Layer 2-3 (Middle): Shapes, patterns
//! • Circles, rectangles
//! • Simple textures
//! • Mostly transferable
//!
//! Layer 4-5 (Late): Task-specific features
//! • Cat ears, dog noses (for ImageNet)
//! • Your task: Different high-level features
//! • Need to adapt these layers
//! ```
//!
//! ### Intuition
//!
//! ```
//! Learning to classify images is like learning to paint:
//!
//! Traditional: Learn to paint from scratch
//! • Learn to hold brush
//! • Learn color mixing
//! • Learn composition
//! • Takes years!
//!
//! Transfer Learning: Start with painting skills
//! • Already know techniques
//! • Just learn your specific style
//! • Takes weeks, not years
//! ```
//!
//! ## Two Main Approaches
//!
//! ### 1. Feature Extraction (Freeze Early Layers)
//!
//! ```
//! Use pre-trained model as fixed feature extractor:
//!
//! Pre-trained Model:
//! Input → Conv1 → Conv2 → ... → ConvN → FC
//!         ↓      ↓           ↓       ↓
//!       Freeze Freeze     Freeze   Train (new)
//!
//! Only train:
//! • Final classification layer
//! • Maybe last conv block
//!
//! When to use:
//! ✅ Small dataset (< 1000 samples)
//! ✅ Similar to pre-training task
//! ✅ Limited compute
//! ✅ Fast training needed
//! ```
//!
//! **Example:**
//! ```
//! Pre-trained: ResNet-50 on ImageNet (1.2M images, 1000 classes)
//! Your task: Classify 5 types of flowers (500 images)
//!
//! Approach:
//! 1. Load ResNet-50 weights
//! 2. Remove final layer (1000 classes)
//! 3. Add new layer (5 classes)
//! 4. Freeze all layers except last
//! 5. Train only the new layer (2048 → 5)
//!
//! Training time: Minutes instead of hours!
//! ```
//!
//! ### 2. Fine-Tuning (Train All or Later Layers)
//!
//! ```
//! Gradually unfreeze and train more layers:
//!
//! Stage 1: Train only new head
//! Input → [Frozen CNN] → [New FC] ← Train
//!
//! Stage 2: Unfreeze later conv blocks
//! Input → [Frozen] → [Train] → [Train]
//!
//! Stage 3 (optional): Train all layers
//! Input → [Train] → [Train] → [Train]
//!
//! When to use:
//! ✅ Medium to large dataset (> 10K samples)
//! ✅ Different from pre-training task
//! ✅ Want best possible performance
//! ✅ Have compute budget
//! ```
//!
//! **Learning Rates:**
//! ```
//! Different layers = different learning rates!
//!
//! Early layers (frozen or tiny LR):
//! • LR = 0 (frozen) or 1e-5 (very small)
//! • Already learned universal features
//!
//! Middle layers (small LR):
//! • LR = 1e-4 to 1e-3
//! • Adapt features to your domain
//!
//! New layers (normal LR):
//! • LR = 1e-3 to 1e-2
//! • Random initialization, need more updates
//!
//! This is called "discriminative learning rates"
//! ```
//!
//! ## Computer Vision Transfer Learning
//!
//! ### Popular Pre-trained Models
//!
//! ```
//! ImageNet Pre-trained Models:
//!
//! ResNet Family:
//! • ResNet-18: 11M params, fast
//! • ResNet-50: 25M params, good balance
//! • ResNet-101: 44M params, best accuracy
//!
//! EfficientNet:
//! • EfficientNet-B0: 5M params, efficient
//! • EfficientNet-B7: 66M params, SOTA
//!
//! Vision Transformers:
//! • ViT-Base: 86M params
//! • ViT-Large: 307M params
//!
//! Choice depends on:
//! • Dataset size: Larger data → larger model
//! • Compute: Mobile → small, server → large
//! • Speed requirements: Real-time → small
//! ```
//!
//! ### Typical Workflow
//!
//! ```
//! 1. Choose pre-trained model
//!    dataset < 5K: ResNet-18 or EfficientNet-B0
//!    dataset > 50K: ResNet-50 or EfficientNet-B4
//!
//! 2. Replace final layer
//!    model.fc = Linear(2048, num_classes)  # For ResNet-50
//!
//! 3. Train in stages:
//!    
//!    Stage 1 (2-5 epochs): Freeze all, train head
//!    for param in model.parameters():
//!        param.requires_grad = False
//!    model.fc.requires_grad = True
//!    
//!    Stage 2 (10-20 epochs): Unfreeze all, small LR
//!    for param in model.parameters():
//!        param.requires_grad = True
//!    optimizer = Adam(model.parameters(), lr=1e-4)
//!
//! 4. Data augmentation (critical!)
//!    transforms = [
//!        RandomCrop(224),
//!        RandomHorizontalFlip(),
//!        ColorJitter(),
//!        Normalize(ImageNet_mean, ImageNet_std)  ← Use same normalization!
//!    ]
//! ```
//!
//! ## NLP Transfer Learning
//!
//! ### Pre-trained Language Models
//!
//! ```
//! BERT (Bidirectional Encoder):
//! • Pre-trained on Wikipedia + Books
//! • 110M (base) to 340M (large) params
//! • Best for: Classification, QA, NER
//!
//! GPT-2/3 (Autoregressive Decoder):
//! • Pre-trained on web text
//! • 117M to 175B params
//! • Best for: Generation, completion
//!
//! T5 (Encoder-Decoder):
//! • Pre-trained on C4 dataset
//! • 60M to 11B params
//! • Best for: Translation, summarization
//!
//! RoBERTa (Improved BERT):
//! • Better training procedure
//! • Often outperforms BERT
//! ```
//!
//! ### Fine-Tuning BERT Example
//!
//! ```
//! Task: Sentiment classification
//!
//! 1. Load pre-trained BERT
//!    model = BertForSequenceClassification(num_labels=2)
//!
//! 2. Add task-specific head (already included)
//!    [CLS] token → Linear(768, 2) → Softmax
//!
//! 3. Fine-tune entire model
//!    All layers trainable!
//!    LR = 2e-5 to 5e-5 (very small!)
//!
//! 4. Train for few epochs (2-4)
//!    BERT already knows language
//!    Just adapting to your task
//!
//! 5. Careful with overfitting
//!    Early stopping essential
//!    Dropout = 0.1
//! ```
//!
//! ## Domain Adaptation
//!
//! **When source and target domains differ**
//!
//! ### Strategy 1: Gradual Unfreezing
//!
//! ```
//! Epoch 1-2: Train head only
//! Epoch 3-5: Unfreeze last block
//! Epoch 6-10: Unfreeze second-to-last block
//! ...
//!
//! This is "progressive unfreezing"
//! Used in ULMFit, works well for NLP
//! ```
//!
//! ### Strategy 2: Discriminative Fine-Tuning
//!
//! ```
//! Different learning rates per layer:
//!
//! optimizer = [
//!     {'params': model.layer1, 'lr': 1e-5},
//!     {'params': model.layer2, 'lr': 1e-4},
//!     {'params': model.layer3, 'lr': 1e-3},
//!     {'params': model.head, 'lr': 1e-2}
//! ]
//!
//! Early layers: Learn slowly (preserve knowledge)
//! Later layers: Learn faster (adapt to task)
//! ```
//!
//! ### Strategy 3: Two-Stage Training
//!
//! ```
//! Stage 1: Feature extraction (frozen)
//! • Fast, prevents catastrophic forgetting
//! • Gets head to reasonable state
//!
//! Stage 2: Full fine-tuning
//! • Slower, with small LR
//! • Adapts entire network
//! ```
//!
//! ## Best Practices
//!
//! ### Data Preprocessing
//!
//! ```
//! ⚠️ Critical: Use same normalization as pre-training!
//!
//! ImageNet normalization:
//! mean = [0.485, 0.456, 0.406]
//! std = [0.229, 0.224, 0.225]
//!
//! Bad: Different normalization
//! → Pre-trained features don't work!
//!
//! Good: Match pre-training exactly
//! → Transferable features
//! ```
//!
//! ### Learning Rate Selection
//!
//! ```
//! Rule of thumb:
//!
//! From scratch: LR = 1e-3 to 1e-2
//! Fine-tuning: LR = 1e-5 to 1e-4
//!               ↑ 10-100× smaller!
//!
//! Why smaller?
//! • Pre-trained weights already good
//! • Don't want to destroy learned features
//! • "Fine" tuning, not "coarse" tuning
//!
//! LR schedule:
//! • Warmup: Gradually increase LR
//! • Decay: Reduce LR when plateau
//! • Cosine annealing: Smooth decay
//! ```
//!
//! ### When Transfer Learning Fails
//!
//! ```
//! ❌ Very different domains:
//!    ImageNet (natural images) → X-rays
//!    Solution: Find domain-specific pre-trained model
//!
//! ❌ Very different tasks:
//!    Classification → Segmentation
//!    Solution: Use encoder only, retrain decoder
//!
//! ❌ Tiny dataset (< 100 samples):
//!    Solution: Freeze more layers, heavy augmentation
//!
//! ❌ Wrong normalization:
//!    Solution: Match pre-training exactly
//! ```
//!
//! ## Advanced Techniques
//!
//! ### Multi-Task Learning
//!
//! ```
//! Share encoder across multiple tasks:
//!
//! Shared Encoder
//!    ↓      ↓      ↓
//! Task1  Task2  Task3
//! (head) (head) (head)
//!
//! Benefits:
//! • Better representations
//! • Data efficiency
//! • Regularization effect
//! ```
//!
//! ### Knowledge Distillation + Transfer
//!
//! ```
//! Large teacher model (pre-trained)
//!    ↓
//! Distill to small student
//!    ↓
//! Fine-tune student on target task
//!
//! Get: Small model + good performance
//! ```
//!
//! ### Self-Supervised Pre-training
//!
//! ```
//! Instead of ImageNet:
//! • SimCLR: Contrastive learning
//! • MAE: Masked autoencoders
//! • MoCo: Momentum contrast
//!
//! On your domain data (unlabeled)!
//! Then fine-tune on labeled subset
//! ```
//!
//! ## Real-World Examples
//!
//! ### Medical Imaging
//!
//! ```
//! Pre-training: ImageNet (natural images)
//! Target: X-ray classification
//!
//! Approach:
//! 1. ImageNet pre-trained ResNet-50
//! 2. Fine-tune on ChestX-ray dataset
//! 3. Heavy augmentation (rotation, zoom)
//! 4. Small LR (1e-4), long training
//!
//! Result: 85% → 92% accuracy vs from scratch
//! ```
//!
//! ### Sentiment Analysis
//!
//! ```
//! Pre-training: BERT on Wikipedia
//! Target: Movie review sentiment
//!
//! Approach:
//! 1. Load BERT-base
//! 2. Add classification head
//! 3. Fine-tune 3 epochs, LR=2e-5
//! 4. Early stopping
//!
//! Result: 89% accuracy with 5K samples
//! From scratch: Would need 50K+ samples
//! ```
//!
//! ### Object Detection
//!
//! ```
//! Pre-training: ImageNet classification
//! Target: Custom object detection
//!
//! Approach:
//! 1. YOLO with ResNet-50 backbone
//! 2. Keep backbone frozen initially
//! 3. Train detection head (10 epochs)
//! 4. Unfreeze all, fine-tune (20 epochs)
//!
//! Result: Detect custom objects with 1K images
//! ```
//!
//! ## Measuring Success
//!
//! ```
//! Compare:
//! • From scratch baseline
//! • Transfer learning (feature extraction)
//! • Transfer learning (fine-tuning)
//!
//! Metrics:
//! • Accuracy on test set
//! • Training time
//! • Convergence speed
//! • Data efficiency
//!
//! Transfer learning should:
//! ✅ Converge faster (fewer epochs)
//! ✅ Reach higher accuracy
//! ✅ Need less data
//! ✅ Be more stable
//! ```

fn main() {
    println!("=== Transfer Learning & Fine-Tuning ===\n");

    println!("The most practical deep learning workflow: Start from pre-trained models.\n");

    println!("📚 Key Concepts:");
    println!("  • Feature Extraction: Freeze early layers, train head");
    println!("  • Fine-Tuning: Train all layers with small LR");
    println!("  • Discriminative LR: Different rates per layer");
    println!("  • Domain Adaptation: Adapt to different data\n");

    println!("🎯 Why It Works:");
    println!("  • Early layers learn universal features (edges, textures)");
    println!("  • Later layers learn task-specific features");
    println!("  • Transfer knowledge from large datasets (ImageNet, Wikipedia)");
    println!("  • Train with 100× less data\n");

    println!("💡 Typical Workflow:");
    println!("  1. Load pre-trained model (ResNet-50, BERT)");
    println!("  2. Replace final layer for your task");
    println!("  3. Stage 1: Train head only (2-5 epochs)");
    println!("  4. Stage 2: Fine-tune all layers, small LR (10-20 epochs)");
    println!("  5. Use heavy data augmentation\n");

    println!("🔧 Popular Models:");
    println!("  • Vision: ResNet, EfficientNet, ViT");
    println!("  • NLP: BERT, RoBERTa, GPT-2, T5");
    println!("  • Multi-modal: CLIP\n");

    println!("See source code documentation for comprehensive explanations!");
}

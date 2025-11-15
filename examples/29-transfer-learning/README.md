# Transfer Learning & Fine-Tuning Example

The most practical deep learning workflow: Start from pre-trained models and adapt to your task.

## Overview

Transfer learning enables training deep models with limited data by leveraging knowledge from large-scale pre-training.

## Running

```bash
cargo run --package transfer-learning
```

## Key Concepts

### Why Transfer Learning Works

```
Deep networks learn hierarchical features:

Layer 1 (Early): Edges, colors, textures
• Universal across all vision tasks
• Transferable

Layer 2-3 (Middle): Shapes, patterns
• Mostly transferable
• Domain-specific

Layer 4-5 (Late): High-level features
• Task-specific
• Need adaptation
```

### Two Main Approaches

#### 1. Feature Extraction (Freeze Early Layers)

```
Input → [Frozen Conv Layers] → [New Trainable Head]

When to use:
✅ Small dataset (< 1000 samples)
✅ Similar to pre-training task
✅ Limited compute
✅ Fast training needed

Training time: Minutes instead of hours!
```

#### 2. Fine-Tuning (Train All Layers)

```
Stage 1: Train head only (2-5 epochs)
Stage 2: Unfreeze all, small LR (10-20 epochs)

When to use:
✅ Medium to large dataset (> 10K samples)
✅ Different from pre-training task
✅ Want best possible performance
✅ Have compute budget
```

## Workflow

### Computer Vision

```
1. Choose pre-trained model
   • Small dataset: ResNet-18, EfficientNet-B0
   • Large dataset: ResNet-50, EfficientNet-B4

2. Replace final layer
   model.fc = Linear(2048, num_classes)

3. Train in stages
   Stage 1: Freeze all except head (2-5 epochs)
   Stage 2: Unfreeze all, LR=1e-4 (10-20 epochs)

4. Critical: Use same normalization as pre-training!
   ImageNet: mean=[0.485, 0.456, 0.406]
             std=[0.229, 0.224, 0.225]
```

### NLP (BERT Example)

```
1. Load pre-trained BERT
   model = BertForSequenceClassification(num_labels=2)

2. Fine-tune entire model
   LR = 2e-5 to 5e-5 (very small!)
   Epochs = 2-4 (few epochs needed)

3. Watch for overfitting
   Early stopping essential
   Dropout = 0.1
```

## Discriminative Learning Rates

```
Different layers need different learning rates:

Early layers:  LR = 1e-5  (preserve universal features)
Middle layers: LR = 1e-4  (adapt to domain)
New layers:    LR = 1e-3  (random init, need updates)

This prevents catastrophic forgetting!
```

## Domain Adaptation Strategies

### Progressive Unfreezing

```
Epoch 1-2:   Train head only
Epoch 3-5:   Unfreeze last block
Epoch 6-10:  Unfreeze second-to-last block
...

Gradual adaptation minimizes forgetting
```

### Two-Stage Training

```
Stage 1: Feature extraction (frozen backbone)
• Fast, prevents catastrophic forgetting
• Gets head to reasonable state

Stage 2: Full fine-tuning (small LR)
• Slower, adapts entire network
• Better final performance
```

## Popular Pre-trained Models

### Vision

- **ResNet**: ResNet-18 (11M), ResNet-50 (25M), ResNet-101 (44M)
- **EfficientNet**: B0 (5M, efficient) to B7 (66M, SOTA)
- **Vision Transformers**: ViT-Base (86M), ViT-Large (307M)

### NLP

- **BERT**: Base (110M), Large (340M) - Best for classification, QA
- **RoBERTa**: Improved BERT training - Often better performance
- **GPT-2/3**: 117M to 175B - Best for generation
- **T5**: 60M to 11B - Best for translation, summarization

## Best Practices

### Learning Rate Selection

```
Rule of thumb:

From scratch: LR = 1e-3 to 1e-2
Fine-tuning:  LR = 1e-5 to 1e-4
              ↑ 10-100× smaller!

Why? Pre-trained weights already good
Don't destroy learned features!
```

### Data Preprocessing

```
⚠️ Critical: Match pre-training normalization!

Bad:  Different normalization
      → Pre-trained features don't work!

Good: Exact same normalization
      → Transferable features
```

### When Transfer Learning Fails

```
❌ Very different domains:
   Natural images → X-rays
   Solution: Find domain-specific pre-trained model

❌ Very different tasks:
   Classification → Segmentation
   Solution: Use encoder only, retrain decoder

❌ Tiny dataset (< 100 samples):
   Solution: Freeze more layers, heavy augmentation

❌ Wrong normalization:
   Solution: Match pre-training exactly
```

## Real-World Examples

### Medical Imaging

```
Pre-training: ImageNet (natural images)
Target: X-ray classification

Result: 85% → 92% accuracy vs training from scratch
With 10× less data!
```

### Sentiment Analysis

```
Pre-training: BERT on Wikipedia
Target: Movie review sentiment

Result: 89% accuracy with 5K samples
From scratch would need 50K+ samples
```

### Object Detection

```
Pre-training: ImageNet classification
Target: Custom object detection

Result: Detect custom objects with 1K images
From scratch would need 10K+ images
```

## Impact

```
Transfer learning is how practitioners actually use deep learning:

✅ Train with 10-100× less data
✅ Converge 10× faster
✅ Achieve better final performance
✅ Enable deep learning for small datasets

Powers modern AI applications:
• Medical imaging diagnosis
• Custom object detection
• Multilingual NLP
• Domain-specific classification
```

## Papers

- [Deep Learning](https://www.nature.com/articles/nature14539) (LeCun et al., 2015)
- [ImageNet Pre-training](https://arxiv.org/abs/1811.08883) (Huh et al., 2016)
- [ULMFit](https://arxiv.org/abs/1801.06146) (Howard & Ruder, 2018) - NLP transfer learning
- [BERT](https://arxiv.org/abs/1810.04805) (Devlin et al., 2018)
- [Rethinking ImageNet Pre-training](https://arxiv.org/abs/1811.08883) (He et al., 2018)

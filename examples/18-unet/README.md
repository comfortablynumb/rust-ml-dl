# U-Net Example

This example explains **U-Net**, the convolutional architecture for biomedical image segmentation that works with very few training images.

## Overview

U-Net revolutionized semantic segmentation by combining:
1. **Encoder-decoder** architecture for multi-scale features
2. **Skip connections** for spatial localization
3. **Heavy data augmentation** for small datasets

## Running the Example

```bash
cargo run --package unet
```

## The Segmentation Task

Unlike classification (one label per image), **segmentation** assigns a label to every pixel:

```
Input:  256×256 medical image
Output: 256×256 segmentation mask
        Each pixel classified!

Example:
[0,0,0,1,1,1,0,0]  ← 0 = healthy tissue
[0,0,1,1,1,1,1,0]  ← 1 = tumor
[0,1,1,1,1,1,1,0]
[0,0,1,1,1,1,0,0]
```

## Architecture Overview

```
          Input (572×572)
               ↓
    ┌─────────────────────┐
    │   Contracting Path  │ (Encoder)
    │   (Downsampling)    │
    │         ↓           │
    │    Bottleneck       │ (28×28×512)
    │         ↓           │
    │   Expanding Path    │ (Decoder)
    │   (Upsampling)      │
    └─────────────────────┘
               ↓
        Output (388×388)

Skip connections: Encoder → Decoder at each level
```

## Key Components

### 1. Encoder (Contracting Path)

Extracts features at multiple scales:

```
572×572×1   →  568×568×64   (2× conv 3×3)
            →  284×284×64   (max pool 2×2)
            →  280×280×128  (2× conv 3×3)
            →  140×140×128  (max pool 2×2)
            ...continues to bottleneck
```

### 2. Bottleneck

Most abstract features: `28×28×512`

### 3. Decoder (Expanding Path)

Recovers spatial resolution:

```
28×28×512   →  56×56×256   (upconv 2×2)
            →  56×56×512   (concatenate encoder features)
            →  52×52×256   (2× conv 3×3)
            →  104×104×128 (upconv 2×2)
            ...continues to output
```

## Skip Connections: The Secret

```
Encoder Features (140×140×128)
         │
         │ Crop to match
         │
         └──────────→ Concatenate
                          ↓
         Decoder (100×100×256 + 128)
                          ↓
                   Rich features with
                   spatial detail!
```

**Why they work:**
- **Encoder**: Precise spatial information (where)
- **Bottleneck**: Semantic understanding (what)
- **Skip connections**: Provide both to decoder!

## Loss Functions

### 1. Pixel-wise Cross-Entropy

```
Standard: L = -(1/N) Σ [y log(ŷ) + (1-y) log(1-ŷ)]

Problem: Class imbalance (99% background, 1% object)
```

### 2. Dice Loss

```
Dice = 2×|Intersection| / (|Pred| + |Truth|)
Dice Loss = 1 - Dice

Better for:
✅ Class imbalance
✅ Small objects
✅ Medical imaging

Often combined: BCE + Dice
```

### 3. Weighted Cross-Entropy

```
Weight rare pixels higher:
L = -(1/N) Σ w(x) [y log(ŷ) + (1-y) log(1-ŷ)]

Use for:
• Boundaries between touching objects
• Rare classes
```

## Data Augmentation Strategy

**Critical for small datasets!**

```
Geometric:
✅ Random rotations (0-180°)
✅ Elastic deformations ← KEY for medical
✅ Random scaling (0.8-1.2×)
✅ Horizontal/vertical flips
✅ Random crops

Photometric:
✅ Brightness adjustment
✅ Contrast adjustment
✅ Gaussian noise
✅ Blur

Can train on < 30 images with heavy augmentation!
```

## Training Configuration

```
Optimizer: Adam (lr=0.0001) or SGD (lr=0.01, momentum=0.9)
Batch size: 1-4 (limited by GPU memory)
Loss: Dice + BCE
Epochs: 100-500 with early stopping

Augmentation: Yes (essential!)
Normalization: BatchNorm after each conv
```

## U-Net Variants

### U-Net++

```
Dense skip connections:
• Multiple nested paths
• Better gradient flow
• +2-3% performance
```

### Attention U-Net

```
Attention gates on skips:
• Highlight relevant features
• Suppress noise
• Better for complex images
```

### 3D U-Net

```
For volumetric data (CT/MRI):
• 3D convolutions (3×3×3)
• 3D max pooling (2×2×2)
• Process entire 3D volume
```

### Residual U-Net

```
ResNet blocks instead of plain convs:
• Easier to train deeper
• Better gradient flow
```

## Applications

### Medical Imaging

**Cell Segmentation:**
- Original use case (2015)
- Won ISBI challenge
- 30 images → state-of-the-art

**Tumor Detection:**
- Brain MRI: Dice 0.88-0.91
- Lung CT: Dice 0.85-0.90
- Liver CT: Dice 0.94-0.96

**Organ Segmentation:**
- Automate surgical planning
- Save hours of manual work
- Dice > 0.95

### Autonomous Driving

```
Road scene segmentation:
• 19 classes (road, car, person...)
• Cityscapes dataset
• mIoU: 70-80%
• Real-time inference required
```

### Satellite Imagery

```
Land use classification:
• Urban, forest, water, agriculture
• Large-scale mapping
• Building detection: F1 0.85-0.90
```

### Photography

```
Portrait mode:
• Person/background separation
• Real-time on mobile
• Background blur/replacement
```

## Evaluation Metrics

### IoU (Intersection over Union)

```
IoU = Overlap / Union

Example:
Truth:  ■■■□□
Pred:   □■■■□
Overlap: 2, Union: 4
IoU = 2/4 = 0.5

Good: > 0.7
Great: > 0.85
```

### Dice Coefficient

```
Dice = 2×Overlap / (Pred + Truth)

Similar to IoU
Popular in medical imaging
Range: 0 (bad) to 1 (perfect)
```

## Modern Context

### Still Relevant?

**YES for:**
✅ Medical imaging (small datasets)
✅ Real-time applications
✅ Resource constraints
✅ When interpretability matters

**Transformers** better for:
- Large datasets
- State-of-the-art accuracy
- When compute is available

### Segment Anything (SAM, 2023)

```
Meta's SAM uses U-Net-like architecture:
• Trained on 1B masks
• Zero-shot segmentation
• Click → instant result

Shows U-Net design still state-of-the-art!
```

## Implementation Tips

### Memory Management

```
U-Net is memory-hungry!

Solutions:
1. Smaller batch size (1-2)
2. Patch-based processing
3. Fewer base channels (32 vs 64)
4. Mixed precision (FP16)
5. Gradient checkpointing
```

### Input Size

```
Original: Variable size output (cropping)
Modern: Use padding → same size input/output

For large images:
• Process as patches (256×256)
• Add overlap to avoid artifacts
• Stitch predictions together
```

## Historical Impact

**2015:** Paper published (Ronneberger et al.)
- Won ISBI cell tracking challenge
- 30 images → beat all competitors
- Showed architecture + augmentation power

**2016-2018:** Rapid adoption
- Standard for medical segmentation
- 10,000+ citations in 3 years

**2019-2020:** Variants flourish
- U-Net++, Attention U-Net, 3D U-Net
- Beyond medical: driving, satellites

**2021+:** Foundation for modern work
- 40,000+ citations (most cited segmentation)
- SAM (2023) uses similar design
- Still the baseline to beat

## Further Reading

- [U-Net paper](https://arxiv.org/abs/1505.04597) (Ronneberger et al., 2015)
- [U-Net++](https://arxiv.org/abs/1807.10165) (Zhou et al., 2018)
- [Attention U-Net](https://arxiv.org/abs/1804.03999) (Oktay et al., 2018)
- [3D U-Net](https://arxiv.org/abs/1606.06650) (Çiçek et al., 2016)
- [nnU-Net](https://arxiv.org/abs/1904.08128) (Isensee et al., 2019) - Self-configuring

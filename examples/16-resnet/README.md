# Residual Network (ResNet) Example

This example demonstrates ResNet, the revolutionary architecture that solved the degradation problem and enabled training of extremely deep neural networks.

## Overview

ResNets introduce **skip connections** (residual connections) that allow gradients to flow directly through the network, enabling training of networks with 100+ layers.

**Key Paper:** "Deep Residual Learning for Image Recognition" (He et al., 2015)

## Running the Example

```bash
cargo run --package resnet
```

## The Problem ResNet Solved

### Degradation Problem

Before ResNet (2015), a puzzling phenomenon occurred:

```
Deeper networks should be at least as good as shallow ones
(just learn identity for extra layers)

BUT in practice:
20-layer network: 91% accuracy
56-layer network: 88% accuracy ❌
```

**Deeper ≠ Better!**

This wasn't overfitting (training error was also worse). The network couldn't even learn the identity mapping.

## The Solution: Skip Connections

Instead of learning `H(x)` directly, learn the residual `F(x) = H(x) - x`:

```
Output = F(x) + x

Where:
F(x) = learned residual (what to add)
x = input (passed through unchanged)
```

**Benefits:**
- Easy to learn identity: just set F(x) = 0
- Gradients flow directly through skip connections
- Each layer refines the representation

## Architecture

### Basic Block (ResNet-18, ResNet-34)

```
Input x
   ├─────────────────┐
   │                 │
   ↓                 │
Conv 3×3 (stride 1)  │
   ↓                 │
BatchNorm + ReLU     │
   ↓                 │
Conv 3×3 (stride 1)  │
   ↓                 │
BatchNorm            │
   ↓                 │
   + ←───────────────┘
   ↓
  ReLU
   ↓
Output
```

### Bottleneck Block (ResNet-50, ResNet-101, ResNet-152)

```
Input x (256D)
   ├────────────────────┐
   │                    │
   ↓                    │
Conv 1×1 (64 filters)   │  Reduce dimensions
   ↓                    │
BatchNorm + ReLU        │
   ↓                    │
Conv 3×3 (64 filters)   │  Process
   ↓                    │
BatchNorm + ReLU        │
   ↓                    │
Conv 1×1 (256 filters)  │  Restore dimensions
   ↓                    │
BatchNorm               │
   ↓                    │
   + ←──────────────────┘
   ↓
  ReLU
   ↓
Output (256D)
```

**Advantages:**
- 3× fewer operations than Basic Block
- Enables very deep networks (50-152 layers)
- Better efficiency

## ResNet Variants

| Model | Layers | Parameters | Top-5 Error (ImageNet) |
|-------|--------|------------|------------------------|
| ResNet-18 | 18 | 11.7M | 10.76% |
| ResNet-34 | 34 | 21.8M | 9.58% |
| ResNet-50 | 50 | 25.6M | 7.13% |
| ResNet-101 | 101 | 44.5M | 6.44% |
| ResNet-152 | 152 | 60.2M | 6.16% |

### Full ResNet-50 Architecture

```
Input: 224×224×3 image
   ↓
Conv 7×7, 64 filters, stride 2
   ↓
MaxPool 3×3, stride 2
   ↓
[Bottleneck × 3] → 64 filters
   ↓
[Bottleneck × 4] → 128 filters (stride 2)
   ↓
[Bottleneck × 6] → 256 filters (stride 2)
   ↓
[Bottleneck × 3] → 512 filters (stride 2)
   ↓
Global Average Pool
   ↓
Fully Connected (1000 classes)
   ↓
Softmax
```

## Why Skip Connections Work

### 1. Gradient Flow

**Without skip connections:**
```
∂L/∂x₁ = ∂L/∂x₅₀ · ∂x₅₀/∂x₄₉ · ... · ∂x₂/∂x₁
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         Many multiplications → vanishing gradients
```

**With skip connections:**
```
∂L/∂x₁ = ∂L/∂x₅₀ · (1 + ∂F/∂x)
                   ^
         Gradient highway!
```

The `+1` term provides a direct path for gradients.

### 2. Identity Mapping

If a layer doesn't help, it can learn to:
```
F(x) ≈ 0
Output ≈ x (identity)
```

This is much easier than learning `H(x) = x` from scratch.

### 3. Ensemble Effect

A ResNet can be viewed as an ensemble of exponentially many paths:

```
ResNet with n blocks = 2ⁿ different paths

Example: ResNet-50
- 16 residual blocks
- 2¹⁶ = 65,536 paths!
```

Each path has a different effective depth, creating an implicit ensemble.

## ImageNet 2015 Results

ResNet **won** ILSVRC 2015:

```
Previous Best (VGG): 7.3% error
ResNet-152: 3.57% error (superhuman!)
```

**Key Achievement:**
- First model to surpass human-level performance (5.1% error)
- 152 layers deep (8× deeper than VGG)
- Fewer parameters than VGG despite being deeper

## Training Tips

1. **Batch Normalization**
   - After every convolution
   - Before activation (usually)
   - Critical for training deep networks

2. **Learning Rate Schedule**
   - Start: 0.1
   - Divide by 10 when plateau
   - Typical: epochs 30, 60, 90

3. **Weight Initialization**
   - Kaiming/He initialization
   - Designed for ReLU activations

4. **Data Augmentation**
   - Random crops (224×224 from 256×256)
   - Horizontal flips
   - Color jittering
   - Normalization (ImageNet mean/std)

5. **Optimizer**
   - SGD with momentum (0.9)
   - Weight decay: 0.0001
   - Batch size: 256

6. **Training Time**
   - ImageNet: 2-3 weeks on 8 GPUs
   - Can use pretrained models for transfer learning

## Modern Variants

### ResNeXt (2017)
- Adds "cardinality" dimension
- Multiple parallel paths
- Better accuracy with same complexity

### Wide ResNet (2016)
- Wider layers (more filters)
- Shallower than original
- Better speed/accuracy tradeoff

### DenseNet (2017)
- Connect EVERY layer to every other
- Maximum gradient flow
- More parameter efficient

### EfficientNet (2019)
- Compound scaling (depth + width + resolution)
- Better accuracy with fewer parameters
- State-of-the-art efficiency

### ResNet Strikes Back (2021)
- Improved training procedures
- Matches Vision Transformers
- Shows ResNets still competitive

## Applications

### Image Classification
- Standard backbone for ImageNet
- Transfer learning for custom tasks
- Feature extraction

### Object Detection
- Faster R-CNN + ResNet
- Mask R-CNN
- YOLO variants

### Semantic Segmentation
- FCN with ResNet backbone
- DeepLab
- U-Net variants

### Face Recognition
- ArcFace
- CosFace
- FaceNet variations

### Medical Imaging
- Disease detection
- Organ segmentation
- Lesion classification

## Transfer Learning

ResNet's most common use today:

```python
# Pseudocode
pretrained_resnet = load_pretrained_resnet50()

# Option 1: Feature extractor
features = pretrained_resnet.extract_features(image)
custom_classifier.train(features, labels)

# Option 2: Fine-tuning
pretrained_resnet.freeze_early_layers()
pretrained_resnet.train_on_custom_dataset()
```

**When to use:**
- Small dataset (< 10k images)
- Similar domain (natural images)
- Need quick results

## Comparison with Other Architectures

| Architecture | Year | Depth | Parameters | ImageNet Top-5 |
|--------------|------|-------|------------|----------------|
| AlexNet | 2012 | 8 | 60M | 16.4% |
| VGG-16 | 2014 | 16 | 138M | 7.3% |
| **ResNet-50** | **2015** | **50** | **25.6M** | **7.1%** |
| Inception-v3 | 2015 | 48 | 23.8M | 5.6% |
| DenseNet-201 | 2017 | 201 | 20M | 6.3% |
| EfficientNet-B7 | 2019 | - | 66M | 3.9% |
| Vision Transformer | 2021 | 12 | 86M | 4.6% |

**ResNet advantages:**
- Good accuracy/parameter ratio
- Widely supported
- Proven track record
- Easy to understand and modify

## Historical Impact

**2015:** ResNet paper published
- Solved degradation problem
- Enabled very deep networks (100+ layers)
- Won ImageNet, COCO, ILSVRC competitions

**2016-2018:** Became standard backbone
- Object detection (Faster R-CNN)
- Segmentation (Mask R-CNN)
- Most CV research uses ResNet

**2019-2020:** EfficientNet challenge
- Better efficiency
- ResNet still preferred for many tasks

**2021+:** Vision Transformer era
- Attention-based models competitive
- ResNets remain practical choice
- Hybrid approaches emerging

**Legacy:**
- Most cited CV paper (100,000+ citations)
- Changed how we think about depth
- Inspired countless architectures

## Implementation Considerations

### Memory Usage

Deeper = more memory for activations:

```
ResNet-50: ~11GB activations (batch size 32)
ResNet-152: ~25GB activations (batch size 32)
```

**Solutions:**
- Gradient checkpointing (recompute activations)
- Smaller batch sizes
- Mixed precision training

### Inference Speed

| Model | FLOPs | Latency (GPU) | Latency (Mobile) |
|-------|-------|---------------|------------------|
| ResNet-18 | 1.8B | 2ms | 50ms |
| ResNet-50 | 4.1B | 4ms | 150ms |
| ResNet-152 | 11.3B | 12ms | 500ms |

**For production:**
- ResNet-18/34 for mobile/edge
- ResNet-50 for servers
- Model compression (pruning, quantization)

## Further Reading

- [Deep Residual Learning](https://arxiv.org/abs/1512.03385) (He et al., 2015) - Original paper
- [Identity Mappings](https://arxiv.org/abs/1603.05027) (He et al., 2016) - Improved ResNet
- [ResNeXt](https://arxiv.org/abs/1611.05431) - Aggregated residual transformations
- [Wide ResNet](https://arxiv.org/abs/1605.07146) - Width vs depth
- [EfficientNet](https://arxiv.org/abs/1905.11946) - Compound scaling
- [ResNet Strikes Back](https://arxiv.org/abs/2110.00476) (2021) - Modern training

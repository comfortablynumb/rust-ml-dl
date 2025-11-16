# Normalization Techniques Example

Essential techniques for training deep neural networks: BatchNorm, LayerNorm, GroupNorm, and InstanceNorm.

## Overview

Normalization layers stabilize training, enable higher learning rates, and make deep networks trainable.

## Running

```bash
cargo run --package normalization
```

## Key Techniques

### BatchNorm (2015)
```
Normalize across batch dimension
Best for: CNNs with large batches (>8)
Impact: Enabled ResNet-152, VGG-19
```

### LayerNorm (2016)
```
Normalize across feature dimension
Best for: Transformers, RNNs
Impact: Foundation of BERT, GPT, all modern NLP
```

### GroupNorm (2018)
```
Split channels into groups, normalize within groups
Best for: CNNs with small batches (1-4)
Impact: Standard for object detection, segmentation
```

### InstanceNorm
```
Normalize each sample+channel independently
Best for: Style transfer, GANs
Impact: StyleGAN, Pix2Pix
```

## Quick Decision Tree

```
CNNs?
  ├─ Large batch (>8)  → BatchNorm
  └─ Small batch (1-4) → GroupNorm

Transformers/RNNs?
  └─ LayerNorm

Style Transfer/GANs?
  └─ InstanceNorm
```

## Benefits

- **10-100× faster training**
- **Higher learning rates** (0.1 vs 0.001)
- **Enables very deep networks** (100+ layers)
- **Less sensitive to initialization**
- **Regularization effect**

## Papers

- [Batch Normalization](https://arxiv.org/abs/1502.03167) (Ioffe & Szegedy, 2015)
- [Layer Normalization](https://arxiv.org/abs/1607.06450) (Ba et al., 2016)
- [Group Normalization](https://arxiv.org/abs/1803.08494) (Wu & He, 2018)

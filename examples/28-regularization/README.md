# Regularization & Dropout Example

Essential techniques to prevent overfitting and improve model generalization.

## Overview

Regularization techniques constrain model complexity to improve test performance and prevent memorization.

## Running

```bash
cargo run --package regularization
```

## Key Techniques

### L2 Regularization (Weight Decay)
```
Loss = Data Loss + λ × Σw²

Effect: Prevents large weights, smoother models
Typical λ: 0.0001 - 0.01
```

### L1 Regularization (Lasso)
```
Loss = Data Loss + λ × Σ|w|

Effect: Creates sparse models (feature selection)
Use: When you want few non-zero weights
```

### Dropout
```
Training: Randomly drop p% of neurons
Typical rates:
  • Fully connected: 0.5
  • CNNs: 0.1-0.3
  • Input: 0.1-0.2

Effect: Ensemble of sub-networks
```

### Early Stopping
```
Stop when validation loss stops improving

patience = 10 epochs
Simple, effective, always use!
```

### Data Augmentation
```
Images:
  • Random crop, flip, rotation
  • Color jitter, noise, blur
  • Mixup, CutMix, AutoAugment

Text:
  • Synonym replacement
  • Back-translation

Effect: Increases effective dataset size
```

## Typical Configurations

### Image Classification (ResNet)
- L2: 0.0001
- Dropout: None (BatchNorm provides regularization)
- Augmentation: Heavy
- Early stopping: Yes

### NLP (Transformer)
- L2: 0.01
- Dropout: 0.1
- Augmentation: Light
- Early stopping: Yes
- Label smoothing: 0.1

### Small Dataset
- L2: 0.001-0.01 (strong)
- Dropout: 0.5 (heavy)
- Augmentation: Very heavy
- Early stopping: Yes (patience=20)

## Papers

- [Dropout](https://jmlr.org/papers/v15/srivastava14a.html) (Srivastava et al., 2014)
- [DropConnect](http://proceedings.mlr.press/v28/wan13.html) (Wan et al., 2013)
- [L2 Regularization](https://papers.nips.cc/paper/563-a-simple-weight-decay-can-improve-generalization) (Krogh & Hertz, 1992)

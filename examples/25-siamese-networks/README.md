# Siamese Networks Example

Learn similarity between inputs using twin networks with shared weights.

## Overview

Siamese networks learn embeddings where similar inputs are close and dissimilar inputs are far.

## Running

```bash
cargo run --package siamese-networks
```

## Key Concept

```
Input 1 → Network \
                   → Compare embeddings → Similar/Different
Input 2 → Network /
   (shared weights)
```

## Loss Functions

- **Contrastive Loss**: Pull similar together, push dissimilar apart
- **Triplet Loss**: Anchor closer to positive than negative

## Applications

- Face verification (FaceNet)
- One-shot learning
- Image retrieval
- Signature verification
- Modern: CLIP, contrastive learning

## Papers

- [Siamese Neural Networks](http://www.cs.toronto.edu/~gkoch/files/msc-thesis.pdf) (Koch et al., 2015)
- [FaceNet](https://arxiv.org/abs/1503.03832) (Schroff et al., 2015)

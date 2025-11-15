# Vision Transformer (ViT) Example

Applies Transformer architecture to images by treating them as sequences of patches.

## Overview

ViT splits images into patches and processes them with a standard Transformer encoder, achieving state-of-the-art results.

## Running

```bash
cargo run --package vision-transformer
```

## Key Concept

```
Image (224×224) → Patches (16×16) → 196 tokens → Transformer → Class prediction
```

## Why It Works

- **Global attention**: Sees entire image from layer 1
- **Scalability**: Better than CNNs with massive data
- **Used in**: CLIP, DALL-E, modern vision systems

## Paper

[An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020)

# Autoencoder Example

This example demonstrates autoencoders, unsupervised neural networks for learning compressed representations of data.

## Overview

Autoencoders learn to compress data into a lower-dimensional latent space and then reconstruct the original data from this compressed representation.

## Running the Example

```bash
cargo run --package autoencoder
```

## Architecture

```
Input → Encoder → Bottleneck → Decoder → Output
        (compress)  (latent)   (reconstruct)
```

**Example:**
- Input: 784 pixels (28×28 image)
- Latent: 32 dimensions (24.5x compression!)
- Output: 784 pixels (reconstructed)

## Key Components

### Encoder
Compresses high-dimensional input into compact latent representation:
```
z = encoder(x)
```

### Bottleneck (Latent Space)
The compressed representation that captures essential features:
- Lower dimensional than input
- Forces network to learn important patterns
- Can be used for visualization

### Decoder
Reconstructs original input from latent code:
```
x̂ = decoder(z)
```

## Training

**Objective:** Minimize reconstruction error

```
Loss = ||x - x̂||²  (MSE)
or
Loss = -Σ[x·log(x̂) + (1-x)·log(1-x̂)]  (Binary Cross-Entropy)
```

## Types of Autoencoders

### 1. Vanilla Autoencoder
Basic compress-and-reconstruct architecture.

### 2. Denoising Autoencoder (DAE)
```
Noisy Input → Autoencoder → Clean Output
```
Learns to remove noise.

### 3. Variational Autoencoder (VAE)
```
Input → (μ, σ) → Sample z ~ N(μ,σ) → Decode
```
Can generate new samples!

### 4. Sparse Autoencoder
Enforces sparsity in latent activations for interpretability.

### 5. Contractive Autoencoder
Learns robust representations insensitive to small input changes.

## Applications

### Dimensionality Reduction
- Like PCA but non-linear
- Better for complex data
- Feature extraction

### Anomaly Detection
- Normal data: Low reconstruction error
- Anomalies: High reconstruction error
- Use cases: Fraud detection, defect detection

### Denoising
- Remove noise from images
- Audio noise reduction
- Medical image enhancement

### Data Generation (VAE)
- Generate new faces
- Create artwork
- Drug molecule design

### Compression
- Image/video compression
- Task-specific compression
- Often better than traditional methods

## Autoencoder vs PCA

| Feature | PCA | Autoencoder |
|---------|-----|-------------|
| Type | Linear | Non-linear |
| Training | Closed-form | Iterative |
| Speed | Fast | Slower |
| Flexibility | Low | High |
| Complex patterns | Limited | Excellent |

## Modern Applications

**Generative AI:**
- Stable Diffusion (VAE + Diffusion)
- DALL-E (VAE-based)

**Compression:**
- Google's RAISR
- Better than JPEG for specific domains

**Science:**
- Drug discovery
- Protein structure prediction (AlphaFold)

**Security:**
- Network intrusion detection
- Financial fraud detection

## Training Tips

1. Start with simple architecture
2. Use appropriate activation functions
3. Match loss function to data type
4. Regularize bottleneck (dropout, weight decay)
5. Monitor validation reconstruction error
6. Visualize latent space

## Further Reading

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Denoising Autoencoders](http://www.jmlr.org/papers/v11/vincent10a.html)
- [Tutorial on Autoencoders](https://www.deeplearningbook.org/contents/autoencoders.html)

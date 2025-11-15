# Variational Autoencoder (VAE) Example

This example explains **Variational Autoencoders (VAE)**, a probabilistic generative model that combines deep learning with variational inference.

## Overview

VAE extends standard autoencoders with probabilistic encoding, enabling:
- **Generation** of new, realistic samples
- **Smooth latent spaces** for interpolation
- **Anomaly detection** via reconstruction error

## Running the Example

```bash
cargo run --package vae
```

## Key Innovation

### Standard Autoencoder (Can't Generate)

```
x → Encoder → z → Decoder → x̂

Problem: Latent space has "holes"
Random z → garbage output ❌
```

### VAE (Can Generate!)

```
x → Encoder → (μ, σ) → Sample z ~ N(μ, σ²) → Decoder → x̂

Benefit: Smooth, continuous latent space
Random z ~ N(0,1) → realistic output ✅
```

## Architecture

```
Input x (784-dim)
     ↓
Encoder Network
     ↓    ↓
     μ    log_σ²  (two outputs!)
     ↓    ↓
Reparameterization: z = μ + σ⊙ε  where ε ~ N(0,1)
     ↓
     z (latent)
     ↓
Decoder Network
     ↓
Output x̂ (784-dim)
```

## Reparameterization Trick

**The key to making VAE trainable:**

```
Problem: Can't backprop through z ~ N(μ, σ²)

Solution:
ε ~ N(0, 1)      ← Sample from standard normal
z = μ + σ ⊙ ε    ← Deterministic transformation

Now gradients flow through μ and σ!
```

## Loss Function

```
L = Reconstruction Loss + KL Divergence

L = L_recon + KL[q(z|x) || p(z)]
```

### 1. Reconstruction Loss

```
Binary Cross-Entropy:
L_recon = -Σ [x log(x̂) + (1-x) log(1-x̂)]

Measures: How well can we reconstruct input?
```

### 2. KL Divergence

```
KL = 0.5 × Σ [μ² + σ² - log(σ²) - 1]

Measures: How different is q(z|x) from N(0,1)?

Purpose:
✅ Regularize latent space
✅ Make it continuous
✅ Enable random sampling
```

## Why KL Divergence?

```
Without KL:
• Encoder learns arbitrary distributions
• Latent space fragmented
• Can't sample randomly

With KL:
• Forces q(z|x) close to N(0,1)
• Creates smooth latent space
• Random z ~ N(0,1) works!
```

## Generation Methods

### 1. Random Sampling

```rust
// Sample from standard normal
z = random_normal(mean=0, std=1)

// Decode
x_new = decoder(z)

Result: New, realistic sample!
```

### 2. Latent Interpolation

```rust
// Encode two images
z1 = encoder(img1).mu  // "3"
z2 = encoder(img2).mu  // "8"

// Interpolate
for alpha in 0..1:
    z = alpha * z1 + (1-alpha) * z2
    img = decoder(z)

Result: Smooth morphing from 3 to 8!
```

### 3. Latent Arithmetic

```rust
// Similar to word2vec: king - man + woman = queen

z_smile = encoder(smiling_face)
z_neutral = encoder(neutral_face)

smile_vector = z_smile - z_neutral

// Add smile to any face
z_new = encoder(any_face) + smile_vector
result = decoder(z_new)  // Now smiling!
```

## β-VAE Variant

```
L = L_recon + β × KL

β = 1:  Standard VAE
β > 1:  β-VAE (more disentanglement)
β < 1:  Emphasize reconstruction

β = 2-4: Often gives better disentangled features
Example: Separate dims for rotation vs thickness
```

## Applications

### 1. Image Generation

```
MNIST: Generate new digits
CelebA: Generate new faces
```

### 2. Anomaly Detection

```
Train on normal data only

At test:
error = ||x - reconstruct(x)||²

if error > threshold:
    → Anomaly!

Uses:
• Manufacturing defects
• Network intrusion
• Medical abnormalities
```

### 3. Data Compression

```
Compress: 784 dims → 10 dims (encoder)
Decompress: 10 dims → 784 dims (decoder)

Better than JPEG for domain-specific data
```

### 4. Drug Discovery

```
Molecular VAE:
• Encode molecules → latent space
• Navigate latent space
• Decode → new molecules
• Test promising candidates
```

## VAE vs Other Models

### VAE vs GAN

| Feature | VAE | GAN |
|---------|-----|-----|
| Training | Stable | Unstable |
| Output quality | Blurry | Sharp |
| Latent space | Meaningful | Less structured |
| Encoding | Yes | No |
| Small data | Works well | Struggles |

**Use VAE for:**
- Stable training
- Encoding/decoding
- Anomaly detection

**Use GAN for:**
- Highest quality generation
- Sharp images

### VAE vs Diffusion

| Feature | VAE | Diffusion |
|---------|-----|-----------|
| Speed | Fast (1 pass) | Slow (many steps) |
| Quality | Good | Excellent |
| Encoding | Yes | Difficult |
| Control | Good | Excellent |

**Note:** Stable Diffusion uses both!
- VAE encodes images to latent
- Diffusion works in latent space
- VAE decodes to images

## Training Configuration

```
Latent dimension: 10-512
  • 2: Visualizable
  • 10-20: MNIST
  • 128-512: Faces

Optimizer: Adam (lr=0.001)
Batch size: 32-128
Epochs: 50-200
β: 1.0 (or 2-4 for disentangling)
```

## Training Tips

### KL Annealing

```
Problem: KL dominates early → poor reconstruction

Solution: Gradually increase KL weight
L = L_recon + γ(epoch) × KL

γ schedule:
Epoch 0-10:  0.0 → 0.1
Epoch 10-50: 0.1 → 1.0
Epoch 50+:   1.0
```

### Free Bits

```
Problem: Some latent dims unused (posterior collapse)

Solution: Minimum KL per dimension
KL_per_dim = max(KL_dim, λ)

λ = 0.5 typical
```

## Advanced Variants

### Conditional VAE (CVAE)

```
Encoder: [x, label] → z
Decoder: [z, label] → x̂

Generation:
z = random()
label = 5  ← Control output
x = decoder([z, label])  # Generate a "5"
```

### VQ-VAE (Vector Quantized)

```
Discrete latent space:
z ∈ {e1, e2, ..., eK}  (codebook)

Used in:
• DALL-E
• High-quality image generation
```

### Hierarchical VAE

```
Multiple latent levels:
x → z1 → z2 → z3

Better for complex, multi-scale data
```

## Common Issues

### Blurry Outputs

```
Solutions:
✅ Increase latent dimension
✅ Reduce β
✅ More capacity (wider/deeper)
✅ Try perceptual loss
```

### Posterior Collapse

```
Symptom: KL → 0, latent unused

Solutions:
✅ KL annealing
✅ Free bits
✅ Reduce decoder capacity
```

## Historical Impact

**2013:** VAE introduced (Kingma & Welling)
- Probabilistic deep generative model
- Reparameterization trick

**2015-2017:** Extensions
- β-VAE, CVAE, Hierarchical VAE

**2018-2019:** Applied widely
- VQ-VAE for images
- Molecular VAE
- Text generation

**2020+:** Component in modern systems
- Stable Diffusion (VAE encoder/decoder)
- DALL-E (VQ-VAE)
- Still relevant for anomaly detection

## Further Reading

- [VAE paper](https://arxiv.org/abs/1312.6114) (Kingma & Welling, 2013)
- [β-VAE](https://openreview.net/forum?id=Sy2fzU9gl) (Higgins et al., 2017)
- [VQ-VAE](https://arxiv.org/abs/1711.00937) (van den Oord et al., 2017)
- [Tutorial on VAEs](https://arxiv.org/abs/1606.05908) (Doersch, 2016)

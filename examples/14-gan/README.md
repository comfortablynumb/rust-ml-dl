# Generative Adversarial Network (GAN) Example

This example demonstrates GANs, revolutionary neural networks that generate realistic data through adversarial training.

## Overview

GANs consist of two networks competing against each other:
- **Generator**: Creates fake data to fool the discriminator
- **Discriminator**: Tries to distinguish real data from fake data

Through this competition, the generator learns to create increasingly realistic data.

## Running the Example

```bash
cargo run --package gan
```

## The Adversarial Game

Think of it like a counterfeiter vs detective:

**Generator (Counterfeiter):**
- Starts creating terrible fakes
- Learns from discriminator's feedback
- Eventually creates perfect fakes

**Discriminator (Detective):**
- Starts easily spotting fakes
- Gets harder as generator improves
- Eventually can only guess (50% accuracy)

## Architecture

### Generator
```
Random Noise (100D)
    ↓
Dense(256) + ReLU
    ↓
Dense(512) + ReLU
    ↓
Dense(784) + Tanh
    ↓
Generated Image (28×28)
```

### Discriminator
```
Image (784)
    ↓
Dense(512) + LeakyReLU
    ↓
Dense(256) + LeakyReLU
    ↓
Dense(1) + Sigmoid
    ↓
Probability [0=Fake, 1=Real]
```

## Training Process

For each iteration:

1. **Train Discriminator:**
   - Show it real data (label: 1)
   - Show it fake data from generator (label: 0)
   - Update weights to classify better

2. **Train Generator:**
   - Generate fake data
   - Try to fool discriminator
   - Update weights to create more realistic fakes

## Loss Functions

**Discriminator Loss:**
```
L_D = -[log(D(x_real)) + log(1 - D(G(z)))]
```

**Generator Loss (Non-Saturating):**
```
L_G = -log(D(G(z)))
```

## Training Challenges

### Mode Collapse
**Problem:** Generator produces limited variety
- Example: Only generates one type of digit

**Solutions:**
- Feature matching
- Minibatch discrimination
- Use WGAN

### Vanishing Gradients
**Problem:** Discriminator too good, no learning signal
- Gradients to generator vanish

**Solutions:**
- Non-saturating loss
- Wasserstein loss (WGAN)

### Training Instability
**Problem:** Losses oscillate, don't converge

**Solutions:**
- Careful learning rate tuning
- Spectral normalization
- Train D multiple times per G update

## GAN Variants

### DCGAN (Deep Convolutional)
- Uses Conv layers instead of Dense
- BatchNorm for stability
- Best for image generation

### CGAN (Conditional)
```
Input: [noise, class_label]
Output: Image of specified class
```

### CycleGAN
- Learns mappings between domains
- Example: Horse ↔ Zebra, Photos ↔ Paintings
- No paired training data needed!

### StyleGAN
- Controls different aspects of generated images
- Powers ThisPersonDoesNotExist.com
- State-of-the-art face generation

### WGAN (Wasserstein)
- Uses Wasserstein distance
- More stable training
- Better convergence properties

### Progressive GAN
- Starts at low resolution (4×4)
- Progressively adds layers
- Achieves very high resolution (1024×1024)

## Applications

### Image Generation
- Generate realistic faces
- Create artwork
- Synthesize scenes and objects

### Image-to-Image Translation
- **Pix2Pix**: Sketch → Photo, Day → Night
- **CycleGAN**: Horse → Zebra, Summer → Winter
- **Super-Resolution**: Enhance image quality

### Data Augmentation
- Generate training data
- Balance imbalanced datasets
- Synthetic medical images

### Drug Discovery
- Generate novel molecules
- Optimize chemical properties
- Design new proteins

### Video & Audio
- Video prediction
- Voice conversion
- Music generation

### Controversial Uses
- DeepFakes (face swapping)
- Voice cloning
- Misinformation generation

## Evaluation Metrics

### Inception Score (IS)
- Measures quality and diversity
- Higher is better
- Range: [1, ∞)

### Fréchet Inception Distance (FID)
- Compares real vs generated distributions
- Lower is better (0 = perfect)
- Industry standard metric

### Precision & Recall
- **Precision**: Quality (fakes realistic?)
- **Recall**: Diversity (covers all modes?)

## Training Tips

1. **Normalize** inputs to [-1, 1], use Tanh output
2. **Use LeakyReLU** in discriminator (α=0.2)
3. **BatchNorm** in both networks
4. **Label smoothing**: Use 0.9 instead of 1.0 for real labels
5. **Train D more**: 1-5 discriminator updates per generator update
6. **Adam optimizer**: β1=0.5, β2=0.999, lr=0.0002
7. **Avoid max pooling**: Use stride for downsampling
8. **Monitor** both losses and generated samples regularly
9. **Spectral normalization** for stability
10. **Noise injection** for better diversity

## Modern Context

**GANs (2014-2020):**
- Pioneered realistic generation
- Fast sampling
- Difficult to train

**Diffusion Models (2020+):**
- Stable Diffusion, DALL-E 2/3
- More stable training
- Better quality
- Slower sampling

**Current Landscape:**
- Diffusion models dominate image generation
- GANs still used for real-time applications (video games, AR filters)
- Hybrid approaches combining both

## Famous GAN Applications

### ThisPersonDoesNotExist.com
- StyleGAN-generated faces
- Every refresh: new fake person

### DALL-E / Midjourney
- Text-to-image generation
- Uses diffusion (inspired by GANs)

### Artbreeder
- Genetic algorithms + GANs
- Collaborative image creation

### DeepFake Detection
- GANs used to detect GANs!
- Important for misinformation prevention

## Historical Impact

**2014:** Ian Goodfellow invents GANs
- "Most important idea in ML in 10 years" - Yann LeCun

**2015-2017:** Rapid improvements
- DCGAN, WGAN, Progressive GAN

**2018-2020:** State-of-the-art generation
- StyleGAN, BigGAN

**2020+:** Diffusion takes over
- But GANs remain important

## Further Reading

- [Original GAN Paper](https://arxiv.org/abs/1406.2661) (Goodfellow et al., 2014)
- [DCGAN](https://arxiv.org/abs/1511.06434)
- [StyleGAN](https://arxiv.org/abs/1812.04948)
- [GAN Hacks](https://github.com/soumith/ganhacks) - Practical tips
- [The GAN Zoo](https://github.com/hindupuravinash/the-gan-zoo) - 500+ GAN variants

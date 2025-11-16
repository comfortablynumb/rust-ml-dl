# Diffusion Models Example

This example explains **Diffusion Models**, the state-of-the-art generative models powering Stable Diffusion, DALL-E 2, Midjourney, and Imagen.

## Overview

Diffusion models generate images by learning to reverse a gradual noising process:
1. **Forward**: Clean image → Add noise step-by-step → Pure noise
2. **Reverse**: Pure noise → Denoise step-by-step → Clean image

## Running the Example

```bash
cargo run --package diffusion
```

## Core Concept

### The Diffusion Process

```
Original Image
     ↓ Add noise
Slightly noisy
     ↓ Add noise
More noisy
     ...
Pure noise (Gaussian)

Generation = Reverse this process!
```

## Two Processes

### 1. Forward Diffusion (Noising)

Gradually destroy image structure:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)

Example with T=1000 steps:
x_0: Clean cat photo
x_100: Slightly noisy cat
x_500: Very blurry cat
x_1000: Pure noise (no cat visible)

Key property: Can jump to any step directly!
x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
```

### 2. Reverse Process (Denoising)

Learn to remove noise:

```
p_θ(x_{t-1} | x_t)

Neural network predicts:
Given noisy image at step t → Less noisy at step t-1

Parameterization: Predict noise ε
ε_pred = ε_θ(x_t, t)
```

## Training

```
Algorithm (DDPM):
1. Sample image x_0 from dataset
2. Sample timestep t ~ Uniform(1, 1000)
3. Sample noise ε ~ N(0,1)
4. Create noisy image: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
5. Predict noise: ε_pred = network(x_t, t)
6. Loss: ||ε - ε_pred||²

Train network to predict what noise was added!
```

## Generation (Sampling)

```
Start: x_T ~ N(0,1) (pure noise)

For t = T down to 1:
    ε_pred = network(x_t, t)
    x_{t-1} = denoise(x_t, ε_pred)

Result: x_0 (clean image)

Typical: 50-1000 steps
Speed: 2-5 seconds per image (GPU)
```

## Architecture

### U-Net with Time Embedding

```
Input: x_t (noisy image) + t (timestep)

Time embedding:
• Sinusoidal encoding of t
• Added to features at each layer

U-Net:
• Encoder: Downsample (256→128→64→32)
• Bottleneck: 32×32
• Decoder: Upsample (32→64→128→256)
• Skip connections

Output: Predicted noise ε
```

## Text-to-Image

### Conditioning on Text

```
Network: ε_θ(x_t, t, text)

Components:
1. Text encoder (CLIP):
   "A cat..." → text_embedding

2. Cross-attention in U-Net:
   Image features attend to text

3. Classifier-free guidance:
   ε = ε_uncond + w × (ε_cond - ε_uncond)

   w = 7.5 (typical)
   Higher w → stronger text match
```

## Latent Diffusion (Stable Diffusion)

**Work in compressed space:**

```
Problem: 512×512 pixels = slow

Solution: Latent diffusion
1. VAE encoder: 512×512 → 64×64 (8× smaller)
2. Diffusion in 64×64 space (64× faster!)
3. VAE decoder: 64×64 → 512×512

Components:
• CLIP text encoder
• VAE (compress/decompress)
• U-Net (diffusion)
• Scheduler (noise schedule)

Speed: ~3 seconds for 512×512 image
```

## Applications

### Text-to-Image

```
Prompt: "A photo of an astronaut riding a horse"
Output: Photorealistic image

Models:
• Stable Diffusion (open source)
• DALL-E 2 (OpenAI)
• Midjourney (artistic)
• Imagen (Google)
```

### Image Editing

```
Inpainting:
• Mask region + text prompt
• Fill in coherently

Image-to-Image:
• Start from sketch
• Add style with text
• Get photorealistic result
```

### Other Applications

```
• Super-resolution
• Video generation (Runway, Sora)
• Audio generation
• 3D model generation
• Drug discovery
```

## Why Diffusion Won

### vs GANs

| Feature | Diffusion | GAN |
|---------|-----------|-----|
| Quality | Excellent | Good |
| Diversity | High | Mode collapse risk |
| Training | Stable | Unstable |
| Speed | Slow (50 steps) | Fast (1 step) |

**Result:** Diffusion is now #1 for images (2022+)

### vs VAE

| Feature | Diffusion | VAE |
|---------|-----------|-----|
| Quality | Sharp | Blurry |
| Generation | Slow | Fast |
| Encoding | Hard | Easy |

**Hybrid:** Stable Diffusion uses both!
- VAE for compression
- Diffusion for generation

## Training Configuration

```
Dataset: 10M - 1B images (LAION-5B)
Steps: 1000 (training), 50 (inference)
Noise schedule: Cosine
Optimizer: AdamW (lr=1e-4)
Batch size: 64-256
Resolution: 512×512

Hardware:
• Training: 1000+ GPU-days
• Inference: 2-5 sec on GPU
```

## Advanced Techniques

### Classifier-Free Guidance

```
Train one model for both:
• Conditional: p(x|text)
• Unconditional: p(x) [drop text 10%]

At inference:
noise_pred = noise_uncond + scale × (noise_cond - noise_uncond)

Scale 7.5 = balanced
Scale 15 = very strong text adherence
```

### Faster Sampling

```
DDIM: 50 steps instead of 1000
DPM++: Optimized sampler
Distillation: 1-4 steps (trained student)

Trade-off: Speed vs Quality
```

### ControlNet

```
Add spatial control:
• Edge maps
• Depth maps
• Pose estimation

Precise composition control!
```

## Practical Tips

### Prompt Engineering

```
Bad: "cat"

Good: "A professional photograph of a fluffy orange cat,
      sitting on a wooden table, soft lighting,
      highly detailed, 8k resolution"

Tips:
✅ Be specific
✅ Mention style
✅ Add quality terms
✅ Use negative prompts
```

### Sampling Settings

```
Steps: 20-50
• 20: Fast
• 50: Better quality

Guidance scale: 1-15
• 1: Ignore text, diverse
• 7.5: Balanced
• 15: Strong text, less diverse

Sampler: DDIM, DPM++, Euler
```

## Historical Timeline

**2020:** DDPM (Ho et al.)
- Made diffusion practical
- Beat GANs on some tasks

**2021:** Classifier Guidance
- Conditional generation
- Higher quality

**2022:** DALL-E 2
- Text-to-image breakthrough
- Photorealistic

**2022:** Stable Diffusion
- Open source!
- Latent diffusion
- Democratized AI art

**2023:** Midjourney v5
- Artistic quality
- Commercial success

**2024:** Sora
- Text-to-video
- 1-minute photorealistic videos

## Impact

```
Replaced GANs as #1 generative model
Enabled AI art revolution
Billion-dollar industry (Midjourney, Stability AI)
Creative tool for millions
```

## Further Reading

- [DDPM paper](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
- [Improved Diffusion](https://arxiv.org/abs/2102.09672) (Nichol & Dhariwal, 2021)
- [Latent Diffusion](https://arxiv.org/abs/2112.10752) (Rombach et al., 2022)
- [Stable Diffusion](https://stability.ai/stable-diffusion)
- [DALL-E 2](https://openai.com/dall-e-2)

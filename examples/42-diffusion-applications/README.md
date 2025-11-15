# Diffusion Model Applications 🎨

**Practical applications of diffusion models**: Image editing, inpainting, text-to-image generation, and classifier-free guidance. Powers Stable Diffusion, DALL-E 2, and Midjourney.

## Overview

Diffusion models aren't just for generation - they're incredibly versatile! This example explores the practical applications that have created the AI art revolution:
- **Text-to-Image**: "A cat riding a bicycle" → realistic image
- **Image Editing**: Modify existing images with text prompts
- **Inpainting**: Fill in missing regions intelligently
- **Classifier-Free Guidance**: Control generation strength

## Foundation: Diffusion Process Recap

### Forward Process (Add Noise)
```
x₀ (real image) → x₁ → x₂ → ... → xₜ → ... → xₜ (pure noise)

xₜ = √(α_t) x₀ + √(1-α_t) ε
```

### Reverse Process (Denoise)
```
xₜ (noise) → xₜ₋₁ → ... → x₁ → x₀ (clean image)

Predict noise: ε̂ = model(xₜ, t)
Denoise: xₜ₋₁ = (xₜ - √(1-α_t) ε̂) / √(α_t) + noise
```

## Application 1: Text-to-Image Generation 🖼️

**The killer app** - what made diffusion models famous!

### Architecture (Stable Diffusion Style)

```
Text prompt: "A cat riding a bicycle in Paris"
     ↓
1. Text Encoder (CLIP)
   "A cat..." → text_embedding [77 × 768]

2. Latent Diffusion
   noise [4 × 64 × 64] → denoise with text conditioning
   ↓ (guided by text_embedding)
   latent [4 × 64 × 64]

3. VAE Decoder
   latent → image [3 × 512 × 512]
```

### Key Components

#### Text Encoder (CLIP)
```
Purpose: Convert text → embedding space
Model: CLIP text encoder (Transformer)
Output: 77 tokens × 768 dimensions

Examples:
  "A cat" → [0.23, -0.45, 0.67, ...]
  "A dog" → [0.21, -0.42, 0.71, ...] (similar!)
  "A car" → [-0.12, 0.34, -0.89, ...] (different)
```

#### Cross-Attention Conditioning
```
In each denoising step:

Q = query from noisy image features
K, V = keys, values from text embedding

Attention(Q, K, V) = softmax(QK^T / √d) V

Effect: Image generation attends to text
Result: Cat in generated image, not random object
```

#### Latent Diffusion (Stable Diffusion)
```
Problem: Diffusion on 512×512 RGB is slow
  - 3 × 512 × 512 = 786K pixels
  - 1000 denoising steps
  - Very expensive!

Solution: Work in compressed latent space
  - VAE: 512×512 → 64×64 latent (8× smaller each dim)
  - Diffusion: 4 × 64 × 64 = 16K values
  - 64× faster than pixel space!
  - VAE decode at end: latent → final image
```

### Training Data
```
Dataset: Millions of (image, caption) pairs
  - LAION-5B (5 billion image-text pairs)
  - Filtered for quality and safety

Training:
  1. Get (image, text) pair
  2. Encode image to latent with VAE
  3. Add noise to latent
  4. Predict noise conditioned on text
  5. Loss: MSE(noise_predicted, noise_actual)
```

### Generation Process
```
Input: "A cat riding a bicycle"

1. Encode text with CLIP
   text_emb = clip_encode("A cat riding a bicycle")

2. Start with random noise
   x_T ~ N(0, I)

3. Denoise step-by-step (T → 0)
   for t in reversed(range(T)):
       ε̂ = unet(x_t, t, text_emb)  # Predict noise
       x_{t-1} = denoise_step(x_t, ε̂, t)

4. Decode latent to image
   image = vae_decode(x_0)

Output: 512×512 image of cat on bicycle!
```

## Application 2: Classifier-Free Guidance 🎯

**The secret sauce** that makes text-to-image work well!

### The Problem
```
Without guidance: Generated images are blurry/generic
With guidance: Sharp, prompt-aligned images
```

### How It Works

#### Unconditional vs Conditional
```
Unconditional: ε̂_uncond = model(x_t, t, ∅)
  - No text conditioning (empty prompt)
  - Generates generic images

Conditional: ε̂_cond = model(x_t, t, text)
  - With text conditioning
  - Tries to match prompt
```

#### Guidance Formula
```
ε̂_guided = ε̂_uncond + w × (ε̂_cond - ε̂_uncond)

w = guidance scale (typically 7.5)

Interpretation:
  - ε̂_uncond: What model generates naturally
  - ε̂_cond: What model generates with prompt
  - Difference: "Direction" toward prompt
  - w > 1: Amplify that direction (stronger prompt)
```

#### Effect of Guidance Scale
```
w = 1.0: Same as conditional (no amplification)
  Result: Follows prompt loosely, creative

w = 7.5: Strong guidance (Stable Diffusion default)
  Result: Closely follows prompt, realistic

w = 15.0: Very strong guidance
  Result: Very literal, may be oversaturated

w = 20.0+: Too strong
  Result: Artifacts, color issues
```

### Why It Works
```
Mathematics:
  Guided prediction = unconditional + amplified (conditional - unconditional)

Intuition:
  - Find direction from "random" to "prompted"
  - Move extra distance in that direction
  - Results in stronger prompt adherence
```

### Training for Classifier-Free Guidance
```
During training:
  - 10% of time: Train with empty text (unconditional)
  - 90% of time: Train with text caption (conditional)

Result:
  - Model learns both unconditional and conditional
  - Can do classifier-free guidance at inference
```

## Application 3: Image Editing ✏️

**Modify existing images** using text prompts.

### Method 1: SDEdit (Stochastic Differential Editing)

```
Input: Original image + text prompt

1. Encode image to latent
   z₀ = vae_encode(image)

2. Add noise (partial corruption)
   z_t = add_noise(z₀, noise_level=0.5)
   - noise_level ∈ [0, 1]
   - 0 = no noise (no change)
   - 1 = full noise (complete regeneration)

3. Denoise with new text prompt
   for t in reversed(range(t_start, 0)):
       ε̂ = unet(z_t, t, new_text)
       z_{t-1} = denoise_step(z_t, ε̂, t)

4. Decode to image
   edited_image = vae_decode(z_0)
```

**Noise Level Trade-off:**
```
Low noise (0.2): Minimal changes, preserves structure
Medium noise (0.5): Balanced edit
High noise (0.8): Large changes, may lose original
```

### Method 2: Instruct-Pix2Pix

```
Input: Image + edit instruction

Example:
  Image: Photo of sunny beach
  Instruction: "Make it sunset"
  Output: Same beach at sunset

Training:
  - Dataset: (before_image, instruction, after_image) triplets
  - Model learns to follow edit instructions
```

### Method 3: Prompt-to-Prompt

```
Idea: Edit specific parts by changing attention

Example:
  Original prompt: "A cat on a sofa"
  New prompt: "A dog on a sofa"

Keep attention maps for "on a sofa" (preserve layout)
Change attention for "cat" → "dog"

Result: Same composition, different subject
```

## Application 4: Image Inpainting 🖌️

**Fill in masked regions** intelligently.

### Standard Inpainting

```
Input:
  - Original image
  - Binary mask (1 = inpaint, 0 = keep)
  - Text prompt (what to fill with)

Process:
1. Encode image to latent
   z₀ = vae_encode(image)

2. Encode mask
   mask_latent = downsample(mask)

3. Add noise to FULL latent
   z_t = add_noise(z₀)

4. Denoise step with mask conditioning
   for t in reversed(range(T)):
       ε̂ = unet(z_t, mask_latent, t, text)
       z_{t-1} = denoise_step(z_t, ε̂, t)

       # Keep unmasked regions from original
       z_{t-1} = mask * z_{t-1} + (1-mask) * z₀_noised_to_t

5. Decode
   result = vae_decode(z_0)
```

### Blended Inpainting
```
Problem: Hard boundary between inpainted and original
Solution: Blend in latent space

Blending:
  for each pixel:
    distance_to_mask = distance from masked region
    blend_weight = smooth_transition(distance)
    result = blend_weight * generated + (1-blend_weight) * original
```

### Applications
```
- Remove objects: "grass" (to fill where object was)
- Add objects: "a red car" in empty space
- Outpainting: Extend image beyond borders
- Face restoration: Fix blurry/damaged faces
```

## Application 5: Image-to-Image Translation 🔄

### Depth-to-Image
```
Input: Depth map + prompt
Output: Realistic image matching depth

Use case: Control spatial layout precisely
Example: "A cyberpunk city" with specific depth
```

### Sketch-to-Image
```
Input: Simple sketch + prompt
Output: Detailed image

Process: Condition on edge maps or sketches
Use case: Turn simple drawings into art
```

### Pose-to-Image
```
Input: Human pose (skeleton) + prompt
Output: Person in that pose

Example: "A superhero in flying pose"
Use case: Character animation, fashion
```

## Application 6: Video Generation 🎬

### Temporal Consistency
```
Problem: Generate frames independently → flicker

Solution 1: Temporal attention layers
  - Attend across time dimension
  - Learns temporal coherence

Solution 2: Latent consistency
  - Reuse latent from previous frame
  - Only denoise partially for new frame
```

### Techniques
```
- AnimateDiff: Add temporal layers to Stable Diffusion
- Text-to-Video: "A cat jumping" → short video
- Video editing: Edit existing videos with prompts
```

## Real-World Systems

### Stable Diffusion
```
Architecture:
  - CLIP text encoder
  - VAE (encoder/decoder)
  - U-Net with cross-attention
  - Latent diffusion (64×64)

Capabilities:
  - Text-to-image (512×512, 768×768)
  - Image editing (img2img)
  - Inpainting
  - ControlNet (depth, pose, sketch)

Model sizes:
  - SD 1.5: 860M parameters
  - SD 2.1: 865M parameters
  - SDXL: 2.6B parameters (better quality)
```

### DALL-E 2 (OpenAI)
```
Architecture:
  - CLIP (vision-language)
  - Diffusion prior (text → image prior)
  - Diffusion decoder (prior → image)

Capabilities:
  - Text-to-image
  - Inpainting
  - Variations (similar images)

Quality: Photorealistic, coherent
```

### Midjourney
```
Architecture: Proprietary (likely diffusion-based)

Strengths:
  - Artistic style
  - Coherent compositions
  - V5: Photorealistic

Use case: Artists, designers, creative work
```

### Adobe Firefly
```
Built on diffusion models
Integration: Photoshop, Illustrator
Features: Commercial safe, trained on licensed data
```

## Advanced Techniques

### ControlNet
```
Problem: Hard to control spatial layout with text

Solution: Additional conditioning inputs
  - Depth maps
  - Edge maps (Canny)
  - Pose (human skeleton)
  - Segmentation maps

Result: Precise control over composition
```

### LoRA (Low-Rank Adaptation)
```
Problem: Fine-tuning full model is expensive

Solution: Train small adapter layers
  - Only 1-10MB per style (vs 2GB full model)
  - Fast training (minutes on GPU)
  - Composable (combine multiple LoRAs)

Use case:
  - Custom styles
  - Specific characters/objects
  - Personal fine-tuning
```

### Textual Inversion
```
Learn new "words" for concepts

Example:
  - Train on 3-5 images of your cat
  - Creates embedding for "<my-cat>"
  - Prompt: "A photo of <my-cat> at the beach"
  - Result: Your cat, new scene!
```

## Practical Considerations

### Inference Speed
```
Standard: 50 steps, 3-5 seconds (GPU)
Fast samplers:
  - DPM-Solver: 20 steps, same quality
  - DDIM: 25 steps
  - LCM (Latent Consistency): 4 steps!

Speed vs Quality trade-off
```

### Memory Requirements
```
Stable Diffusion 1.5:
  - FP32: 8GB VRAM
  - FP16: 4GB VRAM
  - INT8 quantized: 2GB VRAM

SDXL:
  - FP16: 8-12GB VRAM
```

### Prompt Engineering
```
Effective prompts:
  - Specific details: "oil painting", "8k resolution"
  - Artist styles: "in the style of Van Gogh"
  - Negative prompts: Avoid unwanted elements
  - Emphasis: (word:1.2) for stronger influence

Example:
  Positive: "A majestic lion, golden hour lighting, photorealistic, 8k, detailed fur"
  Negative: "blurry, low quality, cartoon, distorted"
```

## Key Takeaways

1. **Latent diffusion** makes generation practical (64× faster)
2. **Classifier-free guidance** (w=7.5) is essential for quality
3. **CLIP** enables text-to-image alignment
4. **Inpainting** and editing extend beyond pure generation
5. **ControlNet** enables precise spatial control

## Running the Example

```bash
cargo run --package diffusion-applications
```

Demonstrates:
- Text conditioning simulation
- Classifier-free guidance
- Inpainting mask processing
- Guidance scale effects

## References

- **Stable Diffusion:** Rombach et al. (2022) - "High-Resolution Image Synthesis with Latent Diffusion Models"
- **DALL-E 2:** Ramesh et al. (2022) - "Hierarchical Text-Conditional Image Generation with CLIP Latents"
- **Classifier-Free Guidance:** Ho & Salimans (2022) - "Classifier-Free Diffusion Guidance"
- **ControlNet:** Zhang et al. (2023) - "Adding Conditional Control to Text-to-Image Diffusion Models"
- **Midjourney:** Proprietary, community-driven development

## Impact

Diffusion model applications have:
- ✅ **Democratized AI art** (anyone can create)
- ✅ **Billion-dollar industry** (Midjourney profitable)
- ✅ **Integrated into tools** (Photoshop, Canva)
- ✅ **New workflows** (concept art, design)
- ✅ **Accessibility** (text → image, no art skills needed)

**The AI art revolution - powered by diffusion models!**

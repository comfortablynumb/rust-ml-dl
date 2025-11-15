/// # Diffusion Model Applications 🎨
///
/// Practical applications of diffusion models: text-to-image generation, image editing,
/// inpainting, and classifier-free guidance. Powers Stable Diffusion, DALL-E, Midjourney.
///
/// ## What This Example Demonstrates
///
/// 1. **Text Conditioning**: How text guides image generation
/// 2. **Classifier-Free Guidance**: Control generation strength (w=7.5)
/// 3. **Inpainting**: Fill masked regions intelligently
/// 4. **Image Editing**: Modify existing images with text
///
/// ## Why These Applications Matter
///
/// - **Stable Diffusion**: Billion-user platform
/// - **DALL-E 2**: Democratized AI art
/// - **Midjourney**: Profitable AI art company
/// - **Impact**: $1B+ industry, integrated into Photoshop, Canva
///
/// ## The Revolution
///
/// ```
/// Before 2022: AI art required technical expertise
/// After 2022: "A cat on Mars" → photorealistic image (anyone can create)
/// ```

use ndarray::{Array1, Array2};
use rand::Rng;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         Diffusion Model Applications                     ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Demonstrate text conditioning
    demo_text_conditioning();

    // Demonstrate classifier-free guidance
    demo_classifier_free_guidance();

    // Demonstrate inpainting
    demo_inpainting();
}

/// Demonstrate text conditioning
fn demo_text_conditioning() {
    println!("═══ Text-to-Image: Text Conditioning ═══\n");

    // Simulate text prompts and their embeddings
    let prompts = vec![
        "A cat",
        "A dog",
        "A car",
        "A cat on Mars",
    ];

    println!("How text guides image generation:\n");
    println!("Text prompt embeddings (simulated CLIP encoding):\n");

    let mut embeddings = Vec::new();
    for prompt in &prompts {
        let emb = encode_text_simple(prompt);
        print!("{:20} → ", prompt);
        print_embedding_preview(&emb);
        embeddings.push(emb);
    }

    // Show similarity between prompts
    println!("\nCosine similarity between prompts:");
    println!("──────────────────────────────────────────────");
    for i in 0..prompts.len() {
        for j in i+1..prompts.len() {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            println!("{:20} ↔ {:20} = {:.3}",
                     prompts[i], prompts[j], sim);
        }
    }

    println!("\n💡 Key Insight:");
    println!("   'A cat' and 'A cat on Mars' are similar (both have cat)");
    println!("   'A cat' and 'A car' are different");
    println!("   This similarity guides the diffusion process!\n");
}

/// Simple text encoding (simulates CLIP)
fn encode_text_simple(text: &str) -> Array1<f32> {
    let mut rng = rand::thread_rng();
    let dim = 16;

    // Simple heuristic encoding (real CLIP uses Transformer)
    let mut embedding = Array1::zeros(dim);

    // Hash-based encoding for demo purposes
    for (i, ch) in text.chars().enumerate() {
        let idx = ((ch as u32 + i as u32) % dim as u32) as usize;
        embedding[idx] += 0.5;
    }

    // Add some randomness
    for i in 0..dim {
        embedding[i] += rng.gen_range(-0.1..0.1);
    }

    // Normalize
    let norm: f32 = embedding.iter().map(|x| x*x).sum::<f32>().sqrt();
    embedding / (norm + 1e-8)
}

fn print_embedding_preview(emb: &Array1<f32>) {
    print!("[");
    for i in 0..4.min(emb.len()) {
        print!("{:6.3}", emb[i]);
        if i < 3 { print!(", "); }
    }
    println!(" ...]");
}

fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x*x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x*x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b + 1e-8)
}

/// Demonstrate classifier-free guidance
fn demo_classifier_free_guidance() {
    println!("═══ Classifier-Free Guidance (CFG) ═══\n");

    println!("The secret sauce for high-quality text-to-image!\n");

    // Simulate noise predictions
    let unconditional_noise = Array1::from(vec![0.1, 0.3, 0.2, 0.4]);
    let conditional_noise = Array1::from(vec![0.2, 0.5, 0.3, 0.6]);

    println!("Noise predictions:");
    println!("  ε_uncond (no text):   {:?}", unconditional_noise.to_vec());
    println!("  ε_cond (with text):   {:?}\n", conditional_noise.to_vec());

    // Different guidance scales
    println!("Guided predictions for different scales:\n");
    println!("Scale    Guided Prediction                     Effect");
    println!("───────────────────────────────────────────────────────────────");

    for &w in &[1.0, 3.0, 7.5, 12.0, 20.0] {
        let guided = classifier_free_guidance(
            &unconditional_noise,
            &conditional_noise,
            w
        );

        let effect = match w {
            w if w <= 1.0 => "Same as conditional (no boost)",
            w if w <= 5.0 => "Moderate guidance",
            w if w <= 10.0 => "Strong guidance (default)",
            w if w <= 15.0 => "Very strong guidance",
            _ => "Too strong (artifacts)",
        };

        print!("{:5.1}   ", w);
        print_array_short(&guided);
        println!("   {}", effect);
    }

    println!("\n💡 Classifier-Free Guidance Formula:");
    println!("   ε̂_guided = ε̂_uncond + w × (ε̂_cond - ε̂_uncond)");
    println!("\n   w=1.0:  No amplification (creative, loose)");
    println!("   w=7.5:  Stable Diffusion default (balanced)");
    println!("   w=15.0: Very literal (oversaturated)");
    println!("   w=20.0: Artifacts, color issues\n");
}

fn classifier_free_guidance(
    uncond: &Array1<f32>,
    cond: &Array1<f32>,
    guidance_scale: f32
) -> Array1<f32> {
    // ε̂_guided = ε̂_uncond + w × (ε̂_cond - ε̂_uncond)
    uncond + guidance_scale * (cond - uncond)
}

fn print_array_short(arr: &Array1<f32>) {
    print!("[");
    for i in 0..4.min(arr.len()) {
        print!("{:.2}", arr[i]);
        if i < 3 { print!(", "); }
    }
    print!("]");
}

/// Demonstrate inpainting
fn demo_inpainting() {
    println!("═══ Image Inpainting (Fill Masked Regions) ═══\n");

    // Simulate a small image (8×8 for demo)
    let size = 8;
    let image = Array2::from_shape_fn((size, size), |(i, j)| {
        // Create simple pattern
        if (i + j) % 2 == 0 { 1.0 } else { 0.0 }
    });

    // Create mask (1 = inpaint, 0 = keep)
    let mut mask = Array2::zeros((size, size));
    // Mask center region
    for i in 2..6 {
        for j in 2..6 {
            mask[[i, j]] = 1.0;
        }
    }

    println!("Original image (8×8 checkerboard pattern):");
    print_image(&image);

    println!("\nMask (█ = inpaint this region, · = keep original):");
    print_mask(&mask);

    // Simulate inpainting
    let prompt = "smooth gradient";
    println!("\nInpainting with prompt: \"{}\"\n", prompt);

    // Simple inpainting simulation
    let inpainted = simulate_inpainting(&image, &mask, prompt);

    println!("Inpainted result:");
    print_image(&inpainted);

    println!("\n💡 Inpainting Process:");
    println!("   1. Encode image to latent space (VAE)");
    println!("   2. Add noise to FULL image");
    println!("   3. Denoise with text conditioning");
    println!("   4. Keep unmasked regions from original each step");
    println!("   5. Decode to image (VAE)\n");

    println!("📊 Applications:");
    println!("   - Object removal: Remove unwanted objects");
    println!("   - Object addition: Add new elements");
    println!("   - Outpainting: Extend image beyond borders");
    println!("   - Face restoration: Fix blurry/damaged faces\n");
}

fn print_image(img: &Array2<f32>) {
    println!("  ┌{}┐", "─".repeat(img.shape()[1]));
    for i in 0..img.shape()[0] {
        print!("  │");
        for j in 0..img.shape()[1] {
            if img[[i, j]] > 0.5 {
                print!("█");
            } else {
                print!("·");
            }
        }
        println!("│");
    }
    println!("  └{}┘", "─".repeat(img.shape()[1]));
}

fn print_mask(mask: &Array2<f32>) {
    println!("  ┌{}┐", "─".repeat(mask.shape()[1]));
    for i in 0..mask.shape()[0] {
        print!("  │");
        for j in 0..mask.shape()[1] {
            if mask[[i, j]] > 0.5 {
                print!("█"); // Inpaint
            } else {
                print!("·"); // Keep
            }
        }
        println!("│");
    }
    println!("  └{}┘", "─".repeat(mask.shape()[1]));
}

fn simulate_inpainting(
    image: &Array2<f32>,
    mask: &Array2<f32>,
    _prompt: &str
) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    let mut result = image.clone();

    // Simple simulation: fill masked region with gradient
    let rows = image.shape()[0];
    let cols = image.shape()[1];

    for i in 0..rows {
        for j in 0..cols {
            if mask[[i, j]] > 0.5 {
                // Generate smooth gradient
                let value = (i as f32 + j as f32) / ((rows + cols) as f32);
                // Add slight noise
                result[[i, j]] = value + rng.gen_range(-0.05..0.05);
            }
        }
    }

    result
}

/// Key Concepts Summary
///
/// **Text-to-Image Pipeline (Stable Diffusion):**
/// ```
/// 1. Text Encoder (CLIP):
///    "A cat on Mars" → embedding [77 × 768]
///
/// 2. Latent Diffusion:
///    noise [4×64×64] + text_embedding → latent [4×64×64]
///    - Cross-attention conditioning
///    - U-Net denoising
///    - 1000 steps (or 20-50 with fast samplers)
///
/// 3. VAE Decoder:
///    latent [4×64×64] → image [3×512×512]
///
/// Result: Generated image matching text prompt
/// ```
///
/// **Classifier-Free Guidance:**
/// ```
/// ε̂_guided = ε̂_uncond + w × (ε̂_cond - ε̂_uncond)
///
/// w = guidance scale:
///   - w=1.0: No guidance (creative, loose)
///   - w=7.5: Stable Diffusion default (balanced)
///   - w=15.0: Very literal (oversaturated)
///
/// Why it works:
///   - Amplifies "direction" toward text prompt
///   - Stronger prompt adherence
///   - Essential for high-quality generation
/// ```
///
/// **Latent Diffusion (Efficiency):**
/// ```
/// Problem: 512×512 RGB = 786K pixels (slow!)
/// Solution: VAE compression
///   - Encode: 512×512 → 64×64 latent (8× each dim)
///   - Diffusion: Work in 64×64 latent space
///   - Decode: 64×64 → 512×512 final image
///
/// Speedup: 64× faster than pixel-space diffusion
/// Memory: 64× less than pixel-space
/// ```
///
/// **Inpainting Process:**
/// ```
/// 1. Input: image + mask + text prompt
/// 2. VAE encode to latent
/// 3. Add noise to FULL latent
/// 4. Denoise step-by-step:
///    - Predict noise conditioned on text
///    - Denoise
///    - Keep unmasked regions from original
/// 5. VAE decode to final image
///
/// Applications:
///   - Object removal
///   - Object addition
///   - Outpainting (extend borders)
///   - Face restoration
/// ```
///
/// **Real-World Systems:**
/// - **Stable Diffusion**: Open-source, 860M params, runs on consumer GPUs
/// - **DALL-E 2**: OpenAI, photorealistic quality
/// - **Midjourney**: Artistic style, $1B company
/// - **Adobe Firefly**: Commercial-safe, integrated in Photoshop
///
/// **Advanced Techniques:**
/// - **ControlNet**: Spatial control (depth, pose, edges)
/// - **LoRA**: Fast fine-tuning (1-10MB adapters)
/// - **Textual Inversion**: Learn new concepts
/// - **DPM-Solver**: Fast sampling (20 steps vs 50)
///
/// **Prompt Engineering:**
/// ```
/// Good prompt:
///   "A majestic lion, golden hour lighting,
///    photorealistic, 8k, detailed fur"
///
/// Negative prompt:
///   "blurry, low quality, cartoon, distorted"
///
/// Result: High-quality, controlled generation
/// ```
///
/// **Impact:**
/// The AI art revolution:
/// - Billion-user platforms (Stable Diffusion, Midjourney)
/// - Democratized creativity (no art skills needed)
/// - Integrated into tools (Photoshop, Canva, Figma)
/// - New industries (AI art, concept design)
/// - Accessible to everyone (text → image)
#[allow(dead_code)]
fn _summary() {}

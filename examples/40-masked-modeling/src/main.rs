/// # Masked Modeling 🎭
///
/// Self-supervised learning through masked prediction: Learn by reconstructing
/// masked portions of input. Foundation of BERT, GPT, and MAE.
///
/// ## What This Example Demonstrates
///
/// 1. **BERT-style Masking**: Random token masking for language
/// 2. **MAE-style Masking**: High-ratio patch masking for vision
/// 3. **Reconstruction Loss**: Training signal from masked predictions
/// 4. **Masking Strategies**: Random, block, autoregressive
///
/// ## Why Masked Modeling Matters
///
/// - **BERT**: Revolutionized NLP (Google Search uses it)
/// - **GPT-3 → ChatGPT**: 175B params, autoregressive masking
/// - **MAE**: SOTA vision pre-training (75% masking!)
/// - **No labels needed**: Train on billions of unlabeled examples
///
/// ## The Revolution
///
/// ```
/// Pre-BERT (2017): Task-specific models, limited transfer
/// Post-BERT (2018): Pre-train once → fine-tune everywhere
/// Impact: 10-100× less labeled data needed
/// ```

use ndarray::{Array1, Array2};
use rand::Rng;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         Masked Modeling (Self-Supervised)                ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Demonstrate BERT-style masked language modeling
    demo_masked_language_modeling();

    // Demonstrate MAE-style masked autoencoding
    demo_masked_autoencoder();

    // Demonstrate different masking strategies
    demo_masking_strategies();
}

/// Demonstrate BERT-style masked language modeling
fn demo_masked_language_modeling() {
    println!("═══ BERT-Style Masked Language Modeling ═══\n");

    // Simulate a sentence as token IDs
    let sentence = "The quick brown fox jumps over the lazy dog";
    let tokens: Vec<&str> = sentence.split_whitespace().collect();

    println!("Original sentence:");
    println!("  \"{}\"", sentence);
    println!("  Tokens: {:?}\n", tokens);

    // BERT masking: 15% of tokens
    let mask_ratio = 0.15;
    let (masked_tokens, mask_positions, masked_values) =
        bert_masking(&tokens, mask_ratio);

    println!("Masked sentence (15% masking):");
    print!("  \"");
    for token in &masked_tokens {
        print!("{} ", token);
    }
    println!("\"");

    println!("\nMasked positions: {:?}", mask_positions);
    println!("Original values:  {:?}", masked_values);

    // Simulate predictions
    println!("\n💡 BERT Training:");
    println!("   1. Input: Masked sentence");
    println!("   2. Model: Bidirectional Transformer");
    println!("   3. Output: Predictions for [MASK] tokens");
    println!("   4. Loss: CrossEntropy on masked positions only");

    println!("\n🎯 Key Insight: Bidirectional context");
    println!("   To predict [MASK], model sees:");
    println!("   - Left: \"The [MASK] brown\"");
    println!("   - Right: \"brown fox jumps\"");
    println!("   - Both sides → better understanding!\n");
}

/// BERT-style masking strategy
fn bert_masking<'a>(
    tokens: &[&'a str],
    mask_ratio: f32,
) -> (Vec<String>, Vec<usize>, Vec<String>) {
    let mut rng = rand::thread_rng();
    let mut masked_tokens = Vec::new();
    let mut mask_positions = Vec::new();
    let mut masked_values = Vec::new();

    for (i, &token) in tokens.iter().enumerate() {
        if rng.gen::<f32>() < mask_ratio {
            // This token is selected for masking
            mask_positions.push(i);
            masked_values.push(token.to_string());

            // BERT's masking strategy:
            let rand_val: f32 = rng.gen();
            if rand_val < 0.8 {
                // 80%: Replace with [MASK]
                masked_tokens.push("[MASK]".to_string());
            } else if rand_val < 0.9 {
                // 10%: Replace with random token
                masked_tokens.push("[RANDOM]".to_string());
            } else {
                // 10%: Keep original
                masked_tokens.push(token.to_string());
            }
        } else {
            masked_tokens.push(token.to_string());
        }
    }

    (masked_tokens, mask_positions, masked_values)
}

/// Demonstrate MAE-style masked autoencoding for vision
fn demo_masked_autoencoder() {
    println!("═══ MAE-Style Masked Autoencoder (Vision) ═══\n");

    // Simulate image as patches (14×14 grid for 224×224 image)
    let image_size = 224;
    let patch_size = 16;
    let num_patches = (image_size / patch_size) * (image_size / patch_size);

    println!("Image: {}×{} pixels", image_size, image_size);
    println!("Patch size: {}×{}", patch_size, patch_size);
    println!("Total patches: {} ({}×{})\n", num_patches, 14, 14);

    // MAE uses 75% masking (very high!)
    let mask_ratio = 0.75;
    let num_visible = (num_patches as f32 * (1.0 - mask_ratio)) as usize;
    let num_masked = num_patches - num_visible;

    println!("Masking ratio: {}%", (mask_ratio * 100.0) as i32);
    println!("Visible patches: {} ({}%)", num_visible,
             ((num_visible as f32 / num_patches as f32) * 100.0) as i32);
    println!("Masked patches: {} ({}%)\n", num_masked,
             ((num_masked as f32 / num_patches as f32) * 100.0) as i32);

    // Generate mask
    let mask = generate_random_mask(num_patches, mask_ratio);

    println!("Mask pattern (14×14, 0=visible, 1=masked):");
    print_mask_grid(&mask, 14);

    println!("\n💡 MAE Architecture:");
    println!("   Encoder (heavy, ViT):");
    println!("     - Input: {} visible patches only", num_visible);
    println!("     - 24 Transformer layers");
    println!("     - No mask tokens in encoder (efficient!)");
    println!("\n   Decoder (lightweight):");
    println!("     - Input: Encoded patches + mask tokens");
    println!("     - 8 Transformer layers");
    println!("     - Output: {} patches (reconstruct all)", num_patches);

    println!("\n🎯 Why 75% Masking?");
    println!("   - Low (15%): Too easy, trivial interpolation");
    println!("   - High (75%): Forces semantic understanding");
    println!("   - Images have redundancy → need aggressive masking\n");
}

/// Generate random mask
fn generate_random_mask(num_patches: usize, mask_ratio: f32) -> Vec<bool> {
    let mut rng = rand::thread_rng();
    let mut mask = vec![false; num_patches];

    let num_masked = (num_patches as f32 * mask_ratio) as usize;
    let mut masked_count = 0;

    // Random shuffle approach
    let mut indices: Vec<usize> = (0..num_patches).collect();
    for i in (1..indices.len()).rev() {
        let j = rng.gen_range(0..=i);
        indices.swap(i, j);
    }

    // Mask first num_masked indices
    for i in 0..num_masked {
        mask[indices[i]] = true;
        masked_count += 1;
    }

    mask
}

/// Print mask as 2D grid
fn print_mask_grid(mask: &[bool], width: usize) {
    for row in 0..(mask.len() / width) {
        print!("  ");
        for col in 0..width {
            let idx = row * width + col;
            if mask[idx] {
                print!("██"); // Masked (solid block)
            } else {
                print!("  "); // Visible (space)
            }
        }
        println!();
    }
}

/// Demonstrate different masking strategies
fn demo_masking_strategies() {
    println!("═══ Masking Strategies Comparison ═══\n");

    let sequence_len = 20;

    println!("Sequence length: {} tokens/patches\n", sequence_len);

    // 1. Random masking (BERT, MAE)
    println!("1. Random Masking (BERT, MAE):");
    let random_mask = generate_random_mask(sequence_len, 0.5);
    print!("   ");
    print_mask_sequence(&random_mask);
    println!("   - Pros: Simple, unbiased");
    println!("   - Cons: No spatial structure\n");

    // 2. Block masking
    println!("2. Block Masking:");
    let block_mask = generate_block_mask(sequence_len, 0.5, 4);
    print!("   ");
    print_mask_sequence(&block_mask);
    println!("   - Pros: Forces understanding of larger context");
    println!("   - Cons: Can be too hard\n");

    // 3. Causal masking (GPT)
    println!("3. Causal/Autoregressive Masking (GPT):");
    let causal_mask = generate_causal_mask(sequence_len, 10);
    print!("   ");
    print_mask_sequence(&causal_mask);
    println!("   - Pros: Natural for generation");
    println!("   - Cons: Unidirectional only\n");

    println!("📊 Use Cases:");
    println!("   - BERT (NLP understanding): Random 15%");
    println!("   - GPT (Text generation): Causal (predict next)");
    println!("   - MAE (Vision pre-train): Random 75%");
    println!("   - BEiT (Vision): Block masking 40%\n");
}

fn print_mask_sequence(mask: &[bool]) {
    for &masked in mask {
        if masked {
            print!("█"); // Masked
        } else {
            print!("○"); // Visible
        }
    }
    println!();
}

/// Generate block masking pattern
fn generate_block_mask(length: usize, mask_ratio: f32, block_size: usize) -> Vec<bool> {
    let mut rng = rand::thread_rng();
    let mut mask = vec![false; length];

    let target_masked = (length as f32 * mask_ratio) as usize;
    let mut masked_count = 0;

    while masked_count < target_masked && masked_count + block_size <= length {
        let start = rng.gen_range(0..=(length - block_size));

        // Check if block already masked
        let already_masked = (start..start + block_size)
            .any(|i| mask[i]);

        if !already_masked {
            for i in start..start + block_size {
                mask[i] = true;
            }
            masked_count += block_size;
        }
    }

    mask
}

/// Generate causal masking pattern (for autoregressive models like GPT)
fn generate_causal_mask(length: usize, visible: usize) -> Vec<bool> {
    let mut mask = vec![false; length];
    // First 'visible' tokens are visible, rest are masked
    for i in visible..length {
        mask[i] = true;
    }
    mask
}

/// Key Concepts Summary
///
/// **Masked Language Modeling (BERT):**
/// ```
/// 1. Mask 15% of tokens randomly
/// 2. Predict masked tokens using bidirectional Transformer
/// 3. Loss: CrossEntropy on masked positions
/// 4. Masking strategy:
///    - 80%: [MASK] token
///    - 10%: Random token
///    - 10%: Original token
/// ```
///
/// **Masked Autoencoder (MAE):**
/// ```
/// 1. Divide image into patches (16×16)
/// 2. Mask 75% of patches
/// 3. Encoder: Process visible patches only (efficient!)
/// 4. Decoder: Reconstruct all patches
/// 5. Loss: MSE on masked patches
/// ```
///
/// **Autoregressive (GPT):**
/// ```
/// 1. Predict next token (causal masking)
/// 2. Can only see previous tokens
/// 3. Loss: CrossEntropy on next token
/// 4. Natural for generation tasks
/// ```
///
/// **Why Different Ratios?**
/// - NLP (BERT): 15% - text has low redundancy
/// - Vision (MAE): 75% - images have high redundancy
/// - Higher masking → harder task → deeper understanding
///
/// **Training Recipe:**
/// ```
/// BERT:
///   - Data: Wikipedia + Books (3.3B words)
///   - Batch: 256 sequences
///   - Steps: 1M
///   - Hardware: 64 TPUs, 4 days
///
/// MAE:
///   - Data: ImageNet (1.3M images, unlabeled)
///   - Batch: 4096
///   - Epochs: 800
///   - Hardware: 128 TPUs
/// ```
///
/// **Applications:**
/// 1. **Transfer Learning**: Pre-train → fine-tune
///    - BERT → Sentiment analysis (10× less data)
///    - MAE → Object detection, segmentation
///
/// 2. **Few-Shot Learning**:
///    - Pre-train on millions unlabeled
///    - Fine-tune on 100s labeled
///    - 80%+ accuracy
///
/// 3. **Zero-Shot (GPT-3)**:
///    - No fine-tuning
///    - Just prompt engineering
///    - "Translate: Hello →" → "Bonjour"
///
/// **Famous Models:**
/// - **BERT** (110M): NLP revolution, Google Search
/// - **RoBERTa** (355M): Improved BERT
/// - **GPT-3** (175B): ChatGPT foundation
/// - **MAE**: Vision Transformer pre-training
///
/// **Results:**
/// - BERT: 80.5% GLUE (vs 72.8% pre-BERT)
/// - MAE: 87.8% ImageNet (SOTA for ViT)
/// - GPT-3: Human-level text generation
///
/// **Impact:**
/// The paradigm shift in AI:
/// ```
/// Old: Train task-specific model from scratch
/// New: Pre-train with masking → fine-tune for task
/// Benefit: 10-100× less labeled data needed
/// ```
#[allow(dead_code)]
fn _summary() {}

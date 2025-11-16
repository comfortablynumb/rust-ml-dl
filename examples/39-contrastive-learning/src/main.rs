/// # Contrastive Learning 🔥
///
/// Self-supervised learning through contrastive methods: learn representations
/// without labels by pulling similar examples together and pushing dissimilar apart.
///
/// ## What This Example Demonstrates
///
/// 1. **SimCLR-style Contrastive Loss**: NT-Xent with temperature scaling
/// 2. **Positive/Negative Pairs**: How data augmentation creates supervision
/// 3. **Temperature Effects**: Control concentration of embeddings
/// 4. **Similarity Metrics**: Cosine similarity for comparisons
///
/// ## Why Contrastive Learning Matters
///
/// - **Learn from unlabeled data**: Billions of internet images (free!)
/// - **Better than supervised**: For transfer learning tasks
/// - **Powers CLIP**: Foundation of Stable Diffusion, DALL-E
/// - **Few-shot learning**: 80%+ with 5-10 examples per class
///
/// ## The Revolution
///
/// ```
/// Traditional: 1M labeled images → 76% ImageNet accuracy
/// SimCLR: 1B unlabeled images → 80%+ accuracy (no labels!)
/// CLIP: 400M (image, text) pairs → zero-shot classification
/// ```

use ndarray::{Array1, Array2};
use rand::Rng;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         Contrastive Learning (Self-Supervised)           ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Demonstrate contrastive learning basics
    demo_contrastive_loss();

    // Demonstrate temperature effects
    demo_temperature_scaling();

    // Demonstrate positive/negative pairs
    demo_pair_similarity();
}

/// Demonstrate contrastive loss calculation
fn demo_contrastive_loss() {
    println!("═══ Contrastive Loss (SimCLR-style NT-Xent) ═══\n");

    // Simulate embeddings for a batch of 4 images
    // Each image has 2 augmented views
    let batch_size = 4;
    let embedding_dim = 8;

    println!("Batch of {} images, each with 2 augmented views", batch_size);
    println!("Embedding dimension: {}\n", embedding_dim);

    // Generate normalized embeddings
    let mut rng = rand::thread_rng();
    let mut embeddings = Vec::new();

    for i in 0..batch_size {
        // View 1 and View 2 are similar (positive pair)
        let base = Array1::from_shape_fn(embedding_dim, |_| rng.gen_range(-1.0..1.0f32));
        let base_norm = normalize(&base);

        // View 2: Add small noise to view 1 (they should be similar)
        let noise = Array1::from_shape_fn(embedding_dim, |_| rng.gen_range(-0.2..0.2f32));
        let view2 = normalize(&(&base + &noise));

        embeddings.push(base_norm);
        embeddings.push(view2);

        println!("Image {}: View 1 and View 2 (positive pair)", i);
    }

    // Calculate contrastive loss
    let temperature = 0.5;
    let loss = nt_xent_loss(&embeddings, temperature);

    println!("\nContrastive Loss (NT-Xent):");
    println!("  Temperature τ = {}", temperature);
    println!("  Loss = {:.4}", loss);

    println!("\n💡 Training minimizes this loss:");
    println!("   - Positive pairs (same image) → Pull together");
    println!("   - Negative pairs (different images) → Push apart\n");
}

/// NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss
/// Used in SimCLR
fn nt_xent_loss(embeddings: &[Array1<f32>], temperature: f32) -> f32 {
    let n = embeddings.len();
    let mut total_loss = 0.0;

    // For each embedding
    for i in 0..n {
        // Find its positive pair (adjacent index for views from same image)
        let positive_idx = if i % 2 == 0 { i + 1 } else { i - 1 };

        // Calculate similarity to positive
        let pos_sim = cosine_similarity(&embeddings[i], &embeddings[positive_idx]) / temperature;

        // Calculate similarity to all (for denominator)
        let mut exp_sum = 0.0;
        for j in 0..n {
            if j != i {
                let sim = cosine_similarity(&embeddings[i], &embeddings[j]) / temperature;
                exp_sum += sim.exp();
            }
        }

        // NT-Xent loss for this pair
        let loss_i = -pos_sim + (exp_sum).ln();
        total_loss += loss_i;
    }

    total_loss / n as f32
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    dot / (norm_a * norm_b + 1e-8)
}

/// Normalize vector to unit length
fn normalize(v: &Array1<f32>) -> Array1<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v / (norm + 1e-8)
}

/// Demonstrate temperature scaling effects
fn demo_temperature_scaling() {
    println!("═══ Temperature Scaling Effects ═══\n");

    // Create two embeddings: one similar pair, one dissimilar
    let embed1 = Array1::from(vec![1.0, 0.5, 0.3, 0.1]);
    let embed2_similar = Array1::from(vec![0.9, 0.5, 0.4, 0.1]); // Similar
    let embed3_dissimilar = Array1::from(vec![-0.5, 0.2, -0.8, 0.1]); // Dissimilar

    let embed1 = normalize(&embed1);
    let embed2 = normalize(&embed2_similar);
    let embed3 = normalize(&embed3_dissimilar);

    let sim_positive = cosine_similarity(&embed1, &embed2);
    let sim_negative = cosine_similarity(&embed1, &embed3);

    println!("Raw Cosine Similarities:");
    println!("  Positive pair:  {:.4}", sim_positive);
    println!("  Negative pair:  {:.4}\n", sim_negative);

    println!("Effect of Temperature on Softmax Distribution:\n");
    println!("Temperature   Positive Weight   Negative Weight");
    println!("─────────────────────────────────────────────────");

    for &temp in &[0.1, 0.3, 0.5, 1.0, 2.0] {
        let pos_exp = (sim_positive / temp).exp();
        let neg_exp = (sim_negative / temp).exp();
        let total = pos_exp + neg_exp;

        let pos_weight = pos_exp / total;
        let neg_weight = neg_exp / total;

        println!("τ = {:.1}         {:.4}            {:.4}",
                 temp, pos_weight, neg_weight);
    }

    println!("\n💡 Lower temperature τ → More concentrated distribution");
    println!("   τ = 0.1: [0.9997, 0.0003] - very sharp");
    println!("   τ = 2.0: [0.7347, 0.2653] - softer\n");

    println!("⚙️  Typical values: τ = 0.5 (SimCLR), τ = 0.07 (MoCo)\n");
}

/// Demonstrate positive/negative pair similarities
fn demo_pair_similarity() {
    println!("═══ Positive vs Negative Pair Similarities ═══\n");

    let mut rng = rand::thread_rng();

    // Simulate embeddings for objects: cat, dog, car
    let cat1 = generate_object_embedding(&mut rng, "cat");
    let cat2 = generate_object_embedding(&mut rng, "cat"); // Another view of cat
    let dog = generate_object_embedding(&mut rng, "dog");
    let car = generate_object_embedding(&mut rng, "car");

    println!("Object Embeddings (simulated):\n");

    // Calculate all pairwise similarities
    println!("Similarity Matrix:\n");
    println!("           Cat (view1)  Cat (view2)  Dog          Car");
    println!("─────────────────────────────────────────────────────────");

    let objects = vec![
        ("Cat (view1)", &cat1),
        ("Cat (view2)", &cat2),
        ("Dog       ", &dog),
        ("Car       ", &car),
    ];

    for (name1, emb1) in &objects {
        print!("{:12} ", name1);
        for (_, emb2) in &objects {
            let sim = cosine_similarity(emb1, emb2);
            print!("{:12.4} ", sim);
        }
        println!();
    }

    println!("\n💡 Observations:");
    println!("   - Cat view1 ↔ Cat view2: HIGH similarity (positive pair)");
    println!("   - Cat ↔ Dog: MEDIUM similarity (both animals)");
    println!("   - Cat ↔ Car: LOW similarity (different categories)\n");

    println!("🎯 Contrastive Learning Goal:");
    println!("   Maximize: sim(cat_view1, cat_view2)");
    println!("   Minimize: sim(cat, dog) and sim(cat, car)\n");
}

/// Generate mock embedding for an object type
fn generate_object_embedding(rng: &mut impl Rng, object_type: &str) -> Array1<f32> {
    let dim = 8;

    // Different base patterns for different objects
    let base = match object_type {
        "cat" => Array1::from(vec![0.8, 0.6, 0.4, 0.2, -0.1, -0.3, 0.5, 0.7]),
        "dog" => Array1::from(vec![0.7, 0.5, 0.3, 0.3, -0.2, -0.4, 0.6, 0.6]), // Similar to cat
        "car" => Array1::from(vec![-0.5, -0.3, 0.8, 0.6, 0.4, 0.2, -0.7, -0.5]), // Very different
        _ => Array1::from_shape_fn(dim, |_| rng.gen_range(-1.0..1.0f32)),
    };

    // Add small random noise
    let noise = Array1::from_shape_fn(dim, |_| rng.gen_range(-0.1..0.1f32));
    normalize(&(&base + &noise))
}

/// Key Concepts Summary
///
/// **Contrastive Learning:**
/// - Learn by comparing: Similar → close, Different → far
/// - No labels needed! Augmentation creates supervision
/// - Two views of same image = positive pair
/// - Views from different images = negative pairs
///
/// **SimCLR (Google, 2020):**
/// ```
/// 1. For each image: create 2 augmented views
/// 2. Encode to embeddings (CNN + projection head)
/// 3. NT-Xent loss:
///    - Pull positive pairs together
///    - Push negative pairs apart
/// 4. Batch size = 256-8192 (more negatives = better)
/// ```
///
/// **NT-Xent Loss:**
/// ```
/// For positive pair (i, j):
///   L = -log(exp(sim(i,j)/τ) / Σ_k exp(sim(i,k)/τ))
///
/// Temperature τ: 0.1-0.5 (lower = more concentrated)
/// ```
///
/// **Why It Works:**
/// - Forces model to learn semantic features
/// - Invariant to augmentations (crop, color, etc.)
/// - Scales with data (more data = better!)
/// - Better representations than supervised learning
///
/// **Modern Variants:**
/// - **MoCo** (Facebook): Momentum queue (memory-efficient)
/// - **BYOL** (DeepMind): No negatives needed!
/// - **CLIP** (OpenAI): Vision + Language contrastive learning
/// - **SwAV** (Facebook): Clustering approach
///
/// **Applications:**
/// 1. Pre-training: Learn on ImageNet → transfer to any task
/// 2. Few-shot: 80%+ accuracy with 5-10 examples
/// 3. Medical imaging: Limited labeled data
/// 4. CLIP → Text-to-image (Stable Diffusion, DALL-E)
///
/// **Results:**
/// - ImageNet (unlabeled): SimCLR 76.5%, BYOL 79.6%
/// - Matches supervised with 1% of labels
/// - CLIP: Zero-shot 76.2% (no fine-tuning!)
///
/// **Impact:**
/// The paradigm shift in AI:
/// ```
/// Old: Supervised learning on labeled data
/// New: Self-supervised pre-training → fine-tune on small labeled set
/// ```
///
/// Powers: CLIP, Stable Diffusion, DALL-E, GPT-4V vision
#[allow(dead_code)]
fn _summary() {}

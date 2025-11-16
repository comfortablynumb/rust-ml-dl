/// # Model Compression & Deployment 📦
///
/// Making deep learning models smaller, faster, and production-ready through:
/// - **Pruning**: Remove unnecessary weights (90% compression possible)
/// - **Quantization**: Reduce precision FP32 → INT8 (4× smaller, 2-4× faster)
/// - **Knowledge Distillation**: Train small student from large teacher
///
/// ## Why Compression Matters
///
/// **Production Reality:**
/// - GPT-3: 175B parameters, 700GB - too large for most deployments
/// - BERT: 110M params, 440MB - slow on CPU
/// - MobileNet: 3.5M params, 14MB - runs on phones
///
/// **Goals:**
/// - 10-100× smaller models
/// - 2-10× faster inference
/// - Deploy to mobile/edge devices
/// - Reduce cloud costs by 10×
///
/// ## Techniques Demonstrated
///
/// 1. **Magnitude Pruning**: Remove weights with |w| < threshold
/// 2. **Quantization**: FP32 → INT8 (simulated)
/// 3. **Knowledge Distillation**: Student learns from teacher's soft outputs

use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::HashMap;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         Model Compression & Deployment                  ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Demonstrate pruning
    demo_pruning();

    // Demonstrate quantization
    demo_quantization();

    // Demonstrate knowledge distillation
    demo_knowledge_distillation();
}

/// Demonstrate weight pruning
fn demo_pruning() {
    println!("═══ Weight Pruning (Remove Small Weights) ═══\n");

    // Create a sample weight matrix
    let mut rng = rand::thread_rng();
    let weights = Array2::from_shape_fn((4, 4), |_| {
        rng.gen_range(-1.0..1.0f32)
    });

    println!("Original weights:");
    print_matrix(&weights);

    let original_size = weights.len() * 4; // 4 bytes per f32
    println!("Original size: {} bytes\n", original_size);

    // Apply magnitude pruning at different sparsity levels
    for sparsity in [0.25, 0.5, 0.75, 0.9] {
        let pruned = magnitude_prune(&weights, sparsity);
        let non_zero = pruned.iter().filter(|&&x| x != 0.0).count();
        let compression = original_size as f32 / (non_zero * 4) as f32;

        println!("{}% sparsity ({}% weights removed):",
                 (sparsity * 100.0) as i32,
                 (sparsity * 100.0) as i32);
        print_matrix(&pruned);
        println!("Non-zero weights: {} / {}", non_zero, weights.len());
        println!("Compression ratio: {:.1}×\n", compression);
    }

    println!("💡 Real-world: 90% pruning achieves <1% accuracy loss on ResNet/BERT");
    println!("   with iterative magnitude pruning + fine-tuning\n");
}

/// Magnitude-based pruning: remove weights with |w| < threshold
fn magnitude_prune(weights: &Array2<f32>, sparsity: f32) -> Array2<f32> {
    // Calculate threshold: sparsity% of weights below this
    let mut magnitudes: Vec<f32> = weights.iter().map(|&w| w.abs()).collect();
    magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let threshold_idx = (magnitudes.len() as f32 * sparsity) as usize;
    let threshold = magnitudes[threshold_idx.min(magnitudes.len() - 1)];

    // Zero out weights below threshold
    weights.map(|&w| if w.abs() < threshold { 0.0 } else { w })
}

fn print_matrix(mat: &Array2<f32>) {
    for row in mat.rows() {
        print!("  [");
        for (i, &val) in row.iter().enumerate() {
            if i > 0 { print!(" "); }
            if val == 0.0 {
                print!(" 0.00 ");
            } else {
                print!("{:6.2}", val);
            }
        }
        println!("]");
    }
    println!();
}

/// Demonstrate quantization
fn demo_quantization() {
    println!("═══ Quantization (Reduce Precision) ═══\n");

    // Sample FP32 weights
    let fp32_weights = vec![-0.876, -0.234, 0.0, 0.123, 0.456, 0.789, 1.0];

    println!("Original FP32 weights:");
    for &w in &fp32_weights {
        print!("{:7.3} ", w);
    }
    println!("\n");

    // Quantize to INT8
    let (quantized, scale, zero_point) = quantize_int8(&fp32_weights);
    println!("Quantized to INT8:");
    println!("Scale: {:.6}, Zero point: {}", scale, zero_point);
    for &q in &quantized {
        print!("{:4} ", q);
    }
    println!("\n");

    // Dequantize back
    let dequantized = dequantize_int8(&quantized, scale, zero_point);
    println!("Dequantized back to FP32:");
    for &d in &dequantized {
        print!("{:7.3} ", d);
    }
    println!("\n");

    // Calculate quantization error
    let errors: Vec<f32> = fp32_weights.iter()
        .zip(&dequantized)
        .map(|(orig, deq)| (orig - deq).abs())
        .collect();

    let max_error = errors.iter().cloned().fold(0.0f32, f32::max);
    let avg_error = errors.iter().sum::<f32>() / errors.len() as f32;

    println!("Quantization Error:");
    println!("  Max error: {:.6}", max_error);
    println!("  Avg error: {:.6}", avg_error);
    println!("\n📊 Size reduction: FP32 (4 bytes) → INT8 (1 byte) = 4× smaller");
    println!("⚡ Speedup: 2-4× on CPU/mobile with INT8 operations");
    println!("💡 Typical accuracy loss: <1% for INT8, 2-5% for INT4\n");
}

/// Quantize FP32 to INT8 using affine quantization
fn quantize_int8(weights: &[f32]) -> (Vec<i8>, f32, i8) {
    // Find min and max
    let min_val = weights.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Calculate scale and zero point
    let range = max_val - min_val;
    let scale = range / 255.0; // INT8 range: -128 to 127 (255 values)
    let zero_point = -128.0 - min_val / scale;

    // Quantize
    let quantized: Vec<i8> = weights.iter()
        .map(|&w| {
            let q = ((w / scale) + zero_point).round();
            q.clamp(-128.0, 127.0) as i8
        })
        .collect();

    (quantized, scale, zero_point as i8)
}

/// Dequantize INT8 back to FP32
fn dequantize_int8(quantized: &[i8], scale: f32, zero_point: i8) -> Vec<f32> {
    quantized.iter()
        .map(|&q| (q as f32 - zero_point as f32) * scale)
        .collect()
}

/// Demonstrate knowledge distillation
fn demo_knowledge_distillation() {
    println!("═══ Knowledge Distillation (Teacher-Student) ═══\n");

    // Simulate teacher and student model outputs
    let teacher_logits = vec![3.0, 1.0, 0.1]; // Large model
    let student_logits = vec![2.5, 1.2, 0.3]; // Small model (before training)

    println!("Teacher model (large, accurate):");
    println!("  Logits: {:?}", teacher_logits);

    println!("\nStudent model (small, before distillation):");
    println!("  Logits: {:?}\n", student_logits);

    // Temperature scaling
    println!("Effect of Temperature on Softmax:\n");
    println!("Temperature  Teacher Probabilities                  Info Content");
    println!("───────────────────────────────────────────────────────────────────");

    for &temp in &[1.0, 2.0, 3.0, 5.0] {
        let teacher_probs = softmax_with_temperature(&teacher_logits, temp);
        let entropy = calculate_entropy(&teacher_probs);

        print!("T = {:.1}        ", temp);
        for &p in &teacher_probs {
            print!("{:.3}  ", p);
        }
        println!("   Entropy: {:.3}", entropy);
    }

    println!("\n💡 Higher temperature → softer probabilities → more information");
    println!("   T=1: [0.88, 0.09, 0.03] - nearly one-hot, little info");
    println!("   T=3: [0.61, 0.29, 0.10] - reveals relative confidences\n");

    // Distillation loss calculation
    let temperature = 3.0;
    let alpha = 0.5; // Balance between hard and soft targets

    let teacher_soft = softmax_with_temperature(&teacher_logits, temperature);
    let student_soft = softmax_with_temperature(&student_logits, temperature);

    let true_label = 0; // First class is correct
    let hard_loss = cross_entropy_loss(&student_soft, true_label);
    let soft_loss = kl_divergence(&student_soft, &teacher_soft);
    let total_loss = alpha * hard_loss + (1.0 - alpha) * soft_loss;

    println!("Distillation Loss Components:");
    println!("  Hard loss (true labels):    {:.4}", hard_loss);
    println!("  Soft loss (teacher KL div): {:.4}", soft_loss);
    println!("  Total loss (α={:.1}):        {:.4}", alpha, total_loss);

    println!("\n🎓 Famous Examples:");
    println!("  BERT-base (110M) → DistilBERT (66M): 1.6× smaller, 97% accuracy");
    println!("  BERT-large (340M) → TinyBERT (14M): 24× smaller, 96% accuracy");
    println!("  ResNet-152 → ResNet-18: 8.4× smaller, 95% accuracy");
    println!();
}

/// Softmax with temperature
fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Subtract max for numerical stability
    let exp_values: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_values.iter().sum();

    exp_values.iter().map(|&x| x / sum).collect()
}

/// Calculate entropy of probability distribution
fn calculate_entropy(probs: &[f32]) -> f32 {
    -probs.iter()
        .filter(|&&p| p > 1e-10) // Avoid log(0)
        .map(|&p| p * p.log2())
        .sum::<f32>()
}

/// Cross-entropy loss with true label
fn cross_entropy_loss(probs: &[f32], true_label: usize) -> f32 {
    -probs[true_label].ln()
}

/// KL divergence between student and teacher distributions
fn kl_divergence(student: &[f32], teacher: &[f32]) -> f32 {
    teacher.iter()
        .zip(student.iter())
        .map(|(&t, &s)| {
            if t > 1e-10 && s > 1e-10 {
                t * (t / s).ln()
            } else {
                0.0
            }
        })
        .sum()
}

/// Key Concepts Summary
///
/// **Pruning:**
/// - Magnitude pruning: Remove weights with |w| < threshold
/// - Iterative: Prune → Fine-tune → Repeat
/// - 90% sparsity achievable with <1% accuracy loss
/// - Structured pruning (channels/filters) → real speedup
///
/// **Quantization:**
/// - FP32 (4 bytes) → INT8 (1 byte) = 4× smaller
/// - Post-training quantization (PTQ): No retraining
/// - Quantization-aware training (QAT): Better accuracy
/// - 2-4× inference speedup on CPU/mobile
/// - <1% accuracy loss for INT8, 2-5% for INT4
///
/// **Knowledge Distillation:**
/// - Teacher (large, accurate) trains student (small, fast)
/// - Soft targets reveal relative confidences
/// - Temperature scaling: Higher T → more information
/// - Loss = α * hard_loss + (1-α) * soft_loss
/// - 10-24× compression with 95-97% accuracy retained
///
/// **Deployment Strategy:**
/// ```
/// 1. Train large teacher model
/// 2. Apply pruning (70-90% sparsity)
/// 3. Distill to smaller student
/// 4. Quantize to INT8
/// Total: 50-100× compression, 5-10× speedup
/// ```
///
/// **Real-World Usage:**
/// - Mobile: TensorFlow Lite, Core ML (INT8 quantization)
/// - Edge: ONNX Runtime, OpenVINO (pruning + INT8)
/// - GPU: TensorRT (FP16/INT8 mixed precision)
/// - Production: All large models use some compression
///
/// **Best Quick Win:**
/// Post-training quantization to INT8:
/// - 4× smaller, 2-4× faster, <1% accuracy loss
/// - No retraining needed
/// - Supported everywhere
#[allow(dead_code)]
fn _summary() {}

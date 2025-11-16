/// # Efficient Transformers ⚡
///
/// Making Transformers fast and scalable by reducing the O(n²) attention bottleneck
/// to O(n) through linear attention, Flash Attention concepts, and sparse patterns.
///
/// ## What This Example Demonstrates
///
/// 1. **Standard Attention**: O(n²) complexity baseline
/// 2. **Linear Attention**: O(n) complexity approximation
/// 3. **Sparse Attention Patterns**: Local, strided, block-sparse
/// 4. **Complexity Comparison**: Memory and compute analysis
///
/// ## Why Efficient Transformers Matter
///
/// - **Standard Transformers**: Limited to 512-2048 tokens
/// - **Efficient Transformers**: 4K-128K tokens possible
/// - **GPT-4**: 32K context (likely Flash Attention)
/// - **Claude 2**: 100K context (efficient attention)
///
/// ## The Problem
///
/// ```
/// Standard attention: O(n²) memory and compute
/// For n=4096: 16M operations (GPUs struggle)
/// For n=16384: 256M operations (impossible!)
/// ```

use ndarray::Array2;
use rand::Rng;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         Efficient Transformers (O(n²) → O(n))            ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Demonstrate complexity differences
    demo_complexity_comparison();

    // Demonstrate linear vs standard attention
    demo_linear_attention();

    // Demonstrate sparse attention patterns
    demo_sparse_attention();
}

/// Demonstrate complexity comparison
fn demo_complexity_comparison() {
    println!("═══ Complexity Comparison: O(n²) vs O(n) ═══\n");

    println!("Attention Complexity for Different Sequence Lengths:\n");
    println!("Seq Len   Standard O(n²)   Linear O(n)   Speedup");
    println!("───────────────────────────────────────────────────────");

    for &n in &[512, 1024, 2048, 4096, 8192, 16384] {
        let standard_ops = n * n;
        let linear_ops = n * 64; // d=64 typical
        let speedup = standard_ops as f32 / linear_ops as f32;

        println!("{:6}    {:12}    {:11}    {:.1}×",
                 n,
                 format_large_number(standard_ops),
                 format_large_number(linear_ops),
                 speedup);
    }

    println!("\n💡 Key Insight:");
    println!("   For n=16384: Linear attention is 256× faster!");
    println!("   Memory usage: O(n²) → O(n) = 16384× reduction\n");
}

fn format_large_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f32 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f32 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

/// Demonstrate linear attention
fn demo_linear_attention() {
    println!("═══ Linear Attention (O(n) Complexity) ═══\n");

    let seq_len = 256;
    let d_model = 32;
    let mut rng = rand::thread_rng();

    // Generate random Q, K, V matrices
    let q = Array2::from_shape_fn((seq_len, d_model), |_| rng.gen_range(-1.0..1.0f32));
    let k = Array2::from_shape_fn((seq_len, d_model), |_| rng.gen_range(-1.0..1.0f32));
    let v = Array2::from_shape_fn((seq_len, d_model), |_| rng.gen_range(-1.0..1.0f32));

    println!("Input dimensions:");
    println!("  Q, K, V: {} × {} (seq_len × d_model)\n", seq_len, d_model);

    // Standard attention
    println!("1. Standard Attention: O(n²)");
    let start = Instant::now();
    let output_standard = standard_attention(&q, &k, &v);
    let time_standard = start.elapsed();
    println!("   Time: {:?}", time_standard);
    println!("   Memory: {} × {} = {} attention scores\n",
             seq_len, seq_len, seq_len * seq_len);

    // Linear attention
    println!("2. Linear Attention: O(n)");
    let start = Instant::now();
    let output_linear = linear_attention(&q, &k, &v);
    let time_linear = start.elapsed();
    println!("   Time: {:?}", time_linear);
    println!("   Memory: {} × {} = {} intermediate values\n",
             d_model, d_model, d_model * d_model);

    // Compare outputs
    let diff = compute_difference(&output_standard, &output_linear);
    println!("Output difference (L2 norm): {:.6}", diff);
    println!("Relative error: {:.2}%", (diff / output_norm(&output_standard)) * 100.0);

    println!("\n💡 Linear Attention Trade-off:");
    println!("   Pros: O(n) complexity, constant memory");
    println!("   Cons: Approximation (~1-2% accuracy loss)");
    println!("   Use when: n > 4K and speed > exact accuracy\n");
}

/// Standard attention: O(n²)
fn standard_attention(q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
    let seq_len = q.shape()[0];
    let d_model = q.shape()[1];

    // Compute attention scores: Q @ K^T (n × n matrix)
    let mut scores = Array2::<f32>::zeros((seq_len, seq_len));
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut sum = 0.0;
            for k_idx in 0..d_model {
                sum += q[[i, k_idx]] * k[[j, k_idx]];
            }
            scores[[i, j]] = sum / (d_model as f32).sqrt();
        }
    }

    // Apply softmax row-wise
    for i in 0..seq_len {
        let max_score = (0..seq_len)
            .map(|j| scores[[i, j]])
            .fold(f32::NEG_INFINITY, f32::max);

        let mut sum_exp = 0.0;
        for j in 0..seq_len {
            scores[[i, j]] = (scores[[i, j]] - max_score).exp();
            sum_exp += scores[[i, j]];
        }

        for j in 0..seq_len {
            scores[[i, j]] /= sum_exp;
        }
    }

    // Compute output: Attention @ V
    let mut output = Array2::<f32>::zeros((seq_len, d_model));
    for i in 0..seq_len {
        for k_idx in 0..d_model {
            let mut sum = 0.0;
            for j in 0..seq_len {
                sum += scores[[i, j]] * v[[j, k_idx]];
            }
            output[[i, k_idx]] = sum;
        }
    }

    output
}

/// Linear attention: O(n)
fn linear_attention(q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
    let seq_len = q.shape()[0];
    let d_model = q.shape()[1];

    // Apply feature map φ(x) = relu(x) + 1 (simple approximation)
    let phi_q = q.map(|&x| x.max(0.0) + 1.0);
    let phi_k = k.map(|&x| x.max(0.0) + 1.0);

    // Compute K^T @ V first (d × d matrix, not n × n!)
    let mut kv = Array2::<f32>::zeros((d_model, d_model));
    for i in 0..d_model {
        for j in 0..d_model {
            let mut sum = 0.0;
            for n in 0..seq_len {
                sum += phi_k[[n, i]] * v[[n, j]];
            }
            kv[[i, j]] = sum;
        }
    }

    // Compute Q @ (K^T @ V)
    let mut output = Array2::<f32>::zeros((seq_len, d_model));
    for i in 0..seq_len {
        for j in 0..d_model {
            let mut sum = 0.0;
            for k_idx in 0..d_model {
                sum += phi_q[[i, k_idx]] * kv[[k_idx, j]];
            }

            // Normalization
            let mut norm = 0.0;
            for k_idx in 0..d_model {
                norm += phi_q[[i, k_idx]];
            }

            output[[i, j]] = sum / (norm + 1e-8);
        }
    }

    output
}

fn compute_difference(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    let mut sum_sq = 0.0;
    for i in 0..a.shape()[0] {
        for j in 0..a.shape()[1] {
            let diff = a[[i, j]] - b[[i, j]];
            sum_sq += diff * diff;
        }
    }
    sum_sq.sqrt()
}

fn output_norm(a: &Array2<f32>) -> f32 {
    let mut sum_sq = 0.0;
    for i in 0..a.shape()[0] {
        for j in 0..a.shape()[1] {
            sum_sq += a[[i, j]] * a[[i, j]];
        }
    }
    sum_sq.sqrt()
}

/// Demonstrate sparse attention patterns
fn demo_sparse_attention() {
    println!("═══ Sparse Attention Patterns ═══\n");

    let seq_len = 32;

    println!("Sequence length: {}\n", seq_len);

    // 1. Local (sliding window) attention
    println!("1. Local Attention (Sliding Window):");
    let window_size = 8;
    let local_pattern = generate_local_attention_pattern(seq_len, window_size);
    println!("   Window size: {}", window_size);
    println!("   Connections per token: {}", window_size);
    print_sparse_pattern(&local_pattern, "Local (k=8)");

    // 2. Strided attention
    println!("\n2. Strided Attention:");
    let stride = 4;
    let strided_pattern = generate_strided_attention_pattern(seq_len, stride);
    println!("   Stride: {}", stride);
    println!("   Global tokens: every {}-th position", stride);
    print_sparse_pattern(&strided_pattern, "Strided (s=4)");

    // 3. Block-sparse attention
    println!("\n3. Block-Sparse Attention:");
    let block_size = 8;
    let block_pattern = generate_block_sparse_pattern(seq_len, block_size);
    println!("   Block size: {}", block_size);
    print_sparse_pattern(&block_pattern, "Block-sparse");

    // Complexity comparison
    println!("\n📊 Complexity Comparison:");
    let full_connections = seq_len * seq_len;
    let local_connections = count_connections(&local_pattern);
    let strided_connections = count_connections(&strided_pattern);
    let block_connections = count_connections(&block_pattern);

    println!("   Full attention:   {} connections (100%)", full_connections);
    println!("   Local attention:  {} connections ({:.0}%)",
             local_connections,
             (local_connections as f32 / full_connections as f32) * 100.0);
    println!("   Strided:          {} connections ({:.0}%)",
             strided_connections,
             (strided_connections as f32 / full_connections as f32) * 100.0);
    println!("   Block-sparse:     {} connections ({:.0}%)",
             block_connections,
             (block_connections as f32 / full_connections as f32) * 100.0);

    println!("\n💡 Sparse patterns used in:");
    println!("   - Longformer: Local + Global sparse");
    println!("   - BigBird: Local + Random + Global");
    println!("   - Sparse Transformer: Strided patterns\n");
}

fn generate_local_attention_pattern(seq_len: usize, window_size: usize) -> Vec<Vec<bool>> {
    let mut pattern = vec![vec![false; seq_len]; seq_len];
    let half_window = window_size / 2;

    for i in 0..seq_len {
        for j in 0..seq_len {
            if i.abs_diff(j) <= half_window {
                pattern[i][j] = true;
            }
        }
    }

    pattern
}

fn generate_strided_attention_pattern(seq_len: usize, stride: usize) -> Vec<Vec<bool>> {
    let mut pattern = vec![vec![false; seq_len]; seq_len];

    for i in 0..seq_len {
        // Local attention
        for j in i.saturating_sub(2)..=(i + 2).min(seq_len - 1) {
            pattern[i][j] = true;
        }

        // Global attention to strided positions
        for j in (0..seq_len).step_by(stride) {
            pattern[i][j] = true;
        }
    }

    pattern
}

fn generate_block_sparse_pattern(seq_len: usize, block_size: usize) -> Vec<Vec<bool>> {
    let mut pattern = vec![vec![false; seq_len]; seq_len];
    let num_blocks = seq_len / block_size;

    for block_i in 0..num_blocks {
        for block_j in 0..num_blocks {
            // Diagonal blocks + one block above and below
            if (block_i as i32 - block_j as i32).abs() <= 1 {
                for i in 0..block_size {
                    for j in 0..block_size {
                        let row = block_i * block_size + i;
                        let col = block_j * block_size + j;
                        if row < seq_len && col < seq_len {
                            pattern[row][col] = true;
                        }
                    }
                }
            }
        }
    }

    pattern
}

fn print_sparse_pattern(pattern: &[Vec<bool>], title: &str) {
    println!("\n   {} pattern (. = attend, · = skip):", title);
    println!("   {}", "─".repeat(pattern.len() + 4));

    // Print only first 16x16 for readability
    let display_size = pattern.len().min(16);
    for i in 0..display_size {
        print!("   ");
        for j in 0..display_size {
            if pattern[i][j] {
                print!("█");
            } else {
                print!("·");
            }
        }
        println!();
    }

    if pattern.len() > 16 {
        println!("   (showing first 16×16 of {}×{})", pattern.len(), pattern.len());
    }
}

fn count_connections(pattern: &[Vec<bool>]) -> usize {
    pattern.iter()
        .flat_map(|row| row.iter())
        .filter(|&&x| x)
        .count()
}

/// Key Concepts Summary
///
/// **The Attention Bottleneck:**
/// ```
/// Standard Attention: O(n²) in memory and compute
/// - For n=2048: 4M attention scores
/// - For n=16384: 256M attention scores (impossible!)
/// ```
///
/// **Efficient Approaches:**
///
/// **1. Linear Attention:**
/// ```
/// Standard: Attention = softmax(QK^T) V  [O(n²)]
/// Linear:   Attention = φ(Q) (φ(K)^T V) [O(n)]
///
/// Key: Reorder computation to avoid n×n matrix
/// Trade-off: 1-2% accuracy loss for 10-100× speedup
/// ```
///
/// **2. Flash Attention:**
/// ```
/// Problem: Memory transfers (HBM ↔ SRAM) are bottleneck
/// Solution: Tiled computation + online softmax
/// Result: Exact attention with O(n) memory
/// Speedup: 2-4× faster, 5-20× less memory
/// ```
///
/// **3. Sparse Patterns:**
/// ```
/// Local: Each token attends to k neighbors [O(nk)]
/// Strided: Every s-th token attends globally
/// Block-sparse: Attend to block patterns
/// Complexity: O(nk) where k << n
/// ```
///
/// **Real-World Usage:**
/// - **GPT-4**: 32K context (Flash Attention)
/// - **Claude 2**: 100K context (efficient attention)
/// - **Longformer**: 4K tokens (sparse patterns)
/// - **Llama 2**: Flash Attention for training
///
/// **When to Use:**
/// - n < 512: Standard attention (fast enough)
/// - 512 < n < 4K: Flash Attention (exact, fast)
/// - n > 4K: Linear or sparse (enables long context)
///
/// **Impact:**
/// Without efficient attention:
/// - Limited to 512-2048 token context
/// - High memory requirements
/// - Slow training and inference
///
/// With efficient attention:
/// - 4K-128K token context
/// - 2-100× faster
/// - Practical long-context AI
#[allow(dead_code)]
fn _summary() {}

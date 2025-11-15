/// # Explainability with Grad-CAM
///
/// Gradient-weighted Class Activation Mapping for visualizing CNN decisions.
///
/// ## Why Production Essential:
/// ```
/// Medical AI: FDA requires explainability
/// Finance: GDPR "right to explanation"
/// Debugging: See what model sees → fix faster
/// Trust: Stakeholders need understanding
/// ```
///
/// ## Algorithm:
/// ```
/// 1. Forward pass → activations A
/// 2. Backward pass → gradients ∂y^c/∂A
/// 3. Global avg pool → weights α = mean(gradients)
/// 4. Weighted sum → L = Σ α_k · A^k
/// 5. ReLU(L) → heatmap
/// ```

use ndarray::{Array2, Array3};
use rand::Rng;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║    Explainability with Grad-CAM (Production Essential)   ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Demonstrate Grad-CAM computation
    demo_gradcam_computation();

    // Demonstrate debugging with Grad-CAM
    demo_model_debugging();

    // Demonstrate bias detection
    demo_bias_detection();
}

/// Demonstrate Grad-CAM computation
fn demo_gradcam_computation() {
    println!("═══ Grad-CAM Computation ═══\n");

    // Simulate feature maps from last conv layer (4 channels, 8×8 spatial)
    let channels = 4;
    let height = 8;
    let width = 8;

    println!("Setup:");
    println!("  Last conv layer: {} channels, {}×{} spatial", channels, height, width);
    println!("  Classes: Cat, Dog, Bird\n");

    // Create sample activations (channels × height × width)
    let mut activations = Array3::zeros((channels, height, width));
    let mut rng = rand::thread_rng();

    for c in 0..channels {
        for i in 0..height {
            for j in 0..width {
                // Create patterns: different channels activate in different regions
                let val = if c == 0 && i < 4 && j < 4 {
                    rng.gen_range(0.5..1.0)  // Top-left
                } else if c == 1 && i < 4 && j >= 4 {
                    rng.gen_range(0.5..1.0)  // Top-right
                } else if c == 2 && i >= 4 && j < 4 {
                    rng.gen_range(0.5..1.0)  // Bottom-left
                } else if c == 3 && i >= 4 && j >= 4 {
                    rng.gen_range(0.5..1.0)  // Bottom-right
                } else {
                    rng.gen_range(0.0..0.3)  // Low activation elsewhere
                };
                activations[[c, i, j]] = val;
            }
        }
    }

    // Simulate gradients for "Cat" class
    // In real model: ∂y_cat / ∂A (same shape as activations)
    let mut gradients = Array3::zeros((channels, height, width));

    // Channel 0 and 2 are important for "Cat" (high gradients)
    for c in [0, 2] {
        for i in 0..height {
            for j in 0..width {
                gradients[[c, i, j]] = rng.gen_range(0.5..1.0);
            }
        }
    }

    // Channels 1, 3 less important
    for c in [1, 3] {
        for i in 0..height {
            for j in 0..width {
                gradients[[c, i, j]] = rng.gen_range(0.0..0.3);
            }
        }
    }

    println!("Step 1: Forward Pass");
    println!("  ✓ Activations captured: {} channels", channels);

    println!("\nStep 2: Backward Pass");
    println!("  ✓ Gradients computed: ∂y_cat / ∂A");

    // Step 3: Global Average Pooling (get importance weights per channel)
    let weights = compute_channel_weights(&gradients);

    println!("\nStep 3: Global Average Pooling");
    println!("  Channel importance weights (α):");
    for (i, &w) in weights.iter().enumerate() {
        println!("    Channel {}: {:.3}", i, w);
    }

    // Step 4: Weighted combination
    let raw_heatmap = weighted_combination(&activations, &weights);

    println!("\nStep 4: Weighted Combination");
    println!("  L = Σ α_k · A^k");
    println!("  Heatmap shape: {}×{}", raw_heatmap.nrows(), raw_heatmap.ncols());

    // Step 5: ReLU (keep only positive)
    let heatmap = raw_heatmap.mapv(|x| x.max(0.0));

    println!("\nStep 5: ReLU (positive contributions only)");
    println!("  Final heatmap ready for visualization\n");

    // Visualize heatmap
    println!("Grad-CAM Heatmap (Cat class):");
    println!("━━━━━━━━━━━━━━━━");
    visualize_heatmap(&heatmap);

    println!("\n💡 Interpretation:");
    println!("  ✓ Model focuses on top-left and bottom-left regions");
    println!("  ✓ These areas contain cat-specific features");
    println!("  ✓ Other regions contribute less to 'Cat' prediction\n");
}

/// Demonstrate model debugging with Grad-CAM
fn demo_model_debugging() {
    println!("═══ Model Debugging with Grad-CAM ═══\n");

    println!("Scenario: CNN classifies snowy images");
    println!("──────────────────────────────────────\n");

    // Case 1: Correct reasoning
    println!("Case 1: Husky on grass");
    println!("  Prediction: Husky (92%)");
    println!("  Grad-CAM heatmap:");

    let correct_heatmap = create_focused_heatmap(8, 8, vec![(2, 2), (2, 3), (3, 2), (3, 3)]);
    visualize_heatmap(&correct_heatmap);
    println!("  ✅ Model focuses on dog (center), correct reasoning!\n");

    // Case 2: Spurious correlation
    println!("Case 2: Husky on snow");
    println!("  Prediction: Husky (94%)");
    println!("  Grad-CAM heatmap:");

    // Heatmap shows model focuses on background (snow)
    let spurious_heatmap = create_diffuse_heatmap(8, 8);
    visualize_heatmap(&spurious_heatmap);
    println!("  ⚠️  Model focuses on SNOW, not dog!");
    println!("  Problem: Learned spurious correlation (huskies → snow)");
    println!("  Solution: Add data augmentation, diverse backgrounds\n");

    println!("Debug Process:");
    println!("  1. Notice: High accuracy on training, poor on new backgrounds");
    println!("  2. Grad-CAM: Reveals focus on wrong features");
    println!("  3. Root cause: Dataset bias (all huskies in snow)");
    println!("  4. Fix: Collect diverse data, add augmentation");
    println!("  5. Verify: Grad-CAM shows correct focus after retraining\n");
}

/// Demonstrate bias detection
fn demo_bias_detection() {
    println!("═══ Bias Detection with Grad-CAM ═══\n");

    println!("Scenario: Medical image classifier");
    println!("──────────────────────────────────────\n");

    println!("Hospital A X-rays:");
    println!("  Prediction: Pneumonia (88%)");
    println!("  Grad-CAM:");
    let hospital_a = create_focused_heatmap(8, 8, vec![(3, 3), (3, 4), (4, 3), (4, 4)]);
    visualize_heatmap(&hospital_a);
    println!("  ✅ Focuses on lung region (center)\n");

    println!("Hospital B X-rays:");
    println!("  Prediction: Pneumonia (91%)");
    println!("  Grad-CAM:");
    // Model focuses on corner (hospital marker)
    let hospital_b = create_focused_heatmap(8, 8, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    visualize_heatmap(&hospital_b);
    println!("  ⚠️  Focuses on TOP-LEFT corner (hospital watermark!)");
    println!("  Problem: Dataset bias - Hospital B always has logo");
    println!("  Impact: Model won't generalize to other hospitals\n");

    println!("Real-World Example (COVID-19 2020):");
    println!("  • Study: CNN detects COVID from chest X-rays");
    println!("  • Accuracy: 90%+ on test set");
    println!("  • Grad-CAM revealed: Focused on hospital bed markers!");
    println!("  • Reason: COVID patients → ICU → specific equipment");
    println!("  • Fix: Retrain with diverse hospital data");
    println!("  • Lesson: Always use Grad-CAM before deployment\n");
}

/// Compute channel importance weights via global average pooling
fn compute_channel_weights(gradients: &Array3<f32>) -> Vec<f32> {
    let channels = gradients.shape()[0];
    let mut weights = Vec::with_capacity(channels);

    for c in 0..channels {
        // Global average pool over spatial dimensions
        let channel_slice = gradients.slice(s![c, .., ..]);
        let mean = channel_slice.mean().unwrap_or(0.0);
        weights.push(mean);
    }

    weights
}

/// Weighted combination of activation maps
fn weighted_combination(activations: &Array3<f32>, weights: &[f32]) -> Array2<f32> {
    let channels = activations.shape()[0];
    let height = activations.shape()[1];
    let width = activations.shape()[2];

    let mut result = Array2::zeros((height, width));

    for c in 0..channels {
        let channel_slice = activations.slice(s![c, .., ..]);
        result = result + &(channel_slice.to_owned() * weights[c]);
    }

    result
}

/// Create a heatmap focused on specific locations
fn create_focused_heatmap(height: usize, width: usize, focus_points: Vec<(usize, usize)>) -> Array2<f32> {
    let mut heatmap = Array2::zeros((height, width));

    for &(i, j) in &focus_points {
        heatmap[[i, j]] = 1.0;

        // Add gradient around focus points
        for di in -1..=1_i32 {
            for dj in -1..=1_i32 {
                let ni = (i as i32 + di).max(0).min((height - 1) as i32) as usize;
                let nj = (j as i32 + dj).max(0).min((width - 1) as i32) as usize;
                if (di, dj) != (0, 0) {
                    heatmap[[ni, nj]] = f32::max(heatmap[[ni, nj]], 0.5);
                }
            }
        }
    }

    // Normalize
    let max_val = heatmap.iter().cloned().fold(0.0_f32, f32::max);
    if max_val > 0.0 {
        heatmap = heatmap.mapv_into(|x| x / max_val);
    }

    heatmap
}

/// Create a diffuse heatmap (no clear focus)
fn create_diffuse_heatmap(height: usize, width: usize) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    let mut heatmap = Array2::zeros((height, width));

    for i in 0..height {
        for j in 0..width {
            // Higher values on edges (simulating background focus)
            let edge_dist = i.min(j).min(height - 1 - i).min(width - 1 - j) as f32;
            let val = if edge_dist < 2.0 {
                rng.gen_range(0.5..1.0)
            } else {
                rng.gen_range(0.0..0.4)
            };
            heatmap[[i, j]] = val;
        }
    }

    // Normalize
    let max_val = heatmap.iter().cloned().fold(0.0_f32, f32::max);
    if max_val > 0.0 {
        heatmap = heatmap.mapv_into(|x| x / max_val);
    }

    heatmap
}

/// Visualize heatmap using ASCII art
fn visualize_heatmap(heatmap: &Array2<f32>) {
    let chars = [' ', '░', '▒', '▓', '█'];

    for i in 0..heatmap.nrows() {
        print!("  ");
        for j in 0..heatmap.ncols() {
            let val = heatmap[[i, j]];
            let idx = (val * (chars.len() - 1) as f32).round() as usize;
            let idx = idx.min(chars.len() - 1);
            print!("{}", chars[idx]);
        }
        println!();
    }
}

// Import for slice macro
use ndarray::s;

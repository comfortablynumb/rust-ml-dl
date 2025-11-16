use ndarray::{Array3, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::Rng;
use std::f64::consts::PI;

/// Represents an RGB image as a 3D array (height, width, channels)
type Image = Array3<f64>;

/// Data augmentation demonstration for computer vision tasks
///
/// This example shows various image augmentation techniques that are fundamental
/// to modern computer vision. These techniques provide 5-15% accuracy improvements
/// by teaching models to be invariant to transformations and effectively expanding
/// the training dataset.

fn main() {
    println!("=== Data Augmentation: The Easy Performance Boost ===\n");

    // Create a synthetic test image (checkerboard pattern with gradient)
    let image = create_test_image(64, 64);
    println!("Created test image: {}x{} pixels, {} channels",
             image.shape()[0], image.shape()[1], image.shape()[2]);

    println!("\n--- Geometric Transformations ---\n");

    // Horizontal flip - simplest augmentation, doubles dataset
    let flipped_h = horizontal_flip(&image);
    println!("✓ Horizontal flip: Teaches left-right invariance");
    print_image_stats("Original", &image);
    print_image_stats("Flipped H", &flipped_h);

    // Vertical flip - useful for medical/aerial imagery
    let flipped_v = vertical_flip(&image);
    println!("\n✓ Vertical flip: Useful for orientation-agnostic tasks");
    print_image_stats("Flipped V", &flipped_v);

    // Rotation - teaches orientation invariance
    let rotated = rotate(&image, 30.0);
    println!("\n✓ Rotation (30°): Handles different viewing angles");
    print_image_stats("Rotated", &rotated);

    // Random crop - forces model to recognize objects at different scales
    let cropped = random_crop(&image, 48, 48);
    println!("\n✓ Random crop (48x48 from 64x64): Scale and position invariance");
    print_image_stats("Cropped", &cropped);

    // Random scaling
    let scaled = random_scale(&image, 0.8, 1.2);
    println!("\n✓ Random scale (0.8-1.2x): Handles different object sizes");
    print_image_stats("Scaled", &scaled);

    println!("\n--- Color Space Transformations ---\n");

    // Brightness adjustment
    let bright = adjust_brightness(&image, 0.3);
    println!("✓ Brightness +30%: Handles different lighting conditions");
    print_color_stats("Original", &image);
    print_color_stats("Brightened", &bright);

    // Contrast adjustment
    let contrast = adjust_contrast(&image, 1.5);
    println!("\n✓ Contrast 1.5x: Handles different exposure settings");
    print_color_stats("Enhanced", &contrast);

    // Saturation adjustment
    let saturated = adjust_saturation(&image, 1.3);
    println!("\n✓ Saturation 1.3x: Color intensity variations");
    print_color_stats("Saturated", &saturated);

    // Hue shift
    let hue_shifted = adjust_hue(&image, 0.1);
    println!("\n✓ Hue shift: Color tint variations");
    print_color_stats("Hue shifted", &hue_shifted);

    // Color jitter - combines all color augmentations
    let jittered = color_jitter(&image, 0.2, 0.2, 0.2, 0.1);
    println!("\n✓ Color jitter: Random brightness/contrast/saturation/hue");
    print_color_stats("Jittered", &jittered);

    println!("\n--- Advanced Augmentations ---\n");

    // Cutout - randomly mask patches
    let cutout_img = cutout(&image, 16, 16, 1);
    println!("✓ Cutout (1 patch of 16x16): Forces model to use multiple cues");
    print_image_stats("With cutout", &cutout_img);

    // Gaussian noise
    let noisy = add_gaussian_noise(&image, 0.05);
    println!("\n✓ Gaussian noise (σ=0.05): Robustness to image quality");
    print_image_stats("Noisy", &noisy);

    // Gaussian blur
    let blurred = gaussian_blur(&image, 1.5);
    println!("\n✓ Gaussian blur (σ=1.5): Simulates defocus");
    print_image_stats("Blurred", &blurred);

    println!("\n--- Transformation Matrices ---\n");
    demonstrate_affine_transforms();

    println!("\n--- Mixup Augmentation ---\n");
    demonstrate_mixup();

    println!("\n--- Performance Impact Analysis ---\n");
    analyze_augmentation_impact();

    println!("\n--- Augmentation Pipeline ---\n");
    demonstrate_pipeline(&image);

    println!("\n=== Summary ===\n");
    print_summary();
}

/// Create a test image with interesting patterns
fn create_test_image(height: usize, width: usize) -> Image {
    let mut img = Array3::<f64>::zeros((height, width, 3));

    for i in 0..height {
        for j in 0..width {
            // Checkerboard pattern
            let checker = if (i / 8 + j / 8) % 2 == 0 { 0.8 } else { 0.2 };

            // Gradient
            let grad_h = i as f64 / height as f64;
            let grad_w = j as f64 / width as f64;

            // Combine patterns with different colors for each channel
            img[[i, j, 0]] = checker * (1.0 - grad_h * 0.3); // Red channel
            img[[i, j, 1]] = checker * (1.0 - grad_w * 0.3); // Green channel
            img[[i, j, 2]] = checker * (1.0 - (grad_h + grad_w) * 0.15); // Blue channel
        }
    }

    img
}

// ============================================================================
// Geometric Transformations
// ============================================================================

/// Flip image horizontally (left-right mirror)
fn horizontal_flip(image: &Image) -> Image {
    image.slice(s![.., ..; -1, ..]).to_owned()
}

/// Flip image vertically (top-bottom mirror)
fn vertical_flip(image: &Image) -> Image {
    image.slice(s![..; -1, .., ..]).to_owned()
}

/// Rotate image by angle (in degrees) using affine transformation
fn rotate(image: &Image, angle: f64) -> Image {
    let (h, w, c) = (image.shape()[0], image.shape()[1], image.shape()[2]);
    let mut rotated = Array3::<f64>::zeros((h, w, c));

    let angle_rad = angle * PI / 180.0;
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;

    // Apply rotation transformation
    for i in 0..h {
        for j in 0..w {
            // Translate to origin
            let x = j as f64 - cx;
            let y = i as f64 - cy;

            // Rotate
            let x_rot = x * cos_a - y * sin_a;
            let y_rot = x * sin_a + y * cos_a;

            // Translate back
            let src_x = x_rot + cx;
            let src_y = y_rot + cy;

            // Bilinear interpolation
            if src_x >= 0.0 && src_x < (w - 1) as f64 &&
               src_y >= 0.0 && src_y < (h - 1) as f64 {
                let value = bilinear_interpolate(image, src_y, src_x);
                for ch in 0..c {
                    rotated[[i, j, ch]] = value[ch];
                }
            }
        }
    }

    rotated
}

/// Extract random crop from image
fn random_crop(image: &Image, crop_h: usize, crop_w: usize) -> Image {
    let (h, w, _) = (image.shape()[0], image.shape()[1], image.shape()[2]);

    let mut rng = rand::thread_rng();
    let start_h = rng.gen_range(0..=(h - crop_h));
    let start_w = rng.gen_range(0..=(w - crop_w));

    image.slice(s![start_h..start_h + crop_h, start_w..start_w + crop_w, ..]).to_owned()
}

/// Random scaling with interpolation
fn random_scale(image: &Image, min_scale: f64, max_scale: f64) -> Image {
    let mut rng = rand::thread_rng();
    let scale = rng.gen_range(min_scale..=max_scale);

    let (h, w, _c) = (image.shape()[0], image.shape()[1], image.shape()[2]);
    let new_h = (h as f64 * scale) as usize;
    let new_w = (w as f64 * scale) as usize;

    resize(image, new_h, new_w)
}

/// Resize image using bilinear interpolation
fn resize(image: &Image, new_h: usize, new_w: usize) -> Image {
    let (h, w, c) = (image.shape()[0], image.shape()[1], image.shape()[2]);
    let mut resized = Array3::<f64>::zeros((new_h, new_w, c));

    let scale_h = h as f64 / new_h as f64;
    let scale_w = w as f64 / new_w as f64;

    for i in 0..new_h {
        for j in 0..new_w {
            let src_y = i as f64 * scale_h;
            let src_x = j as f64 * scale_w;

            let value = bilinear_interpolate(image, src_y, src_x);
            for ch in 0..c {
                resized[[i, j, ch]] = value[ch];
            }
        }
    }

    resized
}

/// Bilinear interpolation for smooth transformations
fn bilinear_interpolate(image: &Image, y: f64, x: f64) -> Vec<f64> {
    let (h, w, c) = (image.shape()[0], image.shape()[1], image.shape()[2]);

    let x0 = x.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y0 = y.floor() as usize;
    let y1 = (y0 + 1).min(h - 1);

    let dx = x - x0 as f64;
    let dy = y - y0 as f64;

    let mut result = vec![0.0; c];
    for ch in 0..c {
        let top = image[[y0, x0, ch]] * (1.0 - dx) + image[[y0, x1, ch]] * dx;
        let bottom = image[[y1, x0, ch]] * (1.0 - dx) + image[[y1, x1, ch]] * dx;
        result[ch] = top * (1.0 - dy) + bottom * dy;
    }

    result
}

// ============================================================================
// Color Space Transformations
// ============================================================================

/// Adjust brightness by adding/subtracting a value
fn adjust_brightness(image: &Image, factor: f64) -> Image {
    (image + factor).mapv(|x| x.max(0.0).min(1.0))
}

/// Adjust contrast by scaling around mean
fn adjust_contrast(image: &Image, factor: f64) -> Image {
    let mean = image.mean().unwrap();
    let adjusted = (image - mean) * factor + mean;
    adjusted.mapv(|x| x.max(0.0).min(1.0))
}

/// Adjust saturation (requires RGB to HSV conversion)
fn adjust_saturation(image: &Image, factor: f64) -> Image {
    let (h, w, c) = (image.shape()[0], image.shape()[1], image.shape()[2]);
    let mut result = image.clone();

    if c != 3 {
        return result;
    }

    for i in 0..h {
        for j in 0..w {
            let r = image[[i, j, 0]];
            let g = image[[i, j, 1]];
            let b = image[[i, j, 2]];

            // Convert RGB to HSV
            let (h_val, s, v) = rgb_to_hsv(r, g, b);

            // Adjust saturation
            let s_new = (s * factor).max(0.0).min(1.0);

            // Convert back to RGB
            let (r_new, g_new, b_new) = hsv_to_rgb(h_val, s_new, v);

            result[[i, j, 0]] = r_new;
            result[[i, j, 1]] = g_new;
            result[[i, j, 2]] = b_new;
        }
    }

    result
}

/// Adjust hue by shifting in HSV color space
fn adjust_hue(image: &Image, shift: f64) -> Image {
    let (h, w, c) = (image.shape()[0], image.shape()[1], image.shape()[2]);
    let mut result = image.clone();

    if c != 3 {
        return result;
    }

    for i in 0..h {
        for j in 0..w {
            let r = image[[i, j, 0]];
            let g = image[[i, j, 1]];
            let b = image[[i, j, 2]];

            let (h_val, s, v) = rgb_to_hsv(r, g, b);

            // Shift hue (wrap around at 1.0)
            let h_new = (h_val + shift) % 1.0;

            let (r_new, g_new, b_new) = hsv_to_rgb(h_new, s, v);

            result[[i, j, 0]] = r_new;
            result[[i, j, 1]] = g_new;
            result[[i, j, 2]] = b_new;
        }
    }

    result
}

/// Apply random color jitter (brightness, contrast, saturation, hue)
fn color_jitter(image: &Image, brightness: f64, contrast: f64,
                saturation: f64, hue: f64) -> Image {
    let mut rng = rand::thread_rng();
    let mut result = image.clone();

    // Randomly apply transformations in random order
    let transforms = [0, 1, 2, 3];

    for &t in &transforms {
        match t {
            0 if brightness > 0.0 => {
                let factor = rng.gen_range(-brightness..=brightness);
                result = adjust_brightness(&result, factor);
            }
            1 if contrast > 0.0 => {
                let factor = rng.gen_range(1.0 - contrast..=1.0 + contrast);
                result = adjust_contrast(&result, factor);
            }
            2 if saturation > 0.0 => {
                let factor = rng.gen_range(1.0 - saturation..=1.0 + saturation);
                result = adjust_saturation(&result, factor);
            }
            3 if hue > 0.0 => {
                let shift = rng.gen_range(-hue..=hue);
                result = adjust_hue(&result, shift);
            }
            _ => {}
        }
    }

    result
}

/// Convert RGB to HSV color space
fn rgb_to_hsv(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;
    let s = if max > 0.0 { delta / max } else { 0.0 };

    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        ((g - b) / delta) % 6.0
    } else if max == g {
        (b - r) / delta + 2.0
    } else {
        (r - g) / delta + 4.0
    };

    let h = (h / 6.0 + 1.0) % 1.0;

    (h, s, v)
}

/// Convert HSV to RGB color space
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    let h = h * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f64;

    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

// ============================================================================
// Advanced Augmentations
// ============================================================================

/// Cutout: Randomly mask rectangular patches
fn cutout(image: &Image, patch_h: usize, patch_w: usize, n_patches: usize) -> Image {
    let mut result = image.clone();
    let (h, w, c) = (image.shape()[0], image.shape()[1], image.shape()[2]);
    let mut rng = rand::thread_rng();

    for _ in 0..n_patches {
        let start_h = rng.gen_range(0..h);
        let start_w = rng.gen_range(0..w);

        let end_h = (start_h + patch_h).min(h);
        let end_w = (start_w + patch_w).min(w);

        // Fill with zeros (or could use mean/random values)
        for i in start_h..end_h {
            for j in start_w..end_w {
                for ch in 0..c {
                    result[[i, j, ch]] = 0.0;
                }
            }
        }
    }

    result
}

/// Add Gaussian noise
fn add_gaussian_noise(image: &Image, std: f64) -> Image {
    let noise = Array3::<f64>::random(image.dim(), Uniform::new(-std, std));
    let noisy = image + noise;
    noisy.mapv(|x| x.max(0.0).min(1.0))
}

/// Gaussian blur (simple box blur approximation)
fn gaussian_blur(image: &Image, sigma: f64) -> Image {
    let (h, w, c) = (image.shape()[0], image.shape()[1], image.shape()[2]);
    let mut blurred = image.clone();

    let kernel_size = (sigma * 3.0).ceil() as usize;
    let kernel_size = if kernel_size % 2 == 0 { kernel_size + 1 } else { kernel_size };
    let radius = kernel_size / 2;

    for ch in 0..c {
        for i in radius..(h - radius) {
            for j in radius..(w - radius) {
                let mut sum = 0.0;
                let mut count = 0.0;

                for di in 0..kernel_size {
                    for dj in 0..kernel_size {
                        let ni = i + di - radius;
                        let nj = j + dj - radius;
                        sum += image[[ni, nj, ch]];
                        count += 1.0;
                    }
                }

                blurred[[i, j, ch]] = sum / count;
            }
        }
    }

    blurred
}

// ============================================================================
// Demonstration Functions
// ============================================================================

/// Demonstrate affine transformation matrices
fn demonstrate_affine_transforms() {
    println!("Affine transformations use 2x3 matrices for geometric operations:\n");

    // Identity (no transformation)
    println!("Identity matrix:");
    println!("  [1  0  0]");
    println!("  [0  1  0]");

    // Translation
    println!("\nTranslation (move 10 pixels right, 5 down):");
    println!("  [1  0  10]");
    println!("  [0  1  5]");

    // Scaling
    println!("\nScaling (2x horizontal, 1.5x vertical):");
    println!("  [2    0   0]");
    println!("  [0   1.5  0]");

    // Rotation (30 degrees)
    let angle = 30.0 * PI / 180.0;
    println!("\nRotation (30°):");
    println!("  [{:.3}  {:.3}  0]", angle.cos(), -angle.sin());
    println!("  [{:.3}  {:.3}  0]", angle.sin(), angle.cos());

    // Shearing
    println!("\nShearing (horizontal):");
    println!("  [1   0.5  0]");
    println!("  [0    1   0]");

    println!("\nThese matrices can be combined through multiplication!");
}

/// Demonstrate Mixup augmentation
fn demonstrate_mixup() {
    let img1 = create_test_image(32, 32);
    let img2 = Array3::<f64>::ones((32, 32, 3)) * 0.5;

    let lambda = 0.6;
    let mixed = &img1 * lambda + &img2 * (1.0 - lambda);

    println!("Mixup combines two images with weighted average:");
    println!("  mixed = λ × image1 + (1-λ) × image2");
    println!("  With λ = {:.1}, we get {:.0}% of img1 and {:.0}% of img2",
             lambda, lambda * 100.0, (1.0 - lambda) * 100.0);
    print_image_stats("Mixed", &mixed);

    println!("\nLabels are also mixed:");
    println!("  If img1 is class 'cat' and img2 is class 'dog'");
    println!("  Mixed label = [cat: 0.6, dog: 0.4]");
    println!("  This creates smoother decision boundaries!");
}

/// Analyze augmentation impact on model training
fn analyze_augmentation_impact() {
    println!("Typical performance improvements:\n");

    let scenarios = [
        ("ImageNet (1M images)", "Moderate aug", "2-5%"),
        ("Medical imaging (100s)", "Heavy aug", "10-20%"),
        ("Small dataset (1000s)", "Heavy aug", "5-15%"),
        ("Text classification", "Light aug", "1-3%"),
        ("Speech recognition", "SpecAugment", "10-20%"),
    ];

    for (dataset, aug_type, improvement) in &scenarios {
        println!("  {:25} {:15} → {} accuracy gain", dataset, aug_type, improvement);
    }

    println!("\nComputation vs. Benefit trade-off:");
    println!("  Flip/Crop:      Negligible cost, 3-5% gain");
    println!("  Color jitter:   Very low cost, 1-3% gain");
    println!("  Cutout/Mixup:   Low cost, 1-3% gain");
    println!("  AutoAugment:    High cost (search), 1-2% extra gain");
    println!("\nConclusion: Start simple, add complexity only if needed!");
}

/// Demonstrate complete augmentation pipeline
fn demonstrate_pipeline(image: &Image) {
    println!("Typical training pipeline applies multiple augmentations:\n");

    println!("1. Random crop (for scale/position variance)");
    let mut aug = random_crop(image, 56, 56);

    println!("2. Random horizontal flip (50% probability)");
    if rand::random::<bool>() {
        aug = horizontal_flip(&aug);
    }

    println!("3. Color jitter (brightness, contrast, saturation)");
    aug = color_jitter(&aug, 0.2, 0.2, 0.2, 0.1);

    println!("4. Random rotation (±15°)");
    let angle = rand::thread_rng().gen_range(-15.0..=15.0);
    aug = rotate(&aug, angle);

    println!("5. Cutout (regularization)");
    aug = cutout(&aug, 12, 12, 1);

    println!("\nFinal augmented image:");
    print_image_stats("Augmented", &aug);

    println!("\nFor validation/testing: Use deterministic preprocessing");
    println!("  - Center crop (not random)");
    println!("  - No flip/rotation");
    println!("  - No color jitter");
    println!("  - Only normalization");
}

// ============================================================================
// Utility Functions
// ============================================================================

fn print_image_stats(name: &str, image: &Image) {
    let min = image.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = image.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = image.mean().unwrap();

    println!("  {} - shape: {:?}, range: [{:.3}, {:.3}], mean: {:.3}",
             name, image.shape(), min, max, mean);
}

fn print_color_stats(name: &str, image: &Image) {
    if image.shape()[2] != 3 {
        return;
    }

    let r_mean = image.slice(s![.., .., 0]).mean().unwrap();
    let g_mean = image.slice(s![.., .., 1]).mean().unwrap();
    let b_mean = image.slice(s![.., .., 2]).mean().unwrap();

    println!("  {} - RGB means: [{:.3}, {:.3}, {:.3}]", name, r_mean, g_mean, b_mean);
}

fn print_summary() {
    println!("Key Takeaways:\n");
    println!("1. EASY WINS:");
    println!("   • Flip + Crop = 5% accuracy gain for almost free");
    println!("   • Color jitter = 2-3% gain, trivial to implement");
    println!("   • Always use these for computer vision tasks");

    println!("\n2. ADVANCED TECHNIQUES:");
    println!("   • Cutout/Mixup = Extra 1-3% for competitions");
    println!("   • AutoAugment/RandAugment = Final 1-2% but costly");
    println!("   • Use when you need every bit of performance");

    println!("\n3. BEST PRACTICES:");
    println!("   • Start simple, measure impact, add complexity");
    println!("   • Match augmentations to your domain (medical ≠ natural images)");
    println!("   • Don't augment validation/test sets (except TTA)");
    println!("   • Visualize augmented samples to verify sanity");

    println!("\n4. WHY IT WORKS:");
    println!("   • Expands dataset (more training examples)");
    println!("   • Regularization (prevents overfitting)");
    println!("   • Invariance learning (cat is cat, regardless of position)");

    println!("\n5. REAL-WORLD IMPACT:");
    println!("   • AlexNet (ImageNet 2012): Used flip + crop to win");
    println!("   • Modern CNNs: 10-15% worse without augmentation");
    println!("   • Medical imaging: Enables training with 10x less data");

    println!("\nData augmentation is the highest ROI technique in ML!");
    println!("Implement it first, before trying bigger models or more data.");
}

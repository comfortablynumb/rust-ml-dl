//! # Convolutional Neural Network (CNN) Example
//!
//! This example demonstrates the fundamental concepts of Convolutional Neural Networks,
//! the architecture that revolutionized computer vision.
//!
//! ## What is a CNN?
//!
//! Convolutional Neural Networks are specialized neural networks designed to process
//! grid-like data such as images. They use convolution operations to automatically learn
//! spatial hierarchies of features.
//!
//! ## Why CNNs for Images?
//!
//! Traditional neural networks don't work well for images because:
//! - **Too many parameters**: A 200x200 RGB image has 120,000 inputs!
//! - **No spatial structure**: Fully connected layers ignore that nearby pixels are related
//! - **Not translation invariant**: Must relearn features at every position
//!
//! CNNs solve these problems through:
//! 1. **Local connectivity**: Neurons connect only to nearby pixels
//! 2. **Parameter sharing**: Same filter applies across the entire image
//! 3. **Translation invariance**: Features learned once work everywhere
//!
//! ## Key Components
//!
//! ### 1. Convolutional Layer
//!
//! Applies filters (kernels) to detect features like edges, textures, and patterns.
//!
//! **Operation:**
//! ```
//! Input: 28×28 image
//! Filter: 3×3 kernel
//! Stride: 1
//! Output: 26×26 feature map
//! ```
//!
//! **Example 3×3 edge detection filter:**
//! ```
//! [-1, -1, -1]
//! [ 0,  0,  0]
//! [ 1,  1,  1]
//! ```
//!
//! **Convolution Math:**
//! ```
//! Output[i,j] = Σ Σ Input[i+m, j+n] × Filter[m,n]
//! ```
//!
//! ### 2. Pooling Layer
//!
//! Reduces spatial dimensions while retaining important features.
//!
//! **Max Pooling (2×2):**
//! ```
//! Input:           Output:
//! [1  3]  [2  4]    [3  4]
//! [0  2]  [1  3]    [5  8]
//!
//! [4  1]  [2  3]
//! [5  0]  [8  1]
//! ```
//! Takes maximum value in each 2×2 region.
//!
//! **Benefits:**
//! - Reduces parameters and computation
//! - Provides translation invariance
//! - Controls overfitting
//!
//! ### 3. Typical CNN Architecture
//!
//! ```
//! Input Image (28×28×1)
//!     ↓
//! Conv1 (26×26×32) + ReLU
//!     ↓
//! MaxPool (13×13×32)
//!     ↓
//! Conv2 (11×11×64) + ReLU
//!     ↓
//! MaxPool (5×5×64)
//!     ↓
//! Flatten (1600)
//!     ↓
//! Dense (128) + ReLU
//!     ↓
//! Output (10) + Softmax
//! ```
//!
//! ## Feature Hierarchy
//!
//! CNNs learn features at different levels:
//!
//! **Layer 1 (Early):**
//! - Edges (horizontal, vertical, diagonal)
//! - Colors and gradients
//! - Simple textures
//!
//! **Layer 2-3 (Middle):**
//! - Corners and curves
//! - Simple shapes (circles, rectangles)
//! - Textures and patterns
//!
//! **Layer 4-5 (Deep):**
//! - Object parts (eyes, wheels, windows)
//! - Complex patterns
//!
//! **Final Layers:**
//! - Complete objects (faces, cars, cats)
//! - Scene understanding
//!
//! ## Key Hyperparameters
//!
//! **Filter Size:**
//! - 3×3: Most common, captures local patterns
//! - 5×5: Larger receptive field
//! - 1×1: Reduces channels, adds non-linearity
//!
//! **Stride:**
//! - 1: Maximum overlap, preserves information
//! - 2: Reduces dimensions by half
//!
//! **Padding:**
//! - Valid (no padding): Output smaller than input
//! - Same (zero padding): Output same size as input
//!
//! ## Famous CNN Architectures
//!
//! - **LeNet-5 (1998)**: First successful CNN, digit recognition
//! - **AlexNet (2012)**: Won ImageNet, started deep learning boom
//! - **VGG (2014)**: Very deep, simple 3×3 filters
//! - **ResNet (2015)**: Skip connections, 100+ layers
//! - **Inception (2014)**: Multi-scale feature extraction
//! - **EfficientNet (2019)**: Optimal scaling
//!
//! ## Applications
//!
//! - **Image Classification**: Cat vs Dog, digit recognition
//! - **Object Detection**: YOLO, Faster R-CNN
//! - **Segmentation**: Medical imaging, self-driving cars
//! - **Face Recognition**: Unlock phones, security
//! - **Style Transfer**: Turn photos into paintings
//! - **Image Generation**: GANs, diffusion models

use ndarray::{Array1, Array2, Array3, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Uniform, StandardNormal};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Simple 2D convolution operation
fn convolve2d(input: &Array2<f64>, kernel: &Array2<f64>, stride: usize) -> Array2<f64> {
    let (in_h, in_w) = input.dim();
    let (k_h, k_w) = kernel.dim();

    let out_h = (in_h - k_h) / stride + 1;
    let out_w = (in_w - k_w) / stride + 1;

    let mut output = Array2::zeros((out_h, out_w));

    for i in 0..out_h {
        for j in 0..out_w {
            let mut sum = 0.0;
            for ki in 0..k_h {
                for kj in 0..k_w {
                    sum += input[[i * stride + ki, j * stride + kj]] * kernel[[ki, kj]];
                }
            }
            output[[i, j]] = sum;
        }
    }

    output
}

/// Max pooling operation
fn max_pool2d(input: &Array2<f64>, pool_size: usize) -> Array2<f64> {
    let (in_h, in_w) = input.dim();
    let out_h = in_h / pool_size;
    let out_w = in_w / pool_size;

    let mut output = Array2::zeros((out_h, out_w));

    for i in 0..out_h {
        for j in 0..out_w {
            let mut max_val = f64::NEG_INFINITY;
            for pi in 0..pool_size {
                for pj in 0..pool_size {
                    let val = input[[i * pool_size + pi, j * pool_size + pj]];
                    if val > max_val {
                        max_val = val;
                    }
                }
            }
            output[[i, j]] = max_val;
        }
    }

    output
}

/// ReLU activation
fn relu(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| v.max(0.0))
}

fn main() -> anyhow::Result<()> {
    println!("=== Convolutional Neural Network (CNN) Basics ===\n");

    let mut rng = StdRng::seed_from_u64(42);

    // Create a simple 8x8 "image" with a vertical edge
    println!("1. Creating a simple image with features\n");

    let mut image = Array2::zeros((8, 8));
    // Left half: light (1.0), right half: dark (0.0)
    for i in 0..8 {
        for j in 0..4 {
            image[[i, j]] = 1.0;
        }
    }

    println!("   Original 8×8 image (vertical edge):");
    println!("   (1.0 = white, 0.0 = black)\n");
    for i in 0..8 {
        print!("   ");
        for j in 0..8 {
            print!("{:.1} ", image[[i, j]]);
        }
        println!();
    }

    // Define edge detection filters
    println!("\n2. Applying Convolution Filters\n");

    // Vertical edge detector
    let vertical_filter = Array2::from_shape_vec(
        (3, 3),
        vec![
            -1.0, 0.0, 1.0,
            -1.0, 0.0, 1.0,
            -1.0, 0.0, 1.0,
        ],
    )?;

    println!("   A) Vertical Edge Detector Filter (3×3):");
    for i in 0..3 {
        print!("      ");
        for j in 0..3 {
            print!("{:4.1} ", vertical_filter[[i, j]]);
        }
        println!();
    }

    // Apply convolution
    let conv_result = convolve2d(&image, &vertical_filter, 1);

    println!("\n   Convolution Output (6×6):");
    println!("   (Shows detected vertical edges)\n");
    for i in 0..conv_result.nrows() {
        print!("   ");
        for j in 0..conv_result.ncols() {
            print!("{:5.1} ", conv_result[[i, j]]);
        }
        println!();
    }

    println!("\n   Notice: High values (3.0) indicate strong vertical edge!");

    // Horizontal edge detector
    let horizontal_filter = Array2::from_shape_vec(
        (3, 3),
        vec![
            -1.0, -1.0, -1.0,
             0.0,  0.0,  0.0,
             1.0,  1.0,  1.0,
        ],
    )?;

    println!("\n   B) Horizontal Edge Detector Filter (3×3):");
    for i in 0..3 {
        print!("      ");
        for j in 0..3 {
            print!("{:4.1} ", horizontal_filter[[i, j]]);
        }
        println!();
    }

    let h_conv_result = convolve2d(&image, &horizontal_filter, 1);

    println!("\n   Convolution Output (6×6):");
    println!("   (No horizontal edges detected)\n");
    for i in 0..h_conv_result.nrows() {
        print!("   ");
        for j in 0..h_conv_result.ncols() {
            print!("{:5.1} ", h_conv_result[[i, j]]);
        }
        println!();
    }

    println!("\n   Notice: All zeros - no horizontal edges in this image!");

    // ReLU activation
    println!("\n3. Applying ReLU Activation\n");

    let activated = relu(&conv_result);
    println!("   After ReLU (removes negative values):");
    for i in 0..activated.nrows().min(4) {
        print!("   ");
        for j in 0..activated.ncols() {
            print!("{:5.1} ", activated[[i, j]]);
        }
        println!();
    }
    println!("   ...");

    // Max pooling
    println!("\n4. Max Pooling (2×2)\n");

    let pooled = max_pool2d(&activated, 2);

    println!("   After Max Pooling (6×6 → 3×3):");
    println!("   (Takes maximum value from each 2×2 region)\n");
    for i in 0..pooled.nrows() {
        print!("   ");
        for j in 0..pooled.ncols() {
            print!("{:5.1} ", pooled[[i, j]]);
        }
        println!();
    }

    println!("\n   Benefits:");
    println!("   - Reduced from 6×6 (36 values) to 3×3 (9 values)");
    println!("   - 75% reduction in size");
    println!("   - Kept the most important information!");

    // Demonstrate feature learning on random image
    println!("\n5. Multiple Filters (Multiple Feature Maps)\n");

    let random_image = Array2::random_using((8, 8), Uniform::new(0.0, 1.0), &mut rng);

    println!("   In practice, CNNs use many filters (32, 64, 128+)");
    println!("   Each filter learns different features:");
    println!("   - Filter 1: Vertical edges");
    println!("   - Filter 2: Horizontal edges  ");
    println!("   - Filter 3: Diagonal edges");
    println!("   - Filter 4: Textures");
    println!("   - ... and so on\n");

    // Typical CNN architecture summary
    println!("6. Typical CNN Architecture for Image Classification\n");
    println!("   Input: 28×28×1 grayscale image");
    println!("     ↓");
    println!("   Conv Layer 1: 32 filters (3×3) → 26×26×32");
    println!("   ReLU Activation");
    println!("     ↓");
    println!("   Max Pooling (2×2) → 13×13×32");
    println!("     ↓");
    println!("   Conv Layer 2: 64 filters (3×3) → 11×11×64");
    println!("   ReLU Activation");
    println!("     ↓");
    println!("   Max Pooling (2×2) → 5×5×64");
    println!("     ↓");
    println!("   Flatten → 1600 neurons");
    println!("     ↓");
    println!("   Dense Layer: 128 neurons + ReLU");
    println!("     ↓");
    println!("   Output: 10 classes + Softmax");

    println!("\n7. Why CNNs Work So Well\n");
    println!("   ✓ Local Connectivity: Focuses on nearby pixels");
    println!("   ✓ Parameter Sharing: Same filter across entire image");
    println!("   ✓ Translation Invariance: Detects features anywhere");
    println!("   ✓ Hierarchical Learning: Simple → Complex features");
    println!("   ✓ Dimensionality Reduction: Pooling reduces overfitting");

    println!("\n8. Key Differences from Regular Neural Networks\n");
    println!("   Regular NN               CNN");
    println!("   -----------             -----");
    println!("   Fully connected         Local connections");
    println!("   Millions of params      Thousands of params");
    println!("   No spatial structure    Preserves spatial info");
    println!("   Slow on images          Fast on images");
    println!("   Easy to overfit         Better generalization");

    println!("\n=== Example Complete! ===");
    println!("\nKey Takeaways:");
    println!("- Convolution detects features using small filters");
    println!("- Pooling reduces dimensions while keeping important info");
    println!("- Multiple filters learn different features");
    println!("- Deep layers learn increasingly complex patterns");
    println!("- CNNs are the standard for computer vision tasks");

    Ok(())
}

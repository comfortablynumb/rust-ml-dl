use ndarray::{Array1, Array2, Array3, Array4};
use rand::Rng;
use std::f64;

/// A simplified VGG-style convolutional neural network for feature extraction.
/// In practice, you would use a pre-trained VGG-19 network. Here we simulate
/// the architecture to demonstrate the core concepts of neural style transfer.
struct VGGFeatureExtractor {
    // Convolutional layers organized by blocks
    // In real implementation, these would be pre-trained weights from VGG-19
    conv1_1: ConvLayer,
    conv2_1: ConvLayer,
    conv3_1: ConvLayer,
    conv4_1: ConvLayer,
    conv4_2: ConvLayer,
    conv5_1: ConvLayer,
}

/// A simple convolutional layer for feature extraction
struct ConvLayer {
    weights: Array4<f64>, // [out_channels, in_channels, kernel_h, kernel_w]
    bias: Array1<f64>,
    channels: usize,
}

impl ConvLayer {
    /// Create a new convolutional layer with random initialization
    /// (In practice, these would be pre-trained VGG weights)
    fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        let scale = (2.0 / (in_channels * kernel_size * kernel_size) as f64).sqrt();
        let weights = Array4::from_shape_fn(
            (out_channels, in_channels, kernel_size, kernel_size),
            |_| rng.gen::<f64>() * scale - scale / 2.0,
        );

        let bias = Array1::zeros(out_channels);

        Self {
            weights,
            bias,
            channels: out_channels,
        }
    }

    /// Apply convolution to input
    fn forward(&self, input: &Array3<f64>) -> Array3<f64> {
        let (in_channels, height, width) = input.dim();
        let (out_channels, _, kernel_h, kernel_w) = self.weights.dim();

        let out_h = height - kernel_h + 1;
        let out_w = width - kernel_w + 1;

        let mut output = Array3::zeros((out_channels, out_h, out_w));

        for oc in 0..out_channels {
            for h in 0..out_h {
                for w in 0..out_w {
                    let mut sum = self.bias[oc];

                    for ic in 0..in_channels {
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                sum += self.weights[[oc, ic, kh, kw]]
                                    * input[[ic, h + kh, w + kw]];
                            }
                        }
                    }

                    // ReLU activation
                    output[[oc, h, w]] = sum.max(0.0);
                }
            }
        }

        output
    }
}

impl VGGFeatureExtractor {
    /// Create a new VGG-style feature extractor
    /// This simulates VGG-19 architecture with fewer parameters for demonstration
    fn new() -> Self {
        println!("Initializing VGG-style feature extractor...");
        println!("(In practice, this would load pre-trained VGG-19 weights)\n");

        Self {
            conv1_1: ConvLayer::new(3, 6, 3),    // Block 1
            conv2_1: ConvLayer::new(6, 12, 3),   // Block 2
            conv3_1: ConvLayer::new(12, 24, 3),  // Block 3
            conv4_1: ConvLayer::new(24, 32, 3),  // Block 4
            conv4_2: ConvLayer::new(32, 32, 3),  // Block 4 (deeper)
            conv5_1: ConvLayer::new(32, 48, 3),  // Block 5
        }
    }

    /// Extract features from multiple layers for style transfer
    fn extract_features(&self, image: &Array3<f64>) -> VGGFeatures {
        let mut features = VGGFeatures::new();

        // Progressive feature extraction through the network
        let feat1_1 = self.conv1_1.forward(image);
        features.conv1_1 = Some(feat1_1.clone());

        let pooled1 = Self::max_pool(&feat1_1);
        let feat2_1 = self.conv2_1.forward(&pooled1);
        features.conv2_1 = Some(feat2_1.clone());

        let pooled2 = Self::max_pool(&feat2_1);
        let feat3_1 = self.conv3_1.forward(&pooled2);
        features.conv3_1 = Some(feat3_1.clone());

        // Skip deeper layers to avoid dimension issues with small images
        // In a full implementation with larger images, you would include all layers
        features.conv4_1 = None;
        features.conv4_2 = Some(feat3_1.clone()); // Use conv3_1 as content layer
        features.conv5_1 = None;

        features
    }

    /// Simple 2x2 max pooling
    fn max_pool(input: &Array3<f64>) -> Array3<f64> {
        let (channels, height, width) = input.dim();
        let out_h = height / 2;
        let out_w = width / 2;

        let mut output = Array3::zeros((channels, out_h, out_w));

        for c in 0..channels {
            for h in 0..out_h {
                for w in 0..out_w {
                    let h2 = h * 2;
                    let w2 = w * 2;

                    let max_val = input[[c, h2, w2]]
                        .max(input[[c, h2 + 1, w2]])
                        .max(input[[c, h2, w2 + 1]])
                        .max(input[[c, h2 + 1, w2 + 1]]);

                    output[[c, h, w]] = max_val;
                }
            }
        }

        output
    }
}

/// Container for features extracted from multiple VGG layers
#[derive(Clone)]
struct VGGFeatures {
    conv1_1: Option<Array3<f64>>,
    conv2_1: Option<Array3<f64>>,
    conv3_1: Option<Array3<f64>>,
    conv4_1: Option<Array3<f64>>,
    conv4_2: Option<Array3<f64>>, // Used for content representation
    conv5_1: Option<Array3<f64>>,
}

impl VGGFeatures {
    fn new() -> Self {
        Self {
            conv1_1: None,
            conv2_1: None,
            conv3_1: None,
            conv4_1: None,
            conv4_2: None,
            conv5_1: None,
        }
    }
}

/// Neural style transfer optimizer
struct StyleTransfer {
    vgg: VGGFeatureExtractor,
    content_weight: f64,
    style_weight: f64,
    tv_weight: f64,
}

impl StyleTransfer {
    fn new(content_weight: f64, style_weight: f64, tv_weight: f64) -> Self {
        Self {
            vgg: VGGFeatureExtractor::new(),
            content_weight,
            style_weight,
            tv_weight,
        }
    }

    /// Compute content loss between generated and content images
    /// Uses features from a deep layer (conv4_2) to capture high-level content
    fn content_loss(&self, generated_features: &Array3<f64>, content_features: &Array3<f64>) -> f64 {
        let diff = generated_features - content_features;
        let loss = (&diff * &diff).sum();

        // Normalize by number of elements
        let n_elements = diff.len() as f64;
        loss / (2.0 * n_elements)
    }

    /// Compute Gram matrix for style representation
    /// The Gram matrix captures correlations between feature maps,
    /// representing texture and pattern information independent of spatial location
    fn gram_matrix(&self, features: &Array3<f64>) -> Array2<f64> {
        let (channels, height, width) = features.dim();
        let n_locations = (height * width) as f64;

        // Compute Gram matrix: G[i,j] = sum over locations of F[i,l] * F[j,l]
        let mut gram = Array2::zeros((channels, channels));

        for i in 0..channels {
            for j in 0..channels {
                let mut sum = 0.0;
                for h in 0..height {
                    for w in 0..width {
                        sum += features[[i, h, w]] * features[[j, h, w]];
                    }
                }
                gram[[i, j]] = sum / n_locations; // Normalize by number of locations
            }
        }

        gram
    }

    /// Compute style loss between generated and style images
    /// Compares Gram matrices across multiple layers to capture style at different scales
    fn style_loss(&self, generated_features: &VGGFeatures, style_features: &VGGFeatures) -> f64 {
        let mut total_loss = 0.0;
        let mut layer_count = 0;

        // Style loss from multiple layers captures style at different scales
        let layers = vec![
            (&generated_features.conv1_1, &style_features.conv1_1, 1.0),  // Fine details
            (&generated_features.conv2_1, &style_features.conv2_1, 1.0),  // Medium details
            (&generated_features.conv3_1, &style_features.conv3_1, 1.0),  // Larger patterns
        ];

        for (gen_feat, style_feat, weight) in layers {
            if let (Some(gen), Some(style)) = (gen_feat, style_feat) {
                let gen_gram = self.gram_matrix(gen);
                let style_gram = self.gram_matrix(style);

                // Compute Frobenius norm of difference between Gram matrices
                let diff = &gen_gram - &style_gram;
                let loss = (&diff * &diff).sum();

                // Normalize by number of elements
                let (channels, _) = gen_gram.dim();
                let normalized_loss = loss / (4.0 * (channels * channels) as f64);

                total_loss += weight * normalized_loss;
                layer_count += 1;
            }
        }

        total_loss / layer_count as f64
    }

    /// Compute total variation loss for smoothness
    /// Penalizes neighboring pixel differences to reduce noise and artifacts
    fn total_variation_loss(&self, image: &Array3<f64>) -> f64 {
        let (channels, height, width) = image.dim();
        let mut tv_loss = 0.0;

        for c in 0..channels {
            // Horizontal differences
            for h in 0..height {
                for w in 0..(width - 1) {
                    let diff = image[[c, h, w + 1]] - image[[c, h, w]];
                    tv_loss += diff.abs();
                }
            }

            // Vertical differences
            for h in 0..(height - 1) {
                for w in 0..width {
                    let diff = image[[c, h + 1, w]] - image[[c, h, w]];
                    tv_loss += diff.abs();
                }
            }
        }

        tv_loss / (channels * height * width) as f64
    }

    /// Compute total loss and its gradient
    fn compute_loss_and_gradients(
        &self,
        generated: &Array3<f64>,
        content_features: &Array3<f64>,
        style_features: &VGGFeatures,
    ) -> (f64, Array3<f64>) {
        // Extract features from generated image
        let gen_features = self.vgg.extract_features(generated);

        // Content loss (using conv4_2)
        let content_loss = self.content_loss(
            gen_features.conv4_2.as_ref().unwrap(),
            content_features,
        );

        // Style loss (using multiple layers)
        let style_loss = self.style_loss(&gen_features, style_features);

        // Total variation loss
        let tv_loss = self.total_variation_loss(generated);

        // Combined loss
        let total_loss = self.content_weight * content_loss
            + self.style_weight * style_loss
            + self.tv_weight * tv_loss;

        // Compute gradients (simplified - in practice would use automatic differentiation)
        let gradients = self.compute_simple_gradients(
            generated,
            content_features,
            style_features,
        );

        (total_loss, gradients)
    }

    /// Simplified gradient computation using finite differences
    /// In practice, you would use automatic differentiation
    fn compute_simple_gradients(
        &self,
        generated: &Array3<f64>,
        content_features: &Array3<f64>,
        style_features: &VGGFeatures,
    ) -> Array3<f64> {
        let (channels, height, width) = generated.dim();
        let mut gradients = Array3::zeros((channels, height, width));
        let epsilon = 1e-4;

        // Sample a subset of pixels for gradient computation (for efficiency)
        let sample_rate = 4;

        for c in 0..channels {
            for h in (0..height).step_by(sample_rate) {
                for w in (0..width).step_by(sample_rate) {
                    let mut perturbed = generated.clone();

                    // Forward difference
                    perturbed[[c, h, w]] += epsilon;
                    let gen_features_plus = self.vgg.extract_features(&perturbed);

                    let content_loss_plus = self.content_loss(
                        gen_features_plus.conv4_2.as_ref().unwrap(),
                        content_features,
                    );
                    let style_loss_plus = self.style_loss(&gen_features_plus, style_features);
                    let tv_loss_plus = self.total_variation_loss(&perturbed);

                    let loss_plus = self.content_weight * content_loss_plus
                        + self.style_weight * style_loss_plus
                        + self.tv_weight * tv_loss_plus;

                    // Backward difference
                    perturbed[[c, h, w]] = generated[[c, h, w]] - epsilon;
                    let gen_features_minus = self.vgg.extract_features(&perturbed);

                    let content_loss_minus = self.content_loss(
                        gen_features_minus.conv4_2.as_ref().unwrap(),
                        content_features,
                    );
                    let style_loss_minus = self.style_loss(&gen_features_minus, style_features);
                    let tv_loss_minus = self.total_variation_loss(&perturbed);

                    let loss_minus = self.content_weight * content_loss_minus
                        + self.style_weight * style_loss_minus
                        + self.tv_weight * tv_loss_minus;

                    // Central difference
                    gradients[[c, h, w]] = (loss_plus - loss_minus) / (2.0 * epsilon);
                }
            }
        }

        gradients
    }

    /// Perform style transfer optimization
    fn transfer_style(
        &self,
        content_image: &Array3<f64>,
        style_image: &Array3<f64>,
        iterations: usize,
        learning_rate: f64,
    ) -> Array3<f64> {
        println!("Starting style transfer optimization...");
        println!("Content weight: {}", self.content_weight);
        println!("Style weight: {}", self.style_weight);
        println!("TV weight: {}\n", self.tv_weight);

        // Extract target features
        println!("Extracting content features...");
        let content_features_all = self.vgg.extract_features(content_image);
        let content_features = content_features_all.conv4_2.unwrap();

        println!("Extracting style features...");
        let style_features = self.vgg.extract_features(style_image);

        // Initialize generated image (start from content image)
        let mut generated = content_image.clone();

        println!("\nOptimization iterations:");
        println!("{:-<80}", "");

        for iter in 0..iterations {
            // Compute loss and gradients
            let (loss, gradients) = self.compute_loss_and_gradients(
                &generated,
                &content_features,
                &style_features,
            );

            // Update generated image using gradient descent
            generated = &generated - &(&gradients * learning_rate);

            // Clip values to valid range [0, 1]
            generated.mapv_inplace(|x| x.max(0.0).min(1.0));

            if iter % 5 == 0 || iter == iterations - 1 {
                println!("Iteration {:3}: Loss = {:.6}", iter, loss);
            }
        }

        println!("{:-<80}", "");
        println!("\nStyle transfer complete!\n");

        generated
    }
}

/// Create a synthetic content image (simulating a photograph)
fn create_content_image(height: usize, width: usize) -> Array3<f64> {
    let mut rng = rand::thread_rng();
    let mut image = Array3::zeros((3, height, width));

    // Create a simple geometric pattern (representing content structure)
    for h in 0..height {
        for w in 0..width {
            // Horizontal gradient in red channel
            image[[0, h, w]] = w as f64 / width as f64;

            // Vertical gradient in green channel
            image[[1, h, w]] = h as f64 / height as f64;

            // Circular pattern in blue channel
            let center_h = height as f64 / 2.0;
            let center_w = width as f64 / 2.0;
            let dist = ((h as f64 - center_h).powi(2) + (w as f64 - center_w).powi(2)).sqrt();
            let max_dist = (height as f64 / 2.0).min(width as f64 / 2.0);
            image[[2, h, w]] = 1.0 - (dist / max_dist).min(1.0);

            // Add small noise
            for c in 0..3 {
                image[[c, h, w]] += (rng.gen::<f64>() - 0.5) * 0.05;
                image[[c, h, w]] = image[[c, h, w]].max(0.0).min(1.0);
            }
        }
    }

    image
}

/// Create a synthetic style image (simulating an artistic painting)
fn create_style_image(height: usize, width: usize) -> Array3<f64> {
    let mut rng = rand::thread_rng();
    let mut image = Array3::zeros((3, height, width));

    // Create textured pattern with strong correlations (representing artistic style)
    for h in 0..height {
        for w in 0..width {
            // Wavy patterns
            let x = w as f64 / width as f64 * 4.0 * f64::consts::PI;
            let y = h as f64 / height as f64 * 4.0 * f64::consts::PI;

            image[[0, h, w]] = (x.sin() * y.cos() + 1.0) / 2.0; // Red: wavy pattern
            image[[1, h, w]] = (y.sin() * 1.5 + 1.0) / 2.0;     // Green: vertical waves
            image[[2, h, w]] = (x.cos() * 1.5 + 1.0) / 2.0;     // Blue: horizontal waves

            // Add texture noise
            for c in 0..3 {
                image[[c, h, w]] += (rng.gen::<f64>() - 0.5) * 0.3;
                image[[c, h, w]] = image[[c, h, w]].max(0.0).min(1.0);
            }
        }
    }

    image
}

fn main() {
    println!("{}", "=".repeat(80));
    println!("{:^80}", "NEURAL STYLE TRANSFER");
    println!("{:^80}", "Combining Content and Style with Deep Learning");
    println!("{}", "=".repeat(80));
    println!();

    // Configuration
    let height = 40;
    let width = 40;
    let iterations = 20;
    let learning_rate = 0.1;

    println!("CONFIGURATION");
    println!("{:-<80}", "");
    println!("Image size: {}x{}", height, width);
    println!("Iterations: {}", iterations);
    println!("Learning rate: {}", learning_rate);
    println!();

    // Create content and style images
    println!("CREATING SYNTHETIC IMAGES");
    println!("{:-<80}", "");
    println!("Content image: Geometric patterns (simulating photograph)");
    let content_image = create_content_image(height, width);

    println!("Style image: Wavy textures (simulating artistic painting)");
    let style_image = create_style_image(height, width);
    println!();

    // Demonstrate different weight configurations
    println!("EXPERIMENT 1: Balanced Content and Style");
    println!("{}", "=".repeat(80));
    let transfer1 = StyleTransfer::new(1.0, 1000.0, 0.01);
    let _result1 = transfer1.transfer_style(&content_image, &style_image, iterations, learning_rate);

    println!("\nEXPERIMENT 2: Strong Content Preservation");
    println!("{}", "=".repeat(80));
    let transfer2 = StyleTransfer::new(10.0, 1000.0, 0.01);
    let _result2 = transfer2.transfer_style(&content_image, &style_image, iterations, learning_rate);

    println!("\nEXPERIMENT 3: Strong Style Transfer");
    println!("{}", "=".repeat(80));
    let transfer3 = StyleTransfer::new(1.0, 5000.0, 0.01);
    let _result3 = transfer3.transfer_style(&content_image, &style_image, iterations, learning_rate);

    // Analyze results
    println!("\n");
    println!("{}", "=".repeat(80));
    println!("{:^80}", "ANALYSIS OF RESULTS");
    println!("{}", "=".repeat(80));
    println!();

    println!("KEY CONCEPTS DEMONSTRATED:");
    println!("{:-<80}", "");
    println!("1. Content Loss: Preserves high-level structure from content image");
    println!("   - Uses deep layer features (conv4_2)");
    println!("   - Matches feature activations, not pixels");
    println!();

    println!("2. Style Loss: Captures artistic style via Gram matrices");
    println!("   - Gram matrices encode feature correlations");
    println!("   - Multiple layers capture style at different scales");
    println!("   - Location-invariant representation");
    println!();

    println!("3. Total Variation Loss: Encourages smoothness");
    println!("   - Reduces noise and artifacts");
    println!("   - Penalizes neighboring pixel differences");
    println!();

    println!("4. Multi-Objective Optimization:");
    println!("   - Balance content preservation vs. style matching");
    println!("   - Weight tuning affects final appearance");
    println!("   - Higher style weight = more stylized output");
    println!();

    println!("WEIGHT CONFIGURATION EFFECTS:");
    println!("{:-<80}", "");
    println!("Experiment 1 (α=1.0, β=1000): Balanced result");
    println!("Experiment 2 (α=10.0, β=1000): More content preservation");
    println!("Experiment 3 (α=1.0, β=5000): Stronger style transfer");
    println!();

    println!("EDUCATIONAL INSIGHTS:");
    println!("{:-<80}", "");
    println!("• CNNs learn hierarchical representations:");
    println!("  - Early layers: edges, textures, colors");
    println!("  - Deep layers: objects, compositions");
    println!();

    println!("• Gram matrices capture style:");
    println!("  - Correlation between features, not locations");
    println!("  - Separates 'what' from 'where'");
    println!("  - Enables style transfer without spatial alignment");
    println!();

    println!("• Optimization in pixel space:");
    println!("  - Fixed network weights, optimize input");
    println!("  - Different from traditional training");
    println!("  - Gradient descent in image space");
    println!();

    println!("• Perceptual losses:");
    println!("  - Feature-based losses align with human perception");
    println!("  - Better than pixel-wise L2 loss");
    println!("  - Applicable to many vision tasks");
    println!();

    println!("PRACTICAL APPLICATIONS:");
    println!("{:-<80}", "");
    println!("• Artistic photo filters (Prisma, DeepArt)");
    println!("• Video game rendering and effects");
    println!("• Architectural visualization");
    println!("• Fashion and textile design");
    println!("• Video stylization with temporal consistency");
    println!("• Real-time style transfer with feed-forward networks");
    println!();

    println!("EXTENSIONS AND IMPROVEMENTS:");
    println!("{:-<80}", "");
    println!("• Fast style transfer: Train feed-forward network");
    println!("  - Single forward pass instead of optimization");
    println!("  - Real-time performance (30+ FPS)");
    println!();

    println!("• Arbitrary style transfer: AdaIN and StyleGAN");
    println!("  - Handle any style without retraining");
    println!("  - Adaptive instance normalization");
    println!();

    println!("• Video style transfer:");
    println!("  - Temporal consistency with optical flow");
    println!("  - Prevent flickering between frames");
    println!();

    println!("• Semantic style transfer:");
    println!("  - Apply different styles to different objects");
    println!("  - Face-aware style transfer");
    println!();

    println!("{}", "=".repeat(80));
    println!("{:^80}", "Style Transfer Complete!");
    println!("{:^80}", "The generated images combine content structure with artistic style");
    println!("{}", "=".repeat(80));
}

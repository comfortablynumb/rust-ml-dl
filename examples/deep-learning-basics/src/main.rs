//! # Deep Learning Basics Example
//!
//! This example covers fundamental concepts in deep learning including
//! different architectures, techniques, and best practices.
//!
//! ## What is Deep Learning?
//!
//! Deep learning is a subset of machine learning that uses neural networks
//! with multiple layers (hence "deep") to learn hierarchical representations
//! of data.
//!
//! ## Key Concepts
//!
//! ### 1. Deep vs Shallow Networks
//!
//! **Shallow Network** (1-2 hidden layers):
//! ```
//! Input → Hidden → Output
//! ```
//! - Simpler, faster to train
//! - Limited representation capacity
//! - Works well for simple problems
//!
//! **Deep Network** (many hidden layers):
//! ```
//! Input → Hidden₁ → Hidden₂ → ... → Hiddenₙ → Output
//! ```
//! - Can learn complex hierarchies
//! - More parameters, harder to train
//! - State-of-the-art for complex tasks
//!
//! ### 2. Why Go Deep?
//!
//! Deep networks learn hierarchical features:
//!
//! **Image Recognition Example:**
//! - Layer 1: Edges and corners
//! - Layer 2: Simple shapes (circles, lines)
//! - Layer 3: Object parts (eyes, wheels)
//! - Layer 4: Complete objects (faces, cars)
//!
//! ### 3. Common Challenges
//!
//! #### Vanishing Gradients
//! - Problem: Gradients become very small in early layers
//! - Cause: Repeated multiplication of small numbers
//! - Solution: ReLU, ResNet, batch normalization
//!
//! #### Exploding Gradients
//! - Problem: Gradients become very large
//! - Cause: Repeated multiplication of large numbers
//! - Solution: Gradient clipping, careful initialization
//!
//! #### Overfitting
//! - Problem: Model memorizes training data
//! - Signs: High train accuracy, low test accuracy
//! - Solutions:
//!   - **Dropout**: Randomly disable neurons during training
//!   - **L1/L2 Regularization**: Penalize large weights
//!   - **Early Stopping**: Stop when validation loss increases
//!   - **Data Augmentation**: Create more training data
//!   - **Batch Normalization**: Normalize layer inputs
//!
//! ### 4. Activation Functions Comparison
//!
//! | Function | Formula | Range | Use Case |
//! |----------|---------|-------|----------|
//! | Sigmoid | 1/(1+e⁻ˣ) | (0,1) | Binary output |
//! | Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1,1) | Hidden layers |
//! | ReLU | max(0,x) | [0,∞) | Most hidden layers |
//! | Leaky ReLU | max(0.01x,x) | (-∞,∞) | Avoid dead neurons |
//! | Softmax | eˣⁱ/Σeˣʲ | (0,1) | Multi-class output |
//!
//! ### 5. Weight Initialization
//!
//! Proper initialization is crucial:
//!
//! - **Random Small Values**: Can work but may be suboptimal
//! - **Xavier/Glorot**: For sigmoid/tanh: `√(1/n_in)`
//! - **He Initialization**: For ReLU: `√(2/n_in)`
//! - **Zero Initialization**: Don't! All neurons learn same thing
//!
//! ### 6. Optimization Algorithms
//!
//! Evolution from basic to advanced:
//!
//! 1. **SGD**: Basic, can be slow
//! 2. **Momentum**: Accelerates in consistent directions
//! 3. **RMSprop**: Adaptive learning rates per parameter
//! 4. **Adam**: Combines momentum + RMSprop (most popular!)
//! 5. **AdamW**: Adam with better weight decay
//!
//! ### 7. Batch Normalization
//!
//! Normalizes inputs to each layer:
//! - Reduces internal covariate shift
//! - Allows higher learning rates
//! - Acts as regularization
//! - Accelerates training
//!
//! ### 8. Dropout
//!
//! Randomly sets neurons to zero during training:
//! - Prevents co-adaptation of neurons
//! - Forces redundancy in representations
//! - Typical rate: 0.2-0.5
//!
//! ### 9. Learning Rate Schedules
//!
//! - **Step Decay**: Reduce by factor every N epochs
//! - **Exponential Decay**: Continuous reduction
//! - **Cosine Annealing**: Smooth oscillation
//! - **Warm Restarts**: Periodic resets
//! - **One Cycle**: Increase then decrease
//!
//! ### 10. Architecture Patterns
//!
//! **Feedforward (MLP)**:
//! ```
//! Input → Dense → Dense → ... → Output
//! ```
//! Use: Tabular data, simple patterns
//!
//! **Convolutional (CNN)**:
//! ```
//! Input → Conv → Pool → Conv → Pool → Dense → Output
//! ```
//! Use: Images, spatial data
//!
//! **Recurrent (RNN/LSTM/GRU)**:
//! ```
//! Input₁ → RNN → RNN → RNN → Output
//!   ↑        ↓      ↓      ↓
//! Input₂ → Input₃ → Input₄
//! ```
//! Use: Sequential data, time series, text
//!
//! **Transformer**:
//! ```
//! Input → Attention → Feed Forward → Output
//! ```
//! Use: NLP, sequential data, GPT/BERT
//!
//! **Autoencoder**:
//! ```
//! Input → Encoder → Bottleneck → Decoder → Output
//! ```
//! Use: Dimensionality reduction, denoising
//!
//! **GAN (Generative Adversarial Network)**:
//! ```
//! Noise → Generator → Fake Image
//!                        ↓
//! Real Image → Discriminator → Real/Fake?
//! ```
//! Use: Image generation, data augmentation

use ndarray::Array1;

fn main() {
    println!("=== Deep Learning Basics ===\n");

    println!("This example covers key concepts in deep learning.\n");

    // Demonstrate different activation functions
    println!("1. Activation Functions\n");

    let x_values: Vec<f64> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

    println!("   Input values: {:?}\n", x_values);

    for &x in &x_values {
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        let tanh = x.tanh();
        let relu = x.max(0.0);
        let leaky_relu = if x > 0.0 { x } else { 0.01 * x };

        println!("   x = {:.1}", x);
        println!("      Sigmoid:     {:.4}", sigmoid);
        println!("      Tanh:        {:.4}", tanh);
        println!("      ReLU:        {:.4}", relu);
        println!("      Leaky ReLU:  {:.4}", leaky_relu);
        println!();
    }

    // Softmax example
    println!("2. Softmax (for multi-class classification)\n");

    let logits: Array1<f64> = Array1::from_vec(vec![2.0, 1.0, 0.1]);
    let exp_sum: f64 = logits.iter().map(|&x| x.exp()).sum();
    let softmax: Vec<f64> = logits.iter()
        .map(|&x| x.exp() / exp_sum)
        .collect();

    println!("   Logits:      {:?}", logits.to_vec());
    println!("   Softmax:     {:?}", softmax);
    println!("   Sum:         {:.4} (should be 1.0)", softmax.iter().sum::<f64>());
    println!("   Interpretation: Class probabilities\n");

    // Common loss functions
    println!("3. Loss Functions\n");

    println!("   A) Binary Cross-Entropy (BCE):");
    println!("      L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]");
    println!("      Use: Binary classification\n");

    let y_true: f64 = 1.0;
    let y_pred: f64 = 0.8;
    let bce = -(y_true * y_pred.ln() + (1.0 - y_true) * (1.0 - y_pred).ln());
    println!("      Example: y_true = {}, y_pred = {:.2}, BCE = {:.4}\n", y_true, y_pred, bce);

    println!("   B) Categorical Cross-Entropy:");
    println!("      L = -Σ yᵢ·log(ŷᵢ)");
    println!("      Use: Multi-class classification\n");

    println!("   C) Mean Squared Error (MSE):");
    println!("      L = (1/n)·Σ(y - ŷ)²");
    println!("      Use: Regression\n");

    println!("   D) Mean Absolute Error (MAE):");
    println!("      L = (1/n)·Σ|y - ŷ|");
    println!("      Use: Regression (robust to outliers)\n");

    // Regularization
    println!("4. Regularization Techniques\n");

    println!("   A) L2 Regularization (Ridge):");
    println!("      Loss = Original Loss + λ·Σw²");
    println!("      Effect: Penalizes large weights, smooth models\n");

    println!("   B) L1 Regularization (Lasso):");
    println!("      Loss = Original Loss + λ·Σ|w|");
    println!("      Effect: Drives some weights to zero, feature selection\n");

    println!("   C) Dropout:");
    println!("      Randomly set neurons to 0 during training");
    println!("      Effect: Prevents overfitting, ensemble effect\n");

    println!("   D) Early Stopping:");
    println!("      Stop training when validation loss increases");
    println!("      Effect: Prevents overfitting\n");

    // Training tips
    println!("5. Training Best Practices\n");

    println!("   ✓ Always shuffle training data");
    println!("   ✓ Normalize/standardize inputs");
    println!("   ✓ Use batch normalization for deep networks");
    println!("   ✓ Start with simple architecture, add complexity if needed");
    println!("   ✓ Monitor both training and validation metrics");
    println!("   ✓ Use learning rate schedules");
    println!("   ✓ Apply data augmentation (for images)");
    println!("   ✓ Initialize weights properly (He/Xavier)");
    println!("   ✓ Use Adam optimizer as default");
    println!("   ✓ Try different architectures and hyperparameters\n");

    // Hyperparameter tuning
    println!("6. Key Hyperparameters\n");

    println!("   Network Architecture:");
    println!("   - Number of layers (depth)");
    println!("   - Neurons per layer (width)");
    println!("   - Activation functions\n");

    println!("   Training:");
    println!("   - Learning rate (most important!)");
    println!("   - Batch size (32, 64, 128, 256)");
    println!("   - Number of epochs");
    println!("   - Optimizer (Adam, SGD, etc.)\n");

    println!("   Regularization:");
    println!("   - Dropout rate (0.2-0.5)");
    println!("   - L1/L2 penalty (1e-5 to 1e-3)");
    println!("   - Batch normalization momentum\n");

    // Common architectures
    println!("7. Popular Deep Learning Architectures\n");

    println!("   Computer Vision:");
    println!("   - LeNet (1998): First CNN");
    println!("   - AlexNet (2012): Deep CNN, won ImageNet");
    println!("   - VGG (2014): Very deep, simple architecture");
    println!("   - ResNet (2015): Skip connections, 100+ layers");
    println!("   - EfficientNet (2019): Optimized scaling\n");

    println!("   Natural Language Processing:");
    println!("   - RNN/LSTM/GRU: Sequential processing");
    println!("   - Transformer (2017): Attention mechanism");
    println!("   - BERT (2018): Bidirectional transformer");
    println!("   - GPT (2018-2023): Autoregressive transformer");
    println!("   - T5: Text-to-text framework\n");

    println!("   Generative Models:");
    println!("   - VAE: Variational autoencoder");
    println!("   - GAN: Generative adversarial network");
    println!("   - Diffusion Models: Stable Diffusion, DALL-E\n");

    println!("=== Summary ===\n");
    println!("Deep learning key points:");
    println!("1. More layers = more complex patterns (but harder to train)");
    println!("2. ReLU is the go-to activation for hidden layers");
    println!("3. Adam optimizer works well in most cases");
    println!("4. Regularization prevents overfitting");
    println!("5. Proper initialization and normalization are crucial");
    println!("6. Learning rate is the most important hyperparameter");
    println!("7. Start simple, add complexity gradually");
    println!("8. Monitor validation metrics to detect overfitting");
    println!();
    println!("Next steps: Explore CNNs, RNNs, Transformers for specific tasks!");
}

//! # Neural Network Example
//!
//! This example implements a simple feedforward neural network from scratch
//! to demonstrate the fundamental concepts of deep learning.
//!
//! ## What is a Neural Network?
//!
//! A neural network is a computational model inspired by biological neurons.
//! It consists of layers of interconnected nodes (neurons) that transform
//! input data through learned parameters (weights and biases).
//!
//! ## Architecture
//!
//! ```
//! Input Layer → Hidden Layer(s) → Output Layer
//!
//!    x₁ ─┐
//!        ├─→ h₁ ─┐
//!    x₂ ─┤       ├─→ o₁
//!        ├─→ h₂ ─┘
//!    x₃ ─┘
//! ```
//!
//! ## Forward Propagation
//!
//! Data flows forward through the network:
//!
//! 1. **Input Layer**: Receives raw features
//! 2. **Hidden Layers**: Each neuron computes:
//!    ```
//!    z = w₁x₁ + w₂x₂ + ... + b    (linear combination)
//!    a = σ(z)                      (activation function)
//!    ```
//! 3. **Output Layer**: Produces predictions
//!
//! ## Activation Functions
//!
//! Non-linear functions that enable learning complex patterns:
//!
//! ### Sigmoid
//! ```
//! σ(x) = 1 / (1 + e⁻ˣ)
//! Range: (0, 1)
//! Use: Binary classification, gates in LSTM
//! ```
//!
//! ### Tanh
//! ```
//! tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
//! Range: (-1, 1)
//! Use: Hidden layers (zero-centered)
//! ```
//!
//! ### ReLU (Rectified Linear Unit)
//! ```
//! ReLU(x) = max(0, x)
//! Range: [0, ∞)
//! Use: Most common for hidden layers
//! Advantage: No vanishing gradient, fast
//! ```
//!
//! ### Softmax
//! ```
//! softmax(xᵢ) = eˣⁱ / Σⱼeˣʲ
//! Range: (0, 1), sum to 1
//! Use: Multi-class classification output
//! ```
//!
//! ## Backpropagation
//!
//! Algorithm for training neural networks:
//!
//! 1. **Forward pass**: Compute predictions
//! 2. **Compute loss**: Compare predictions to targets
//! 3. **Backward pass**: Compute gradients using chain rule
//! 4. **Update weights**: Use gradient descent
//!
//! The chain rule allows us to compute gradients layer by layer:
//! ```
//! ∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w
//! ```
//!
//! ## Why Neural Networks Work
//!
//! **Universal Approximation Theorem**: A neural network with at least
//! one hidden layer can approximate any continuous function!
//!
//! ## Use Cases
//!
//! - Image recognition (CNNs)
//! - Natural language processing (RNNs, Transformers)
//! - Speech recognition
//! - Game playing (AlphaGo, AlphaZero)
//! - Recommendation systems
//! - Time series forecasting
//!
//! ## This Example
//!
//! We implement a simple 2-layer network for binary classification:
//! - Input layer: 2 features
//! - Hidden layer: 4 neurons with ReLU
//! - Output layer: 1 neuron with sigmoid
//! - Training: Backpropagation with gradient descent

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Uniform, StandardNormal};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

/// Sigmoid activation function
fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

/// Derivative of sigmoid
fn sigmoid_derivative(x: &Array1<f64>) -> Array1<f64> {
    let sig = sigmoid(x);
    &sig * &sig.mapv(|v| 1.0 - v)
}

/// ReLU activation function
fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.max(0.0))
}

/// Derivative of ReLU
fn relu_derivative(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
}

/// Binary cross-entropy loss
fn binary_cross_entropy(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    let epsilon = 1e-15;
    let pred_clipped = predictions.mapv(|p| p.max(epsilon).min(1.0 - epsilon));
    let loss = targets * pred_clipped.mapv(|p| p.ln())
        + (targets.mapv(|t| 1.0 - t)) * pred_clipped.mapv(|p| (1.0 - p).ln());
    -loss.mean().unwrap()
}

/// Simple 2-layer neural network
struct NeuralNetwork {
    // Layer 1: input → hidden
    weights1: Array2<f64>,
    bias1: Array1<f64>,
    // Layer 2: hidden → output
    weights2: Array2<f64>,
    bias2: Array1<f64>,
    learning_rate: f64,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64, rng: &mut StdRng) -> Self {
        // Initialize weights with small random values (Xavier initialization)
        let scale1 = (2.0 / input_size as f64).sqrt();
        let scale2 = (2.0 / hidden_size as f64).sqrt();

        NeuralNetwork {
            weights1: Array2::random_using((input_size, hidden_size), StandardNormal, rng) * scale1,
            bias1: Array1::zeros(hidden_size),
            weights2: Array2::random_using((hidden_size, output_size), StandardNormal, rng) * scale2,
            bias2: Array1::zeros(output_size),
            learning_rate,
        }
    }

    /// Forward propagation
    fn forward(&self, input: &Array1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
        // Hidden layer
        let z1 = input.dot(&self.weights1) + &self.bias1;
        let a1 = relu(&z1);

        // Output layer
        let z2 = a1.dot(&self.weights2) + &self.bias2;
        let a2 = sigmoid(&z2);

        (z1, a1, z2, a2)
    }

    /// Backward propagation and weight update
    fn backward(&mut self, input: &Array1<f64>, target: &Array1<f64>,
                z1: &Array1<f64>, a1: &Array1<f64>, _z2: &Array1<f64>, output: &Array1<f64>) {
        // Output layer gradients
        let output_error = output - target;  // derivative of BCE w.r.t. output
        let output_delta = output_error;     // sigmoid derivative already in BCE gradient

        // Hidden layer gradients
        let hidden_error = output_delta.dot(&self.weights2.t());
        let hidden_delta = &hidden_error * &relu_derivative(z1);

        // Update weights and biases
        // Output layer
        let weights2_gradient = a1.view().insert_axis(Axis(1)).dot(&output_delta.view().insert_axis(Axis(0)));
        self.weights2 = &self.weights2 - self.learning_rate * &weights2_gradient;
        self.bias2 = &self.bias2 - self.learning_rate * &output_delta;

        // Hidden layer
        let weights1_gradient = input.view().insert_axis(Axis(1)).dot(&hidden_delta.view().insert_axis(Axis(0)));
        self.weights1 = &self.weights1 - self.learning_rate * &weights1_gradient;
        self.bias1 = &self.bias1 - self.learning_rate * &hidden_delta;
    }

    /// Predict for a single sample
    fn predict(&self, input: &Array1<f64>) -> f64 {
        let (_, _, _, output) = self.forward(input);
        output[0]
    }
}

fn main() -> anyhow::Result<()> {
    println!("=== Neural Network Example ===\n");

    let mut rng = StdRng::seed_from_u64(42);

    // Generate XOR-like dataset (non-linearly separable)
    println!("1. Generating non-linear classification data...");
    let n_samples_per_class = 250;

    // Create circular decision boundary
    let mut features = Vec::new();
    let mut targets = Vec::new();

    // Class 0: points inside circle
    for _ in 0..n_samples_per_class {
        let angle: f64 = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        let radius: f64 = rng.gen_range(0.0..2.0);
        let x = radius * angle.cos();
        let y = radius * angle.sin();
        features.push(vec![x, y]);
        targets.push(0.0);
    }

    // Class 1: points outside circle
    for _ in 0..n_samples_per_class {
        let angle: f64 = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        let radius: f64 = rng.gen_range(3.0..5.0);
        let x = radius * angle.cos();
        let y = radius * angle.sin();
        features.push(vec![x, y]);
        targets.push(1.0);
    }

    println!("   - Generated {} samples", features.len());
    println!("   - This is a non-linear problem (circular boundary)\n");

    // Shuffle data
    let mut indices: Vec<usize> = (0..features.len()).collect();
    indices.shuffle(&mut rng);

    let shuffled_features: Vec<Vec<f64>> = indices.iter().map(|&i| features[i].clone()).collect();
    let shuffled_targets: Vec<f64> = indices.iter().map(|&i| targets[i]).collect();

    // Split train/test
    let split_idx = (shuffled_features.len() as f64 * 0.8) as usize;
    let train_features = &shuffled_features[..split_idx];
    let train_targets = &shuffled_targets[..split_idx];
    let test_features = &shuffled_features[split_idx..];
    let test_targets = &shuffled_targets[split_idx..];

    println!("2. Building neural network...");
    let input_size = 2;
    let hidden_size = 4;
    let output_size = 1;
    let learning_rate = 0.1;

    let mut nn = NeuralNetwork::new(input_size, hidden_size, output_size, learning_rate, &mut rng);

    println!("   Architecture: {} → {} (ReLU) → {} (Sigmoid)", input_size, hidden_size, output_size);
    println!("   Learning rate: {}\n", learning_rate);

    // Train the network
    println!("3. Training neural network...");
    let epochs = 200;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        // Train on each sample
        for (features, &target) in train_features.iter().zip(train_targets.iter()) {
            let input = Array1::from_vec(features.clone());
            let target_arr = Array1::from_vec(vec![target]);

            // Forward pass
            let (z1, a1, z2, output) = nn.forward(&input);

            // Compute loss
            total_loss += binary_cross_entropy(&output, &target_arr);

            // Backward pass
            nn.backward(&input, &target_arr, &z1, &a1, &z2, &output);
        }

        let avg_loss = total_loss / train_features.len() as f64;

        if epoch % 20 == 0 {
            println!("   Epoch {}: Loss = {:.4}", epoch, avg_loss);
        }
    }

    println!("\n4. Evaluating on test set...");

    // Test the network
    let mut correct = 0;
    for (features, &target) in test_features.iter().zip(test_targets.iter()) {
        let input = Array1::from_vec(features.clone());
        let prediction = nn.predict(&input);
        let predicted_class = if prediction >= 0.5 { 1.0 } else { 0.0 };

        if predicted_class == target {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / test_features.len() as f64;
    println!("   Test Accuracy: {:.2}%\n", accuracy * 100.0);

    // Show some predictions
    println!("5. Sample predictions (first 10):");
    for i in 0..10.min(test_features.len()) {
        let input = Array1::from_vec(test_features[i].clone());
        let prediction = nn.predict(&input);
        let predicted_class = if prediction >= 0.5 { 1 } else { 0 };
        let actual_class = test_targets[i] as i32;
        let marker = if predicted_class == actual_class { "✓" } else { "✗" };

        println!("   [{:.2}, {:.2}] → Prob: {:.3}, Predicted: {}, Actual: {} {}",
                 test_features[i][0],
                 test_features[i][1],
                 prediction,
                 predicted_class,
                 actual_class,
                 marker);
    }

    println!("\n=== Example Complete! ===");
    println!("\nKey Takeaways:");
    println!("- Neural networks can learn non-linear decision boundaries");
    println!("- Hidden layers with activation functions enable complex patterns");
    println!("- Backpropagation efficiently computes gradients");
    println!("- More layers/neurons = more complex functions (but risk overfitting)");

    Ok(())
}

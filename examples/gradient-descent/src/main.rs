//! # Gradient Descent Example
//!
//! This example implements gradient descent from scratch to demonstrate
//! one of the most fundamental optimization algorithms in machine learning.
//!
//! ## What is Gradient Descent?
//!
//! Gradient descent is an iterative optimization algorithm for finding the
//! minimum of a function. In machine learning, we use it to minimize the
//! cost/loss function and find optimal model parameters.
//!
//! ## The Core Idea
//!
//! Imagine you're on a mountain in fog and want to reach the valley:
//! 1. Look around and find the direction of steepest descent
//! 2. Take a step in that direction
//! 3. Repeat until you reach the bottom
//!
//! Mathematically:
//! ```
//! θ = θ - α * ∇J(θ)
//! ```
//!
//! Where:
//! - θ (theta) are the parameters we're optimizing
//! - α (alpha) is the learning rate (step size)
//! - ∇J(θ) is the gradient (direction of steepest ascent)
//! - We subtract because we want to go downhill (minimize)
//!
//! ## Types of Gradient Descent
//!
//! ### 1. Batch Gradient Descent
//! - Uses entire dataset to compute gradient
//! - Pros: Stable convergence, reaches global minimum for convex functions
//! - Cons: Slow for large datasets, requires all data in memory
//!
//! ### 2. Stochastic Gradient Descent (SGD)
//! - Uses one sample at a time
//! - Pros: Fast updates, can escape local minima, works with streaming data
//! - Cons: Noisy updates, fluctuating loss
//!
//! ### 3. Mini-batch Gradient Descent
//! - Uses small batches of samples
//! - Pros: Balance between batch and SGD, GPU-friendly
//! - Cons: Hyperparameter tuning needed
//!
//! ## Learning Rate
//!
//! The learning rate α controls the step size:
//! - Too small: Slow convergence, may get stuck
//! - Too large: Overshooting, divergence
//! - Just right: Efficient convergence
//!
//! ## Use Cases
//!
//! Gradient descent is used to train:
//! - Linear regression
//! - Logistic regression
//! - Neural networks
//! - Support vector machines
//! - Many other ML models
//!
//! ## This Example
//!
//! We'll implement gradient descent to solve linear regression:
//! - Minimize: MSE = (1/n) Σ(y - ŷ)²
//! - Find optimal: weights (w) and bias (b)
//! - Compare: Batch, SGD, and Mini-batch variants

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

/// Mean Squared Error loss function
fn mse_loss(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    ((predictions - targets).mapv(|x| x.powi(2))).mean().unwrap()
}

/// Make predictions: y = Xw + b
fn predict(features: &Array2<f64>, weights: &Array1<f64>, bias: f64) -> Array1<f64> {
    features.dot(weights) + bias
}

/// Compute gradients for batch gradient descent
fn compute_gradients(
    features: &Array2<f64>,
    targets: &Array1<f64>,
    weights: &Array1<f64>,
    bias: f64,
) -> (Array1<f64>, f64) {
    let n = features.nrows() as f64;
    let predictions = predict(features, weights, bias);
    let errors = &predictions - targets;

    // Gradient of weights: (2/n) * X^T * (Xw + b - y)
    let grad_weights = (2.0 / n) * features.t().dot(&errors);

    // Gradient of bias: (2/n) * Σ(Xw + b - y)
    let grad_bias = (2.0 / n) * errors.sum();

    (grad_weights, grad_bias)
}

/// Batch Gradient Descent
fn batch_gradient_descent(
    features: &Array2<f64>,
    targets: &Array1<f64>,
    learning_rate: f64,
    n_iterations: usize,
) -> (Array1<f64>, f64, Vec<f64>) {
    let n_features = features.ncols();
    let mut weights = Array1::zeros(n_features);
    let mut bias = 0.0;
    let mut loss_history = Vec::new();

    for iteration in 0..n_iterations {
        // Compute gradients using entire dataset
        let (grad_w, grad_b) = compute_gradients(features, targets, &weights, bias);

        // Update parameters
        weights = &weights - learning_rate * &grad_w;
        bias -= learning_rate * grad_b;

        // Track loss
        let predictions = predict(features, &weights, bias);
        let loss = mse_loss(&predictions, targets);
        loss_history.push(loss);

        if iteration % 100 == 0 {
            println!("   Iteration {}: Loss = {:.4}", iteration, loss);
        }
    }

    (weights, bias, loss_history)
}

/// Stochastic Gradient Descent (SGD)
fn stochastic_gradient_descent(
    features: &Array2<f64>,
    targets: &Array1<f64>,
    learning_rate: f64,
    n_epochs: usize,
    rng: &mut StdRng,
) -> (Array1<f64>, f64, Vec<f64>) {
    let n_features = features.ncols();
    let n_samples = features.nrows();
    let mut weights = Array1::zeros(n_features);
    let mut bias = 0.0;
    let mut loss_history = Vec::new();

    for epoch in 0..n_epochs {
        // Shuffle indices
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(rng);

        // Process one sample at a time
        for &idx in &indices {
            let x = features.row(idx);
            let y = targets[idx];

            // Predict
            let prediction = x.dot(&weights) + bias;
            let error = prediction - y;

            // Update parameters using single sample
            for j in 0..n_features {
                weights[j] -= learning_rate * 2.0 * error * x[j];
            }
            bias -= learning_rate * 2.0 * error;
        }

        // Track loss after each epoch
        let predictions = predict(features, &weights, bias);
        let loss = mse_loss(&predictions, targets);
        loss_history.push(loss);

        if epoch % 20 == 0 {
            println!("   Epoch {}: Loss = {:.4}", epoch, loss);
        }
    }

    (weights, bias, loss_history)
}

/// Mini-batch Gradient Descent
fn minibatch_gradient_descent(
    features: &Array2<f64>,
    targets: &Array1<f64>,
    learning_rate: f64,
    batch_size: usize,
    n_epochs: usize,
    rng: &mut StdRng,
) -> (Array1<f64>, f64, Vec<f64>) {
    let n_features = features.ncols();
    let n_samples = features.nrows();
    let mut weights = Array1::zeros(n_features);
    let mut bias = 0.0;
    let mut loss_history = Vec::new();

    for epoch in 0..n_epochs {
        // Shuffle indices
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(rng);

        // Process mini-batches
        for batch_start in (0..n_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n_samples);
            let batch_indices = &indices[batch_start..batch_end];

            // Get batch data
            let batch_features = features.select(Axis(0), batch_indices);
            let batch_targets = targets.select(Axis(0), batch_indices);

            // Compute gradients for batch
            let (grad_w, grad_b) = compute_gradients(&batch_features, &batch_targets, &weights, bias);

            // Update parameters
            weights = &weights - learning_rate * &grad_w;
            bias -= learning_rate * grad_b;
        }

        // Track loss after each epoch
        let predictions = predict(features, &weights, bias);
        let loss = mse_loss(&predictions, targets);
        loss_history.push(loss);

        if epoch % 20 == 0 {
            println!("   Epoch {}: Loss = {:.4}", epoch, loss);
        }
    }

    (weights, bias, loss_history)
}

fn main() -> anyhow::Result<()> {
    println!("=== Gradient Descent Example ===\n");

    // Set random seed
    let mut rng = StdRng::seed_from_u64(42);

    // Generate synthetic data: y = 3 + 2*x1 + 1.5*x2 + noise
    println!("1. Generating synthetic data...");
    let n_samples = 1000;
    let n_features = 2;

    let features = Array2::random_using((n_samples, n_features), Uniform::new(-10.0, 10.0), &mut rng);
    let true_weights = Array1::from_vec(vec![2.0, 1.5]);
    let true_bias = 3.0;
    let noise = Array1::random_using(n_samples, Uniform::new(-1.0, 1.0), &mut rng);
    let targets = features.dot(&true_weights) + true_bias + noise;

    println!("   - Samples: {}", n_samples);
    println!("   - Features: {}", n_features);
    println!("   - True weights: {:?}", true_weights.to_vec());
    println!("   - True bias: {}\n", true_bias);

    // 1. Batch Gradient Descent
    println!("2. Training with Batch Gradient Descent...");
    let learning_rate = 0.01;
    let n_iterations = 500;
    let (bgd_weights, bgd_bias, bgd_loss) = batch_gradient_descent(
        &features,
        &targets,
        learning_rate,
        n_iterations,
    );

    println!("\n   Results:");
    println!("   - Learned weights: {:?}", bgd_weights.to_vec());
    println!("   - Learned bias: {:.4}", bgd_bias);
    println!("   - Final loss: {:.4}\n", bgd_loss.last().unwrap());

    // 2. Stochastic Gradient Descent
    println!("3. Training with Stochastic Gradient Descent (SGD)...");
    let learning_rate_sgd = 0.001; // Smaller learning rate for SGD
    let n_epochs = 100;
    let (sgd_weights, sgd_bias, sgd_loss) = stochastic_gradient_descent(
        &features,
        &targets,
        learning_rate_sgd,
        n_epochs,
        &mut rng,
    );

    println!("\n   Results:");
    println!("   - Learned weights: {:?}", sgd_weights.to_vec());
    println!("   - Learned bias: {:.4}", sgd_bias);
    println!("   - Final loss: {:.4}\n", sgd_loss.last().unwrap());

    // 3. Mini-batch Gradient Descent
    println!("4. Training with Mini-batch Gradient Descent...");
    let learning_rate_mb = 0.01;
    let batch_size = 32;
    let (mb_weights, mb_bias, mb_loss) = minibatch_gradient_descent(
        &features,
        &targets,
        learning_rate_mb,
        batch_size,
        n_epochs,
        &mut rng,
    );

    println!("\n   Results:");
    println!("   - Learned weights: {:?}", mb_weights.to_vec());
    println!("   - Learned bias: {:.4}", mb_bias);
    println!("   - Final loss: {:.4}\n", mb_loss.last().unwrap());

    // Compare all methods
    println!("5. Comparison:\n");
    println!("   True parameters:       weights = {:?}, bias = {:.4}", true_weights.to_vec(), true_bias);
    println!("   Batch GD:              weights = [{:.4}, {:.4}], bias = {:.4}, loss = {:.4}",
             bgd_weights[0], bgd_weights[1], bgd_bias, bgd_loss.last().unwrap());
    println!("   SGD:                   weights = [{:.4}, {:.4}], bias = {:.4}, loss = {:.4}",
             sgd_weights[0], sgd_weights[1], sgd_bias, sgd_loss.last().unwrap());
    println!("   Mini-batch GD:         weights = [{:.4}, {:.4}], bias = {:.4}, loss = {:.4}",
             mb_weights[0], mb_weights[1], mb_bias, mb_loss.last().unwrap());

    println!("\n=== Example Complete! ===");
    println!("\nKey Takeaways:");
    println!("- Batch GD: Stable, smooth convergence, computationally expensive");
    println!("- SGD: Fast updates, noisy convergence, memory efficient");
    println!("- Mini-batch: Best of both worlds, most commonly used in practice");
    println!("- Learning rate is crucial: too high → divergence, too low → slow convergence");

    Ok(())
}

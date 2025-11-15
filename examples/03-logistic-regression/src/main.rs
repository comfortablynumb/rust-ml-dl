//! # Logistic Regression Example
//!
//! This example demonstrates logistic regression using the Linfa library.
//! Logistic regression is a fundamental classification algorithm despite its name.
//!
//! ## What is Logistic Regression?
//!
//! Logistic regression is used for binary classification problems (yes/no, true/false, 0/1).
//! Instead of predicting a continuous value like linear regression, it predicts the
//! probability that an instance belongs to a particular class.
//!
//! The model uses the logistic (sigmoid) function:
//!
//! σ(z) = 1 / (1 + e^(-z))
//!
//! Where z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
//!
//! The output σ(z) is always between 0 and 1, representing a probability.
//!
//! ## Use Cases
//!
//! - Email spam detection (spam vs not spam)
//! - Medical diagnosis (disease vs healthy)
//! - Credit risk assessment (default vs no default)
//! - Customer churn prediction (will leave vs will stay)
//! - Click-through rate prediction
//!
//! ## How it Works
//!
//! 1. Combines features linearly (like linear regression)
//! 2. Applies sigmoid function to get probability between 0 and 1
//! 3. Uses a threshold (typically 0.5) to classify
//! 4. Learns parameters by maximizing likelihood (or minimizing log-loss)
//!
//! ## Decision Boundary
//!
//! The model learns a decision boundary that separates the two classes.
//! For 2D data, this is a line; for higher dimensions, it's a hyperplane.

use linfa::prelude::*;
use linfa_logistic::LogisticRegression;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Uniform, StandardNormal};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() -> anyhow::Result<()> {
    println!("=== Logistic Regression Example ===\n");

    // Set random seed for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // Generate synthetic binary classification data
    println!("1. Generating synthetic classification data...");
    let n_samples_per_class = 500;
    let n_features = 2;

    // Class 0: centered around (-2, -2)
    let class0_features = Array2::random_using(
        (n_samples_per_class, n_features),
        StandardNormal,
        &mut rng,
    ) + Array2::from_elem((n_samples_per_class, n_features), -2.0);

    // Class 1: centered around (2, 2)
    let class1_features = Array2::random_using(
        (n_samples_per_class, n_features),
        StandardNormal,
        &mut rng,
    ) + Array2::from_elem((n_samples_per_class, n_features), 2.0);

    // Combine features
    let features = ndarray::concatenate![Axis(0), class0_features, class1_features];

    // Create labels: 0 for first half, 1 for second half
    let mut targets = Array1::zeros(n_samples_per_class * 2);
    targets.slice_mut(s![n_samples_per_class..]).fill(1.0);

    println!("   - Generated {} samples per class", n_samples_per_class);
    println!("   - Total samples: {}", features.nrows());
    println!("   - Features: {}\n", n_features);

    // Shuffle the data
    let mut indices: Vec<usize> = (0..features.nrows()).collect();
    use rand::seq::SliceRandom;
    indices.shuffle(&mut rng);

    let features = features.select(Axis(0), &indices);
    let targets = targets.select(Axis(0), &indices);

    // Split data into training and testing sets (80/20 split)
    println!("2. Splitting data into train and test sets...");
    let split_idx = (features.nrows() as f64 * 0.8) as usize;

    let train_features = features.slice(s![..split_idx, ..]).to_owned();
    let train_targets = targets.slice(s![..split_idx]).to_owned();
    let test_features = features.slice(s![split_idx.., ..]).to_owned();
    let test_targets = targets.slice(s![split_idx..]).to_owned();

    println!("   - Training samples: {}", train_features.nrows());
    println!("   - Testing samples: {}\n", test_features.nrows());

    // Create a dataset for training
    let train_dataset = Dataset::new(train_features.clone(), train_targets.clone());

    // Train the logistic regression model
    println!("3. Training logistic regression model...");
    let model = LogisticRegression::default()
        .max_iterations(1000)
        .gradient_tolerance(1e-5)
        .fit(&train_dataset)?;

    println!("   - Model trained successfully!\n");

    // Make predictions on test set
    println!("4. Making predictions on test set...");
    let predictions = model.predict(&test_features);

    // Calculate accuracy
    let mut correct = 0;
    for i in 0..predictions.len() {
        if predictions[i] == test_targets[i] as usize {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / predictions.len() as f64;

    println!("\n5. Model Performance:");
    println!("   - Accuracy: {:.2}%", accuracy * 100.0);

    // Calculate confusion matrix
    let mut true_positives = 0;
    let mut true_negatives = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;

    for i in 0..predictions.len() {
        match (test_targets[i] as usize, predictions[i]) {
            (1, 1) => true_positives += 1,
            (0, 0) => true_negatives += 1,
            (0, 1) => false_positives += 1,
            (1, 0) => false_negatives += 1,
            _ => {}
        }
    }

    println!("\n6. Confusion Matrix:");
    println!("                 Predicted");
    println!("               Neg    Pos");
    println!("   Actual Neg  {:4}   {:4}", true_negatives, false_positives);
    println!("          Pos  {:4}   {:4}", false_negatives, true_positives);

    // Calculate precision, recall, and F1 score
    let precision = if (true_positives + false_positives) > 0 {
        true_positives as f64 / (true_positives + false_positives) as f64
    } else {
        0.0
    };

    let recall = if (true_positives + false_negatives) > 0 {
        true_positives as f64 / (true_positives + false_negatives) as f64
    } else {
        0.0
    };

    let f1_score = if (precision + recall) > 0.0 {
        2.0 * (precision * recall) / (precision + recall)
    } else {
        0.0
    };

    println!("\n7. Additional Metrics:");
    println!("   - Precision: {:.4} (of predicted positives, how many are correct?)", precision);
    println!("   - Recall: {:.4} (of actual positives, how many did we find?)", recall);
    println!("   - F1 Score: {:.4} (harmonic mean of precision and recall)", f1_score);

    // Show some example predictions
    println!("\n8. Sample predictions (first 10 test samples):");
    for i in 0..10.min(test_targets.len()) {
        let actual = test_targets[i] as usize;
        let predicted = predictions[i];
        let correct_marker = if actual == predicted { "✓" } else { "✗" };
        println!("   Sample {}: Predicted = {}, Actual = {}, Features = [{:.2}, {:.2}] {}",
                 i + 1,
                 predicted,
                 actual,
                 test_features[[i, 0]],
                 test_features[[i, 1]],
                 correct_marker);
    }

    println!("\n=== Example Complete! ===");

    Ok(())
}

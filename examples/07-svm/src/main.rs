//! # Support Vector Machine (SVM) Example
//!
//! This example demonstrates Support Vector Machines using the Linfa library.
//! SVMs are powerful supervised learning models for classification and regression.
//!
//! ## What is SVM?
//!
//! A Support Vector Machine finds the optimal hyperplane that best separates
//! different classes in the feature space. The "best" hyperplane is the one
//! that maximizes the margin between the closest points of different classes.
//!
//! ## Key Concepts
//!
//! ### The Margin
//! The margin is the distance between the decision boundary (hyperplane) and
//! the nearest data points from each class. These nearest points are called
//! **support vectors**.
//!
//! ```
//! Class A              |              Class B
//!   x                  |                  o
//!     x     margin     |     margin     o
//!       x  <----->  [HYPERPLANE]  <----->  o
//!     x   (support     |     (support     o
//!   x      vector)     |      vector)   o
//! ```
//!
//! ### Hard vs Soft Margin
//!
//! **Hard Margin SVM:**
//! - Requires perfect linear separation
//! - No misclassifications allowed
//! - Works only when data is linearly separable
//!
//! **Soft Margin SVM:**
//! - Allows some misclassifications
//! - Uses penalty parameter C
//! - More practical for real-world data
//!
//! The C parameter controls the trade-off:
//! - Large C: Hard margin, low bias, high variance (may overfit)
//! - Small C: Soft margin, high bias, low variance (may underfit)
//!
//! ### The Kernel Trick
//!
//! For non-linearly separable data, SVMs use the "kernel trick" to
//! transform data into higher dimensions where it becomes separable.
//!
//! Common kernels:
//! - **Linear**: K(x, y) = x · y
//! - **Polynomial**: K(x, y) = (γ·x·y + r)^d
//! - **RBF (Gaussian)**: K(x, y) = exp(-γ||x-y||²)
//! - **Sigmoid**: K(x, y) = tanh(γ·x·y + r)
//!
//! ## Use Cases
//!
//! - Text classification (spam detection, sentiment analysis)
//! - Image classification
//! - Bioinformatics (protein classification)
//! - Handwriting recognition
//! - Face detection
//!
//! ## Advantages
//!
//! - Effective in high-dimensional spaces
//! - Memory efficient (only uses support vectors)
//! - Versatile (different kernel functions)
//! - Works well with clear margin of separation
//!
//! ## Disadvantages
//!
//! - Doesn't work well on large datasets (slow training)
//! - Sensitive to feature scaling
//! - Doesn't provide probability estimates directly
//! - Choosing the right kernel and parameters can be tricky

use linfa::prelude::*;
use linfa_svm::{Svm, SvmParams};
use ndarray::{Array1, Array2, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() -> anyhow::Result<()> {
    println!("=== Support Vector Machine (SVM) Example ===\n");

    // Set random seed for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // Generate synthetic binary classification data
    println!("1. Generating synthetic classification data...");
    let n_samples_per_class = 400;
    let n_features = 2;

    // Class 0: centered around (-3, -3)
    let class0_features = Array2::random_using(
        (n_samples_per_class, n_features),
        StandardNormal,
        &mut rng,
    ) * 1.5 + Array2::from_elem((n_samples_per_class, n_features), -3.0);

    // Class 1: centered around (3, 3)
    let class1_features = Array2::random_using(
        (n_samples_per_class, n_features),
        StandardNormal,
        &mut rng,
    ) * 1.5 + Array2::from_elem((n_samples_per_class, n_features), 3.0);

    // Combine features
    let features = ndarray::concatenate![Axis(0), class0_features, class1_features];

    // Create binary labels: false for class 0, true for class 1
    let mut targets = Vec::new();
    targets.extend(vec![false; n_samples_per_class]);
    targets.extend(vec![true; n_samples_per_class]);

    println!("   - Generated {} samples per class", n_samples_per_class);
    println!("   - Total samples: {}", features.nrows());
    println!("   - Features: {}\n", n_features);

    // Shuffle the data
    use rand::seq::SliceRandom;
    let mut indices: Vec<usize> = (0..features.nrows()).collect();
    indices.shuffle(&mut rng);

    let features = features.select(Axis(0), &indices);
    let targets: Vec<bool> = indices.iter().map(|&i| targets[i]).collect();

    // Split into train and test sets (80/20)
    println!("2. Splitting data into train and test sets...");
    let split_idx = (features.nrows() as f64 * 0.8) as usize;

    let train_features = features.slice(s![..split_idx, ..]).to_owned();
    let train_targets = targets[..split_idx].to_vec();
    let test_features = features.slice(s![split_idx.., ..]).to_owned();
    let test_targets = targets[split_idx..].to_vec();

    println!("   - Training samples: {}", train_features.nrows());
    println!("   - Testing samples: {}\n", test_features.nrows());

    // Create dataset for training
    let train_dataset = Dataset::new(train_features.clone(), train_targets.clone());

    // Train SVM with linear kernel
    println!("3. Training SVM with linear kernel...");
    println!("   (This finds the maximum margin hyperplane)\n");

    // Try different C values to show the effect
    let c_values = vec![0.1, 1.0, 10.0];

    for &c in &c_values {
        println!("   Training with C = {} (penalty parameter):", c);

        let model = Svm::<_, bool>::params()
            .pos_neg_weights(c, c)
            .gaussian_kernel(1.0)
            .fit(&train_dataset)?;

        // Make predictions on test set
        let predictions = model.predict(&test_features);

        // Calculate accuracy
        let mut correct = 0;
        for i in 0..predictions.len() {
            if predictions[i] == test_targets[i] {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / predictions.len() as f64;

        println!("      - Test Accuracy: {:.2}%", accuracy * 100.0);
        // Note: support_vectors() method not available in this version of linfa
        println!();
    }

    // Train final model with optimal C
    println!("4. Training final model with C = 1.0...");
    let model = Svm::<_, bool>::params()
        .pos_neg_weights(1.0, 1.0)
        .gaussian_kernel(1.0)
        .fit(&train_dataset)?;

    println!("   - Model trained successfully!");
    // Note: support_vectors() method not available in this version of linfa
    println!("   - (Support vectors are the points that define the decision boundary)\n");

    // Make predictions
    let predictions = model.predict(&test_features);

    // Calculate detailed metrics
    println!("5. Model Performance:");

    let mut correct = 0;
    let mut tp = 0; // true positives
    let mut tn = 0; // true negatives
    let mut fp = 0; // false positives
    let mut fn_count = 0; // false negatives

    for i in 0..predictions.len() {
        if predictions[i] == test_targets[i] {
            correct += 1;
            if test_targets[i] {
                tp += 1;
            } else {
                tn += 1;
            }
        } else {
            if predictions[i] {
                fp += 1;
            } else {
                fn_count += 1;
            }
        }
    }

    let accuracy = correct as f64 / predictions.len() as f64;
    println!("   - Accuracy: {:.2}%\n", accuracy * 100.0);

    // Confusion matrix
    println!("6. Confusion Matrix:");
    println!("                Predicted");
    println!("             Neg    Pos");
    println!("   Actual Neg {:4}   {:4}", tn, fp);
    println!("          Pos {:4}   {:4}\n", fn_count, tp);

    // Additional metrics
    let precision = if (tp + fp) > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };

    let recall = if (tp + fn_count) > 0 {
        tp as f64 / (tp + fn_count) as f64
    } else {
        0.0
    };

    let f1_score = if (precision + recall) > 0.0 {
        2.0 * (precision * recall) / (precision + recall)
    } else {
        0.0
    };

    println!("7. Metrics:");
    println!("   - Precision: {:.4}", precision);
    println!("   - Recall: {:.4}", recall);
    println!("   - F1 Score: {:.4}\n", f1_score);

    // Show sample predictions
    println!("8. Sample predictions (first 10 test samples):");
    for i in 0..10.min(test_targets.len()) {
        let predicted_label = if predictions[i] { 1 } else { 0 };
        let actual_label = if test_targets[i] { 1 } else { 0 };
        let correct_marker = if predictions[i] == test_targets[i] { "✓" } else { "✗" };

        println!("   Sample {}: Predicted = {}, Actual = {}, Features = [{:.2}, {:.2}] {}",
                 i + 1,
                 predicted_label,
                 actual_label,
                 test_features[[i, 0]],
                 test_features[[i, 1]],
                 correct_marker);
    }

    println!("\n=== Example Complete! ===");
    println!("\nKey Takeaways:");
    println!("- SVM finds the maximum margin hyperplane");
    println!("- Only support vectors (points near boundary) matter");
    println!("- C parameter controls margin softness");
    println!("- Kernel trick enables non-linear decision boundaries");
    println!("- Scales well to high-dimensional data");

    Ok(())
}

//! # Decision Trees Example
//!
//! This example demonstrates decision tree classification using the Linfa library.
//! Decision trees are intuitive, interpretable models that make decisions by
//! learning simple rules from data features.
//!
//! ## What is a Decision Tree?
//!
//! A decision tree is a flowchart-like structure where:
//! - Each internal node represents a test on a feature (e.g., "age < 30?")
//! - Each branch represents the outcome of the test
//! - Each leaf node represents a class label (decision)
//!
//! Example tree:
//! ```
//!          [Feature1 < 5?]
//!           /           \
//!         Yes            No
//!         /               \
//!    [Class A]      [Feature2 < 3?]
//!                    /           \
//!                  Yes            No
//!                  /               \
//!             [Class B]        [Class A]
//! ```
//!
//! ## How Decision Trees Work
//!
//! 1. **Start with all data** at the root
//! 2. **Find the best split**: Choose feature and threshold that best separates classes
//! 3. **Recursively split**: Apply same process to each subset
//! 4. **Stop when**:
//!    - All samples in a node have the same class
//!    - Maximum depth reached
//!    - Minimum samples per node reached
//!
//! ## Splitting Criteria
//!
//! Common methods to measure "best split":
//!
//! ### Gini Impurity
//! Measures the probability of incorrectly classifying a random sample:
//! ```
//! Gini = 1 - Σ(pᵢ)²
//! ```
//! Where pᵢ is the probability of class i.
//! - Gini = 0: Pure node (all same class)
//! - Gini = 0.5: Maximum impurity (binary classification)
//!
//! ### Entropy (Information Gain)
//! Measures disorder or uncertainty:
//! ```
//! Entropy = -Σ(pᵢ log₂(pᵢ))
//! ```
//!
//! ## Use Cases
//!
//! - Medical diagnosis (interpretable rules)
//! - Credit approval (explain decisions)
//! - Customer churn prediction
//! - Email filtering
//! - Any problem requiring interpretability
//!
//! ## Advantages
//!
//! - Easy to understand and interpret
//! - No feature scaling needed
//! - Handles both numerical and categorical data
//! - Can capture non-linear relationships
//! - Fast prediction
//!
//! ## Disadvantages
//!
//! - Prone to overfitting
//! - Unstable (small data changes → different tree)
//! - Biased toward dominant classes
//! - Can create overly complex trees

use linfa::prelude::*;
use linfa_trees::DecisionTree;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Uniform, StandardNormal};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

fn main() -> anyhow::Result<()> {
    println!("=== Decision Tree Classification Example ===\n");

    // Set random seed for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // Generate synthetic classification data
    // We'll create a dataset with clear decision boundaries
    println!("1. Generating synthetic classification data...");
    let n_samples_per_class = 300;
    let n_features = 4;

    // Class 0: Small values in all features
    let mut class0_features = Array2::random_using(
        (n_samples_per_class, n_features),
        Uniform::new(0.0, 3.0),
        &mut rng,
    );

    // Class 1: Large values in first two features, small in others
    let mut class1_features = Array2::zeros((n_samples_per_class, n_features));
    class1_features.slice_mut(s![.., 0..2]).assign(&Array2::random_using(
        (n_samples_per_class, 2),
        Uniform::new(7.0, 10.0),
        &mut rng,
    ));
    class1_features.slice_mut(s![.., 2..4]).assign(&Array2::random_using(
        (n_samples_per_class, 2),
        Uniform::new(0.0, 3.0),
        &mut rng,
    ));

    // Class 2: Medium values everywhere
    let class2_features = Array2::random_using(
        (n_samples_per_class, n_features),
        Uniform::new(4.0, 7.0),
        &mut rng,
    );

    // Combine all classes
    let features = ndarray::concatenate![
        Axis(0),
        class0_features,
        class1_features,
        class2_features
    ];

    // Create labels
    let mut targets = Array1::zeros(n_samples_per_class * 3);
    targets.slice_mut(s![n_samples_per_class..2*n_samples_per_class]).fill(1);
    targets.slice_mut(s![2*n_samples_per_class..]).fill(2);

    println!("   - Generated {} samples per class", n_samples_per_class);
    println!("   - Total samples: {}", features.nrows());
    println!("   - Features: {}", n_features);
    println!("   - Classes: 3\n");

    // Shuffle the data
    let mut indices: Vec<usize> = (0..features.nrows()).collect();
    indices.shuffle(&mut rng);

    let features = features.select(Axis(0), &indices);
    let targets = targets.select(Axis(0), &indices);

    // Split data into training and testing sets (80/20)
    println!("2. Splitting data into train and test sets...");
    let split_idx = (features.nrows() as f64 * 0.8) as usize;

    let train_features = features.slice(s![..split_idx, ..]).to_owned();
    let train_targets = targets.slice(s![..split_idx]).to_owned().mapv(|x| x as usize);
    let test_features = features.slice(s![split_idx.., ..]).to_owned();
    let test_targets = targets.slice(s![split_idx..]).to_owned().mapv(|x| x as usize);

    println!("   - Training samples: {}", train_features.nrows());
    println!("   - Testing samples: {}\n", test_features.nrows());

    // Create training dataset
    let train_dataset = Dataset::new(train_features.clone(), train_targets.clone());

    // Train decision tree with different max depths
    println!("3. Training decision trees with different max depths...\n");

    for max_depth in [3, 5, 10] {
        println!("   Training with max_depth = {}:", max_depth);

        let model = DecisionTree::params()
            .max_depth(Some(max_depth))
            .fit(&train_dataset)?;

        // Make predictions
        let predictions = model.predict(&test_features);

        // Calculate accuracy
        let mut correct = 0;
        for i in 0..predictions.len() {
            if predictions[i] == test_targets[i] {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / predictions.len() as f64;
        println!("   - Test Accuracy: {:.2}%\n", accuracy * 100.0);
    }

    // Train final model with optimal depth
    println!("4. Training final model with max_depth = 5...");
    let model = DecisionTree::params()
        .max_depth(Some(5))
        .fit(&train_dataset)?;

    println!("   - Model trained successfully!\n");

    // Make predictions
    let predictions = model.predict(&test_features);

    // Calculate detailed metrics
    println!("5. Model Performance:");

    // Overall accuracy
    let mut correct = 0;
    for i in 0..predictions.len() {
        if predictions[i] == test_targets[i] {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / predictions.len() as f64;
    println!("   - Overall Accuracy: {:.2}%\n", accuracy * 100.0);

    // Confusion matrix
    let n_classes = 3;
    let mut confusion_matrix = Array2::zeros((n_classes, n_classes));

    for i in 0..predictions.len() {
        confusion_matrix[[test_targets[i], predictions[i]]] += 1;
    }

    println!("6. Confusion Matrix:");
    println!("                Predicted");
    println!("              Class 0  Class 1  Class 2");
    for i in 0..n_classes {
        print!("   Actual {}", i);
        for j in 0..n_classes {
            print!("     {:4}", confusion_matrix[[i, j]]);
        }
        println!();
    }

    // Per-class metrics
    println!("\n7. Per-Class Metrics:");
    for class in 0..n_classes {
        let tp = confusion_matrix[[class, class]];
        let fp: i32 = confusion_matrix.column(class).iter().sum::<i32>() - tp;
        let fn_: i32 = confusion_matrix.row(class).iter().sum::<i32>() - tp;

        let precision = if (tp + fp) > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };

        let recall = if (tp + fn_) > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };

        let f1 = if (precision + recall) > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        };

        println!("   Class {}:", class);
        println!("     - Precision: {:.4}", precision);
        println!("     - Recall: {:.4}", recall);
        println!("     - F1-Score: {:.4}", f1);
    }

    // Sample predictions
    println!("\n8. Sample predictions (first 10 test samples):");
    for i in 0..10.min(test_targets.len()) {
        let correct_marker = if predictions[i] == test_targets[i] { "✓" } else { "✗" };
        println!("   Sample {}: Predicted = {}, Actual = {}, Features = [{:.1}, {:.1}, {:.1}, {:.1}] {}",
                 i + 1,
                 predictions[i],
                 test_targets[i],
                 test_features[[i, 0]],
                 test_features[[i, 1]],
                 test_features[[i, 2]],
                 test_features[[i, 3]],
                 correct_marker);
    }

    println!("\n=== Example Complete! ===");
    println!("\nKey Takeaways:");
    println!("- Decision trees learn hierarchical decision rules");
    println!("- Max depth controls model complexity (too deep = overfitting)");
    println!("- Trees are interpretable: you can visualize the decision path");
    println!("- Random Forests (ensemble of trees) often perform better");

    Ok(())
}

//! # Linear Regression Example
//!
//! This example demonstrates linear regression using the Linfa library.
//! Linear regression is one of the most fundamental supervised learning algorithms.
//!
//! ## What is Linear Regression?
//!
//! Linear regression models the relationship between a dependent variable (target)
//! and one or more independent variables (features) by fitting a linear equation:
//!
//! y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
//!
//! Where:
//! - y is the predicted value
//! - β₀ is the intercept (bias)
//! - β₁, β₂, ..., βₙ are the coefficients (weights)
//! - x₁, x₂, ..., xₙ are the features
//! - ε is the error term
//!
//! ## Use Cases
//!
//! - Predicting house prices based on features (size, location, etc.)
//! - Forecasting sales based on advertising spend
//! - Estimating crop yield based on rainfall and temperature
//! - Any problem where you need to predict a continuous numerical value
//!
//! ## How it Works
//!
//! 1. The algorithm finds the line (or hyperplane) that best fits the data
//! 2. "Best fit" is determined by minimizing the sum of squared errors
//! 3. The model learns the optimal weights (coefficients) for each feature
//!
//! ## Example Output
//!
//! This example will:
//! 1. Generate synthetic data with a known linear relationship
//! 2. Split the data into training and testing sets
//! 3. Train a linear regression model
//! 4. Make predictions on test data
//! 5. Evaluate the model's performance

use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{Array1, Array2, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() -> anyhow::Result<()> {
    println!("=== Linear Regression Example ===\n");

    // Set random seed for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // Generate synthetic data
    // True relationship: y = 3 + 2*x1 + 1.5*x2 + noise
    println!("1. Generating synthetic data...");
    let n_samples = 1000;
    let n_features = 2;

    // Generate random features
    let features = Array2::random_using((n_samples, n_features), Uniform::new(-10.0, 10.0), &mut rng);

    // Generate targets with known coefficients
    let true_coefficients = Array1::from_vec(vec![2.0, 1.5]);
    let true_intercept = 3.0;

    let noise = Array1::random_using(n_samples, Uniform::new(-1.0, 1.0), &mut rng);
    let targets = features.dot(&true_coefficients) + true_intercept + noise;

    println!("   - Generated {} samples with {} features", n_samples, n_features);
    println!("   - True coefficients: {:?}", true_coefficients.to_vec());
    println!("   - True intercept: {}\n", true_intercept);

    // Split data into training and testing sets (80/20 split)
    println!("2. Splitting data into train and test sets...");
    let split_idx = (n_samples as f64 * 0.8) as usize;

    let train_features = features.slice(s![..split_idx, ..]).to_owned();
    let train_targets = targets.slice(s![..split_idx]).to_owned();
    let test_features = features.slice(s![split_idx.., ..]).to_owned();
    let test_targets = targets.slice(s![split_idx..]).to_owned();

    println!("   - Training samples: {}", train_features.nrows());
    println!("   - Testing samples: {}\n", test_features.nrows());

    // Create a dataset for training
    let train_dataset = Dataset::new(train_features, train_targets);

    // Train the linear regression model
    println!("3. Training linear regression model...");
    let model = LinearRegression::new()
        .fit(&train_dataset)?;

    println!("   - Model trained successfully!\n");

    // Get learned parameters
    println!("4. Learned parameters:");
    println!("   - Coefficients: {:?}", model.params().to_vec());
    println!("   - Intercept: {}\n", model.intercept());

    // Make predictions on test set
    println!("5. Making predictions on test set...");
    let test_dataset = DatasetBase::from(test_features.clone());
    let predictions = model.predict(&test_dataset);

    // Calculate metrics
    let mse = (&predictions - &test_targets)
        .mapv(|x| x.powi(2))
        .mean()
        .unwrap();

    let mae = (&predictions - &test_targets)
        .mapv(|x| x.abs())
        .mean()
        .unwrap();

    // Calculate R² score
    let mean_target = test_targets.mean().unwrap();
    let ss_tot: f64 = test_targets.mapv(|x| (x - mean_target).powi(2)).sum();
    let ss_res: f64 = (&predictions - &test_targets).mapv(|x| x.powi(2)).sum();
    let r2_score = 1.0 - (ss_res / ss_tot);

    println!("\n6. Model Performance:");
    println!("   - Mean Squared Error (MSE): {:.4}", mse);
    println!("   - Mean Absolute Error (MAE): {:.4}", mae);
    println!("   - R² Score: {:.4}", r2_score);
    println!("\n   Note: R² = 1.0 means perfect predictions");
    println!("         R² = 0.0 means the model is no better than predicting the mean");

    // Show some example predictions
    println!("\n7. Sample predictions (first 5 test samples):");
    for i in 0..5.min(test_targets.len()) {
        println!("   Sample {}: Predicted = {:.2}, Actual = {:.2}, Error = {:.2}",
                 i + 1,
                 predictions[i],
                 test_targets[i],
                 (predictions[i] - test_targets[i]).abs());
    }

    println!("\n=== Example Complete! ===");

    Ok(())
}

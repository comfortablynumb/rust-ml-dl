//! # Data Preprocessing Example
//!
//! This example demonstrates essential data preprocessing techniques for machine learning.
//! Proper preprocessing is crucial for model performance!
//!
//! ## Why Preprocess Data?
//!
//! Raw data is often:
//! - On different scales (age: 0-100, income: 0-1,000,000)
//! - Missing values
//! - Contains outliers
//! - Has unnecessary features
//! - Imbalanced classes
//!
//! Good preprocessing:
//! - Improves model performance
//! - Speeds up training
//! - Helps with convergence
//! - Prevents features from dominating
//!
//! ## Common Preprocessing Techniques
//!
//! ### 1. Feature Scaling
//!
//! Many algorithms (SVM, Neural Networks, KNN) are sensitive to feature scales.
//!
//! **Standardization (Z-score normalization)**:
//! ```
//! x' = (x - μ) / σ
//! ```
//! - Mean (μ) = 0, Standard deviation (σ) = 1
//! - Preserves outliers
//! - Use when: Features are normally distributed
//!
//! **Min-Max Normalization**:
//! ```
//! x' = (x - min) / (max - min)
//! ```
//! - Range: [0, 1]
//! - Sensitive to outliers
//! - Use when: Features have known bounds
//!
//! **Robust Scaling**:
//! ```
//! x' = (x - median) / IQR
//! ```
//! - Uses median and interquartile range
//! - Robust to outliers
//! - Use when: Data has many outliers
//!
//! ### 2. Handling Missing Values
//!
//! - **Drop**: Remove rows/columns with missing values
//! - **Mean/Median Imputation**: Fill with average
//! - **Mode Imputation**: Fill with most common value
//! - **Forward/Backward Fill**: Use previous/next value
//! - **Model-based**: Predict missing values
//!
//! ### 3. Encoding Categorical Variables
//!
//! - **Label Encoding**: Map to integers (0, 1, 2, ...)
//! - **One-Hot Encoding**: Binary columns for each category
//! - **Target Encoding**: Use target statistics
//!
//! ### 4. Handling Outliers
//!
//! - **IQR Method**: Remove values outside 1.5 × IQR
//! - **Z-score**: Remove |z| > 3
//! - **Clipping**: Cap at percentiles
//!
//! ### 5. Feature Engineering
//!
//! - **Polynomial Features**: x₁, x₂ → x₁, x₂, x₁², x₁x₂, x₂²
//! - **Binning**: Continuous → Categorical
//! - **Log Transform**: Handle skewed distributions

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Uniform, StandardNormal};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Standardize features (z-score normalization)
fn standardize(data: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mean = data.mean_axis(Axis(0)).unwrap();
    let std = data.std_axis(Axis(0), 1.0);

    let standardized = (data - &mean) / &std;

    (standardized, mean, std)
}

/// Min-max normalization
fn min_max_normalize(data: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let min = data.fold_axis(Axis(0), f64::INFINITY, |&acc, &x| acc.min(x));
    let max = data.fold_axis(Axis(0), f64::NEG_INFINITY, |&acc, &x| acc.max(x));

    let range = &max - &min;
    let normalized = (data - &min) / &range;

    (normalized, min, max)
}

/// Detect outliers using IQR method
fn detect_outliers_iqr(data: &Array1<f64>) -> Vec<usize> {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    let q1 = sorted[n / 4];
    let q3 = sorted[3 * n / 4];
    let iqr = q3 - q1;

    let lower_bound = q1 - 1.5 * iqr;
    let upper_bound = q3 + 1.5 * iqr;

    data.iter()
        .enumerate()
        .filter(|(_, &val)| val < lower_bound || val > upper_bound)
        .map(|(idx, _)| idx)
        .collect()
}

/// Simple imputation with mean
fn impute_mean(data: &Array2<f64>, missing_value: f64) -> Array2<f64> {
    let mut imputed = data.clone();

    for col in 0..data.ncols() {
        let column = data.column(col);
        let valid_values: Vec<f64> = column.iter()
            .filter(|&&x| x != missing_value)
            .copied()
            .collect();

        if !valid_values.is_empty() {
            let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;

            for row in 0..data.nrows() {
                if imputed[[row, col]] == missing_value {
                    imputed[[row, col]] = mean;
                }
            }
        }
    }

    imputed
}

fn main() -> anyhow::Result<()> {
    println!("=== Data Preprocessing Example ===\n");

    let mut rng = StdRng::seed_from_u64(42);

    // Generate sample dataset with different scales
    println!("1. Feature Scaling\n");

    println!("   Generating data with different scales:");
    let n_samples = 100;

    // Feature 1: Age (20-60)
    let feature1 = Array1::random_using(n_samples, Uniform::new(20.0, 60.0), &mut rng);

    // Feature 2: Income (30,000-150,000)
    let feature2 = Array1::random_using(n_samples, Uniform::new(30000.0, 150000.0), &mut rng);

    // Feature 3: Score (0-100)
    let feature3 = Array1::random_using(n_samples, Uniform::new(0.0, 100.0), &mut rng);

    let data = ndarray::stack![Axis(1), feature1, feature2, feature3];

    println!("   Original data statistics:");
    println!("   Feature 1 (Age):    mean = {:.2}, std = {:.2}",
             data.column(0).mean().unwrap(),
             data.column(0).std(1.0));
    println!("   Feature 2 (Income): mean = {:.2}, std = {:.2}",
             data.column(1).mean().unwrap(),
             data.column(1).std(1.0));
    println!("   Feature 3 (Score):  mean = {:.2}, std = {:.2}\n",
             data.column(2).mean().unwrap(),
             data.column(2).std(1.0));

    // Standardization
    println!("   A) Standardization (Z-score normalization):");
    let (standardized, mean, std) = standardize(&data);
    println!("      After standardization:");
    println!("      Feature 1: mean = {:.4}, std = {:.4}",
             standardized.column(0).mean().unwrap(),
             standardized.column(0).std(1.0));
    println!("      Feature 2: mean = {:.4}, std = {:.4}",
             standardized.column(1).mean().unwrap(),
             standardized.column(1).std(1.0));
    println!("      Feature 3: mean = {:.4}, std = {:.4}\n",
             standardized.column(2).mean().unwrap(),
             standardized.column(2).std(1.0));

    // Min-Max Normalization
    println!("   B) Min-Max Normalization:");
    let (normalized, min, max) = min_max_normalize(&data);
    println!("      After normalization:");
    println!("      Feature 1: min = {:.4}, max = {:.4}",
             normalized.column(0).iter().fold(f64::INFINITY, |a, &b| a.min(b)),
             normalized.column(0).iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
    println!("      Feature 2: min = {:.4}, max = {:.4}",
             normalized.column(1).iter().fold(f64::INFINITY, |a, &b| a.min(b)),
             normalized.column(1).iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
    println!("      Feature 3: min = {:.4}, max = {:.4}\n",
             normalized.column(2).iter().fold(f64::INFINITY, |a, &b| a.min(b)),
             normalized.column(2).iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));

    // Outlier Detection
    println!("2. Outlier Detection\n");

    let mut data_with_outliers = Array1::random_using(100, StandardNormal, &mut rng);
    // Add some outliers
    data_with_outliers[0] = 10.0;
    data_with_outliers[1] = -8.0;
    data_with_outliers[2] = 12.0;

    let outliers = detect_outliers_iqr(&data_with_outliers);
    println!("   Detected {} outliers using IQR method", outliers.len());
    println!("   Outlier indices: {:?}", &outliers[..outliers.len().min(10)]);
    if outliers.len() > 10 {
        println!("   ... and {} more", outliers.len() - 10);
    }
    println!();

    // Missing Value Imputation
    println!("3. Handling Missing Values\n");

    let mut data_with_missing = Array2::random_using((20, 3), StandardNormal, &mut rng);
    let missing_value = -999.0;

    // Introduce some missing values
    data_with_missing[[2, 0]] = missing_value;
    data_with_missing[[5, 1]] = missing_value;
    data_with_missing[[8, 2]] = missing_value;
    data_with_missing[[12, 0]] = missing_value;

    let missing_count = data_with_missing.iter()
        .filter(|&&x| x == missing_value)
        .count();

    println!("   Dataset has {} missing values", missing_count);
    println!("   Applying mean imputation...");

    let imputed = impute_mean(&data_with_missing, missing_value);

    let remaining_missing = imputed.iter()
        .filter(|&&x| x == missing_value)
        .count();

    println!("   After imputation: {} missing values remain\n", remaining_missing);

    // Train/Test Split
    println!("4. Train/Test Split\n");

    let split_ratio = 0.8;
    let split_idx = (data.nrows() as f64 * split_ratio) as usize;

    println!("   Splitting data with {:.0}% train / {:.0}% test",
             split_ratio * 100.0,
             (1.0 - split_ratio) * 100.0);
    println!("   Total samples: {}", data.nrows());
    println!("   Training samples: {}", split_idx);
    println!("   Test samples: {}\n", data.nrows() - split_idx);

    // Feature Engineering Example
    println!("5. Feature Engineering\n");

    println!("   Creating polynomial features:");
    println!("   Original: [x₁, x₂]");
    println!("   Polynomial (degree 2): [x₁, x₂, x₁², x₁x₂, x₂²]");

    let sample: Array1<f64> = Array1::from_vec(vec![2.0, 3.0]);
    let poly_features = vec![
        sample[0],
        sample[1],
        sample[0].powi(2),
        sample[0] * sample[1],
        sample[1].powi(2),
    ];

    println!("   Example: {:?} → {:?}\n", sample.to_vec(), poly_features);

    println!("=== Example Complete! ===");
    println!("\nKey Takeaways:");
    println!("- Feature scaling is crucial for many ML algorithms");
    println!("- Standardization: Use for normally distributed data");
    println!("- Min-Max: Use when you need a specific range [0, 1]");
    println!("- Handle outliers carefully (they may be important!)");
    println!("- Missing values: Choose imputation based on data type");
    println!("- Always apply same preprocessing to train AND test data");
    println!("- Feature engineering can dramatically improve performance");

    Ok(())
}

//! # Principal Component Analysis (PCA) Example
//!
//! This example demonstrates dimensionality reduction using PCA with the Linfa library.
//! PCA is one of the most widely used unsupervised learning techniques.
//!
//! ## What is PCA?
//!
//! Principal Component Analysis is a dimensionality reduction technique that
//! transforms high-dimensional data into a lower-dimensional representation
//! while preserving as much variance (information) as possible.
//!
//! ## The Core Idea
//!
//! Imagine you have data in 3D space, but all points lie roughly on a 2D plane.
//! PCA finds that plane! It identifies the directions (principal components)
//! where the data varies the most, then projects the data onto those directions.
//!
//! ## How PCA Works
//!
//! 1. **Center the data**: Subtract the mean from each feature
//! 2. **Compute covariance matrix**: Measures how features vary together
//! 3. **Find eigenvectors and eigenvalues**:
//!    - Eigenvectors = principal components (directions)
//!    - Eigenvalues = variance along those directions
//! 4. **Sort by eigenvalues**: Largest eigenvalue = most variance
//! 5. **Project data**: Transform to new coordinate system
//!
//! ## Mathematical Formulation
//!
//! Given data matrix X (n × d):
//!
//! 1. Center: X_centered = X - mean(X)
//! 2. Covariance: Σ = (1/n) X_centered^T · X_centered
//! 3. Eigenvectors v₁, v₂, ..., vₐ of Σ (sorted by eigenvalue)
//! 4. Transform: X_new = X_centered · [v₁ v₂ ... vₖ]
//!
//! Where k < d is the reduced dimensionality.
//!
//! ## Explained Variance
//!
//! Each principal component explains a portion of the total variance:
//! ```
//! Explained Variance Ratio = λᵢ / Σλⱼ
//! ```
//!
//! Where λᵢ is the i-th eigenvalue.
//!
//! ## Use Cases
//!
//! - **Visualization**: Reduce to 2D or 3D for plotting
//! - **Noise reduction**: Remove components with low variance
//! - **Feature extraction**: Create new features that capture variance
//! - **Data compression**: Reduce storage and computation
//! - **Preprocessing**: For neural networks, clustering, etc.
//! - **Image compression**: JPEG uses similar techniques
//!
//! ## Advantages
//!
//! - Reduces dimensionality without supervised labels
//! - Removes correlated features
//! - Can speed up downstream algorithms
//! - Helps with visualization
//! - Reduces overfitting by removing noise
//!
//! ## Limitations
//!
//! - Assumes linear relationships
//! - Loses interpretability (new features are combinations)
//! - Sensitive to feature scaling (must standardize!)
//! - May discard useful information for classification
//! - All principal components are used (unlike feature selection)

use linfa::prelude::*;
use linfa_reduction::Pca;
use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() -> anyhow::Result<()> {
    println!("=== Principal Component Analysis (PCA) Example ===\n");

    // Set random seed for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // Generate high-dimensional data with underlying low-dimensional structure
    println!("1. Generating high-dimensional synthetic data...");
    let n_samples = 500;
    let n_components = 2; // True underlying dimensions
    let n_features = 10;   // Observed dimensions

    // Generate low-dimensional "true" data
    let true_data = Array2::random_using((n_samples, n_components), StandardNormal, &mut rng);

    // Create a random projection matrix to embed into high dimensions
    let projection = Array2::random_using((n_components, n_features), StandardNormal, &mut rng);

    // Project to high dimensions and add noise
    let noise = Array2::random_using((n_samples, n_features), StandardNormal, &mut rng) * 0.1;
    let features = true_data.dot(&projection) + noise;

    println!("   - Samples: {}", n_samples);
    println!("   - Original features: {}", n_features);
    println!("   - True underlying dimensions: {}", n_components);
    println!("   - Data has hidden low-dimensional structure!\n");

    // Create dataset
    let dataset = DatasetBase::from(features.clone());

    // Apply PCA to find the principal components
    println!("2. Applying PCA...");

    // First, let's see variance explained by each component
    let full_pca = Pca::params(n_features)
        .fit(&dataset)?;

    let explained_variance = full_pca.explained_variance();
    let total_variance: f64 = explained_variance.iter().sum();

    println!("\n   Variance explained by each component:");
    let mut cumulative_variance = 0.0;
    for (i, &var) in explained_variance.iter().enumerate() {
        let ratio = var / total_variance;
        cumulative_variance += ratio;
        println!("   PC{}: {:.4} ({:.1}%) - Cumulative: {:.1}%",
                 i + 1,
                 var,
                 ratio * 100.0,
                 cumulative_variance * 100.0);

        // Stop after showing the most important ones
        if i >= 5 || cumulative_variance > 0.99 {
            if i < n_features - 1 {
                println!("   ... ({} more components)", n_features - i - 1);
            }
            break;
        }
    }

    // Determine how many components to keep (95% variance)
    let threshold = 0.95;
    let mut cumsum = 0.0;
    let mut n_components_to_keep = 0;

    for &var in explained_variance.iter() {
        cumsum += var / total_variance;
        n_components_to_keep += 1;
        if cumsum >= threshold {
            break;
        }
    }

    println!("\n   To preserve {}% of variance, we need {} components",
             threshold * 100.0,
             n_components_to_keep);

    // Apply PCA with reduced dimensions
    println!("\n3. Reducing dimensionality from {} to {} dimensions...",
             n_features,
             n_components_to_keep);

    let pca = Pca::params(n_components_to_keep)
        .fit(&dataset)?;

    let reduced_features = pca.predict(features.clone()).records().clone();

    println!("   - Original shape: {} × {}", features.nrows(), features.ncols());
    println!("   - Reduced shape: {} × {}", reduced_features.nrows(), reduced_features.ncols());
    println!("   - Compression ratio: {:.1}x",
             n_features as f64 / n_components_to_keep as f64);

    // Show variance preserved
    let variance_ratio = pca.explained_variance().iter().sum::<f64>() / total_variance;
    println!("   - Variance preserved: {:.2}%\n", variance_ratio * 100.0);

    // Demonstrate reconstruction
    println!("4. Reconstructing original data from reduced representation...");

    // Transform back to original space
    let reconstructed = pca.inverse_transform(reduced_features.clone())?;

    // Calculate reconstruction error
    let reconstruction_error = (&features - &reconstructed)
        .mapv(|x| x.powi(2))
        .mean()
        .unwrap()
        .sqrt();

    println!("   - Reconstruction RMSE: {:.4}", reconstruction_error);
    println!("   - (Lower is better - measures information loss)\n");

    // Show sample transformations
    println!("5. Sample transformations (first 5 samples):");
    println!("\n   Original features ({}D) → Reduced features ({}D):",
             n_features, n_components_to_keep);

    for i in 0..5 {
        print!("   Sample {}: [", i + 1);
        for j in 0..n_features.min(4) {
            print!("{:.2}", features[[i, j]]);
            if j < n_features.min(4) - 1 {
                print!(", ");
            }
        }
        if n_features > 4 {
            print!(", ...");
        }
        print!("] → [");

        for j in 0..n_components_to_keep {
            print!("{:.2}", reduced_features[[i, j]]);
            if j < n_components_to_keep - 1 {
                print!(", ");
            }
        }
        println!("]");
    }

    // Practical example: reduce to 2D for visualization
    println!("\n6. Reducing to 2D for visualization...");
    let pca_2d = Pca::params(2)
        .fit(&dataset)?;

    let features_2d = pca_2d.predict(features.clone()).records().clone();

    println!("   - Reduced {} features to 2 principal components", n_features);
    println!("   - This can now be plotted in 2D!");
    println!("   - Variance explained: {:.1}%",
             (pca_2d.explained_variance().iter().sum::<f64>() / total_variance) * 100.0);

    println!("\n   Sample 2D coordinates (first 5):");
    for i in 0..5 {
        println!("   Sample {}: ({:.2}, {:.2})",
                 i + 1,
                 features_2d[[i, 0]],
                 features_2d[[i, 1]]);
    }

    println!("\n=== Example Complete! ===");
    println!("\nKey Takeaways:");
    println!("- PCA finds directions of maximum variance");
    println!("- Reduces dimensionality while preserving information");
    println!("- First few components often capture most variance");
    println!("- Useful for visualization, noise reduction, and compression");
    println!("- Always standardize features before applying PCA!");

    Ok(())
}

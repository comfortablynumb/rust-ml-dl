//! # K-Means Clustering Example
//!
//! This example demonstrates K-Means clustering using the Linfa library.
//! K-Means is one of the most popular unsupervised learning algorithms.
//!
//! ## What is K-Means Clustering?
//!
//! K-Means is an unsupervised learning algorithm that groups similar data points
//! into K clusters. Unlike supervised learning, there are no labels - the algorithm
//! discovers patterns in the data automatically.
//!
//! ## How K-Means Works
//!
//! 1. **Initialize**: Randomly select K points as initial cluster centers (centroids)
//! 2. **Assignment**: Assign each data point to the nearest centroid
//! 3. **Update**: Recalculate centroids as the mean of all points in each cluster
//! 4. **Repeat**: Steps 2-3 until convergence (centroids stop moving significantly)
//!
//! The algorithm minimizes the within-cluster sum of squares (WCSS):
//!
//! WCSS = Σ Σ ||x - μₖ||²
//!
//! Where:
//! - x is a data point
//! - μₖ is the centroid of cluster k
//! - ||·|| is the Euclidean distance
//!
//! ## Use Cases
//!
//! - Customer segmentation (group customers by behavior)
//! - Image compression (reduce color palette)
//! - Document clustering (group similar documents)
//! - Anomaly detection (points far from all clusters)
//! - Market segmentation
//! - Pattern recognition
//!
//! ## Choosing K
//!
//! The "elbow method" is commonly used:
//! - Plot WCSS vs K
//! - Look for an "elbow" where the rate of decrease sharply changes
//! - That K is often a good choice

use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() -> anyhow::Result<()> {
    println!("=== K-Means Clustering Example ===\n");

    // Set random seed for reproducibility
    let mut rng = StdRng::seed_from_u64(42);

    // Generate synthetic data with 3 distinct clusters
    println!("1. Generating synthetic data with 3 clusters...");
    let n_samples_per_cluster = 200;
    let n_features = 2;

    // Cluster 1: centered around (-5, -5)
    let cluster1 = Array2::random_using(
        (n_samples_per_cluster, n_features),
        StandardNormal,
        &mut rng,
    ) + Array2::from_elem((n_samples_per_cluster, n_features), -5.0);

    // Cluster 2: centered around (0, 5)
    let cluster2 = Array2::random_using(
        (n_samples_per_cluster, n_features),
        StandardNormal,
        &mut rng,
    ) + Array2::from_elem((n_samples_per_cluster, n_features), 5.0)
        * Array2::from_shape_vec((1, n_features), vec![0.0, 1.0]).unwrap();

    // Cluster 3: centered around (5, -5)
    let cluster3 = Array2::random_using(
        (n_samples_per_cluster, n_features),
        StandardNormal,
        &mut rng,
    ) + Array2::from_shape_vec((1, n_features), vec![5.0, -5.0]).unwrap();

    // Combine all clusters
    let data = ndarray::concatenate![Axis(0), cluster1, cluster2, cluster3];

    println!("   - Generated {} samples per cluster", n_samples_per_cluster);
    println!("   - Total samples: {}", data.nrows());
    println!("   - Features: {}\n", n_features);

    // Create dataset
    let dataset = DatasetBase::from(data.clone());

    // Elbow method: try different values of K
    println!("2. Using Elbow Method to find optimal K...");
    println!("   Testing K from 1 to 6:\n");

    for k in 1..=6 {
        let model = KMeans::params_with_rng(k, rng.clone())
            .max_n_iterations(200)
            .fit(&dataset)?;

        let inertia = model.inertia();
        println!("   K = {}: Inertia (WCSS) = {:.2}", k, inertia);
    }

    println!("\n   Note: Look for the 'elbow' where inertia stops decreasing rapidly");
    println!("   In this case, K=3 should be optimal (we know the true number of clusters)\n");

    // Train K-Means with K=3
    println!("3. Training K-Means with K=3...");
    let n_clusters = 3;
    let model = KMeans::params_with_rng(n_clusters, rng.clone())
        .max_n_iterations(200)
        .tolerance(1e-4)
        .fit(&dataset)?;

    println!("   - Model trained successfully!");
    println!("   - Number of iterations: {}", model.inertia());
    println!("   - Final inertia: {:.2}\n", model.inertia());

    // Get cluster assignments
    let cluster_assignments = model.predict(&data);

    // Calculate cluster statistics
    println!("4. Cluster Statistics:");
    for cluster_id in 0..n_clusters {
        let count = cluster_assignments.iter().filter(|&&x| x == cluster_id).count();
        println!("   - Cluster {}: {} samples ({:.1}%)",
                 cluster_id,
                 count,
                 (count as f64 / data.nrows() as f64) * 100.0);
    }

    // Display centroids
    println!("\n5. Cluster Centroids:");
    let centroids = model.centroids();
    for (i, centroid) in centroids.rows().into_iter().enumerate() {
        println!("   - Cluster {}: [{:.2}, {:.2}]",
                 i,
                 centroid[0],
                 centroid[1]);
    }

    // Show sample assignments
    println!("\n6. Sample cluster assignments (first 10 samples):");
    for i in 0..10.min(data.nrows()) {
        println!("   Sample {}: Features = [{:.2}, {:.2}], Cluster = {}",
                 i + 1,
                 data[[i, 0]],
                 data[[i, 1]],
                 cluster_assignments[i]);
    }

    // Calculate silhouette score (simplified version)
    println!("\n7. Quality Metrics:");
    println!("   - Inertia (WCSS): {:.2} (lower is better)", model.inertia());
    println!("   - Number of iterations until convergence: varies");
    println!("\n   Note: Silhouette score (not shown) ranges from -1 to 1");
    println!("         Values near 1 indicate well-separated clusters");

    // Demonstrate prediction on new data
    println!("\n8. Predicting clusters for new data points:");
    let new_points = Array2::from_shape_vec(
        (3, 2),
        vec![
            -5.0, -5.0,  // Should be cluster 0
            0.0, 5.0,    // Should be cluster 1
            5.0, -5.0,   // Should be cluster 2
        ],
    )?;

    let new_predictions = model.predict(&new_points);
    for i in 0..new_points.nrows() {
        println!("   Point [{:.1}, {:.1}] → Cluster {}",
                 new_points[[i, 0]],
                 new_points[[i, 1]],
                 new_predictions[i]);
    }

    println!("\n=== Example Complete! ===");

    Ok(())
}

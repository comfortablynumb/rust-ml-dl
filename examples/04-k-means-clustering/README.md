# K-Means Clustering Example

This example demonstrates unsupervised learning using K-Means clustering in Rust with the Linfa library.

## Overview

K-Means is an unsupervised learning algorithm that partitions data into K distinct clusters without using labeled data.

## Algorithm Steps

1. **Initialization**: Select K initial centroids (cluster centers)
2. **Assignment**: Assign each point to the nearest centroid
3. **Update**: Recompute centroids as the mean of assigned points
4. **Repeat**: Continue assignment and update until convergence

## Mathematical Foundation

K-Means minimizes the **Within-Cluster Sum of Squares (WCSS)**:

```
WCSS = Σₖ Σₓ∈Cₖ ||x - μₖ||²
```

Where:
- `Cₖ` is cluster k
- `μₖ` is the centroid of cluster k
- `||·||` denotes Euclidean distance

## Running the Example

```bash
cargo run --package k-means-clustering
```

## What the Example Does

1. **Generates synthetic data** with 3 well-separated clusters
2. **Demonstrates the elbow method** for choosing optimal K
3. **Trains K-Means** with K=3
4. **Analyzes cluster assignments** and centroids
5. **Predicts clusters** for new data points

## Key Concepts

### Choosing K: The Elbow Method

Plot inertia (WCSS) vs number of clusters:
- Run K-Means for different values of K
- Calculate inertia for each K
- Look for the "elbow" where inertia stops decreasing rapidly

### Evaluation Metrics

- **Inertia (WCSS)**: Sum of squared distances to nearest centroid (lower is better)
- **Silhouette Score**: Measures how similar points are to their own cluster vs other clusters (-1 to 1, higher is better)
- **Davies-Bouldin Index**: Average similarity between clusters (lower is better)

### When to Use K-Means

- Customer/market segmentation
- Image compression and segmentation
- Document clustering
- Anomaly detection (outliers)
- Feature learning
- Data exploration

### Advantages

1. **Simple and intuitive**: Easy to understand and implement
2. **Fast**: O(n·k·i·d) where n=samples, k=clusters, i=iterations, d=dimensions
3. **Scales well**: Can handle large datasets
4. **Guaranteed to converge**: Though not necessarily to global optimum

### Limitations

1. **Must specify K**: Need to know or guess number of clusters
2. **Sensitive to initialization**: Can converge to local minima (use multiple runs)
3. **Assumes spherical clusters**: Works best when clusters are roughly circular
4. **Sensitive to outliers**: Outliers can skew centroids
5. **Doesn't work well with**: Non-convex shapes, different sizes/densities

## Variations and Improvements

- **K-Means++**: Smarter initialization strategy
- **Mini-batch K-Means**: Faster variant using random samples
- **K-Medoids**: Uses actual data points as centroids (more robust to outliers)
- **Fuzzy C-Means**: Soft clustering (points can belong to multiple clusters)

## Alternatives

- **DBSCAN**: Density-based, doesn't require K, handles arbitrary shapes
- **Hierarchical Clustering**: Creates a dendrogram, no need to specify K upfront
- **Gaussian Mixture Models**: Probabilistic clustering, models cluster shape
- **Mean Shift**: Mode-seeking algorithm, automatically finds K

## Libraries Used

- `linfa`: Machine learning framework for Rust
- `linfa-clustering`: Clustering algorithms implementation
- `ndarray`: N-dimensional arrays for numerical computing

## Further Reading

- [K-Means Algorithm](https://en.wikipedia.org/wiki/K-means_clustering)
- [Choosing K: Elbow Method](https://en.wikipedia.org/wiki/Elbow_method_(clustering))
- [Silhouette Score](https://en.wikipedia.org/wiki/Silhouette_(clustering))

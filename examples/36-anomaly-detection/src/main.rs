// ============================================================================
// Anomaly Detection
// ============================================================================
//
// Identify unusual patterns that differ from the majority of the data.
// Critical for fraud detection, system monitoring, quality control, and security.
//
// WHAT IS AN ANOMALY?
// -------------------
//
// Anomaly (outlier): Data point significantly different from normal patterns
//
// Types:
// 1. Point anomalies: Single unusual data point
//    • Credit card charge of $10,000 when usually $50
//
// 2. Contextual anomalies: Unusual in specific context
//    • Temperature 35°C normal in summer, anomaly in winter
//
// 3. Collective anomalies: Collection of points unusual together
//    • Sequence of actions indicating intrusion
//
// WHY DIFFICULT?
// --------------
//
// • Normal behavior hard to define precisely
// • Boundary between normal and anomalous fuzzy
// • Anomalies rare (imbalanced: 99.9% normal, 0.1% anomaly)
// • Constantly evolving (what's normal today may not be tomorrow)
// • High cost of false positives/negatives
//
// APPROACHES:
// -----------
//
// 1. STATISTICAL:
//    • Assume normal data follows distribution (Gaussian)
//    • Flag points far from mean (>3 standard deviations)
//
// 2. DISTANCE-BASED:
//    • K-Nearest Neighbors (KNN)
//    • Points far from neighbors are anomalies
//
// 3. DENSITY-BASED:
//    • Local Outlier Factor (LOF)
//    • Points in low-density regions are anomalies
//
// 4. ISOLATION-BASED:
//    • Isolation Forest
//    • Anomalies easier to isolate (fewer splits needed)
//
// 5. CLUSTERING-BASED:
//    • K-Means, DBSCAN
//    • Points far from cluster centers or in small clusters
//
// 6. RECONSTRUCTION-BASED:
//    • Autoencoders
//    • Anomalies hard to reconstruct (high error)
//
// 7. ONE-CLASS CLASSIFICATION:
//    • One-Class SVM
//    • Learn boundary around normal data
//
// ============================================================================
// ISOLATION FOREST
// ============================================================================
//
// Key insight: Anomalies are easier to isolate than normal points
//
// INTUITION:
// ----------
//
// Imagine randomly partitioning data:
//   • Normal point: Surrounded by many similar points
//     → Need many splits to isolate it
//   • Anomaly: Few similar points around it
//     → Need few splits to isolate it
//
// Example:
// ```
// Normal cluster: (x=50±10, y=50±10)
// Anomaly: (x=90, y=90)
//
// Random split at x=70:
//   • Anomaly isolated immediately!
//   • Normal cluster needs more splits
// ```
//
// ALGORITHM:
// ----------
//
// Training:
// 1. Build ensemble of isolation trees (100-200 trees)
// 2. For each tree:
//    a. Sample data randomly
//    b. Recursively partition:
//       - Pick random feature
//       - Pick random split value between min and max
//       - Split data
//       - Repeat until:
//         * Each point isolated, OR
//         * Reached max depth
//
// Scoring:
// 1. For test point x:
//    Pass through all trees
//    Record path length to isolate x in each tree
// 2. Average path length across trees
// 3. Normalize:
//    score(x) = 2^(-avg_path_length / c(n))
//    • c(n): Average path length for normal data
//    • score ≈ 1: Anomaly (short paths)
//    • score ≈ 0: Normal (long paths)
//
// WHY IT WORKS:
// -------------
//
// • Anomalies have distinctive attribute values
//   → Isolated close to root (short paths)
// • Normal points have similar values
//   → Need many splits to isolate (long paths)
//
// ADVANTAGES:
// -----------
//
// ✅ Fast: O(n log n) training, O(log n) prediction
// ✅ Handles high dimensions
// ✅ No distance computation needed
// ✅ Works without knowing distribution
// ✅ Few parameters to tune
//
// DISADVANTAGES:
// --------------
//
// ❌ Random: Results vary across runs (use ensemble)
// ❌ Less effective when anomalies cluster together
// ❌ Binary decision (anomaly or not), not severity score
//
// ============================================================================
// AUTOENCODER-BASED DETECTION
// ============================================================================
//
// Key insight: Neural network trained to reconstruct normal data will
// fail to reconstruct anomalies (high reconstruction error)
//
// ARCHITECTURE:
// -------------
//
// Autoencoder:
// ```
// Input (n dims)
//   ↓
// Encoder: Compress to latent space (k dims, k << n)
//   ↓
// Bottleneck (compressed representation)
//   ↓
// Decoder: Reconstruct to original space (n dims)
//   ↓
// Output (n dims, should ≈ Input)
// ```
//
// Example:
// ```
// Input: 784 dims (28×28 image)
//   ↓
// Dense(128) → ReLU
//   ↓
// Dense(64) → ReLU
//   ↓
// Dense(32) → ReLU  ← Bottleneck
//   ↓
// Dense(64) → ReLU
//   ↓
// Dense(128) → ReLU
//   ↓
// Dense(784) → Sigmoid
//   ↓
// Output: 784 dims (reconstructed image)
// ```
//
// TRAINING:
// ---------
//
// Train on normal data only!
//
// Loss: Reconstruction error
//   L = ||x - x̂||²
//   • x: Input
//   • x̂: Reconstructed output
//
// Network learns to:
//   • Compress normal patterns into latent space
//   • Reconstruct from compressed representation
//
// ANOMALY DETECTION:
// ------------------
//
// 1. Pass test point through autoencoder
// 2. Compute reconstruction error: e = ||x - x̂||²
// 3. If e > threshold: Anomaly
//    If e ≤ threshold: Normal
//
// Why it works:
//   • Normal data: Seen during training
//     → Low reconstruction error
//   • Anomaly: Not seen during training
//     → High reconstruction error (can't reconstruct well)
//
// THRESHOLD SELECTION:
// --------------------
//
// Methods:
//
// 1. Percentile on validation set:
//    • Compute errors on normal validation data
//    • threshold = 95th percentile
//    • Flags top 5% as anomalies
//
// 2. Statistical (mean + k×std):
//    • threshold = μ + 3σ
//    • Gaussian assumption
//
// 3. Domain knowledge:
//    • Set based on tolerable false positive rate
//
// VARIANTS:
// ---------
//
// 1. Variational Autoencoder (VAE):
//    • Probabilistic latent space
//    • Better for capturing distributions
//
// 2. Denoising Autoencoder:
//    • Add noise to input, train to remove it
//    • More robust features
//
// 3. LSTM Autoencoder:
//    • For time series
//    • Encode sequence, decode sequence
//    • Anomalous sequences hard to reconstruct
//
// ADVANTAGES:
// -----------
//
// ✅ Learns complex patterns automatically
// ✅ Handles high-dimensional data
// ✅ Unsupervised (no anomaly labels needed)
// ✅ Scalable to large datasets
// ✅ Can capture non-linear relationships
//
// DISADVANTAGES:
// --------------
//
// ❌ Requires lots of normal data
// ❌ Threshold tuning needed
// ❌ Black box (hard to interpret)
// ❌ Can overfit to normal data
// ❌ Computationally expensive
//
// ============================================================================
// ONE-CLASS SVM
// ============================================================================
//
// Key insight: Learn boundary around normal data in feature space
//
// IDEA:
// -----
//
// Map data to high-dimensional space (using kernel)
// Find smallest hypersphere that contains most normal data
// Points outside hypersphere are anomalies
//
// FORMULATION:
// ------------
//
// Learn decision function:
//   f(x) = sign(w^T φ(x) - ρ)
//
//   • φ(x): Feature map (kernel trick)
//   • w: Normal vector to hyperplane
//   • ρ: Offset
//   • f(x) > 0: Normal
//   • f(x) < 0: Anomaly
//
// Optimization:
//   Maximize margin (distance to origin)
//   Allow some outliers (soft margin, controlled by ν)
//
// KERNELS:
// --------
//
// 1. RBF (Radial Basis Function):
//    K(x, y) = exp(-γ ||x - y||²)
//    • Most common
//    • γ controls influence radius
//
// 2. Linear:
//    K(x, y) = x^T y
//    • Fastest
//    • For linearly separable data
//
// 3. Polynomial:
//    K(x, y) = (x^T y + c)^d
//    • For non-linear patterns
//
// PARAMETERS:
// -----------
//
// • ν (nu): Upper bound on fraction of outliers
//   - ν = 0.01: Allow 1% outliers
//   - ν = 0.1: Allow 10% outliers
//   - Typical: 0.01 - 0.1
//
// • γ (gamma): RBF kernel parameter
//   - High γ: Tight fit, may overfit
//   - Low γ: Loose fit, may underfit
//   - Typical: 1 / n_features
//
// ADVANTAGES:
// -----------
//
// ✅ Mathematically well-founded
// ✅ Kernel trick handles non-linearity
// ✅ Works with small datasets
// ✅ Fewer parameters than deep learning
//
// DISADVANTAGES:
// --------------
//
// ❌ Kernel and parameter selection critical
// ❌ O(n²) or O(n³) training time
// ❌ Memory intensive for large datasets
// ❌ Sensitive to feature scaling
//
// ============================================================================
// EVALUATION METRICS
// ============================================================================
//
// Challenge: Anomalies rare, imbalanced data
//
// Metrics:
//
// 1. PRECISION:
//    • Of flagged anomalies, how many are true anomalies?
//    • Precision = TP / (TP + FP)
//    • Important when false alarms costly
//
// 2. RECALL (Detection Rate):
//    • Of true anomalies, how many were detected?
//    • Recall = TP / (TP + FN)
//    • Important when missing anomalies costly
//
// 3. F1-SCORE:
//    • Harmonic mean of precision and recall
//    • F1 = 2 × (Precision × Recall) / (Precision + Recall)
//
// 4. AUC-ROC:
//    • Area under ROC curve
//    • Threshold-independent
//    • Good for comparing methods
//
// 5. AUC-PR:
//    • Area under Precision-Recall curve
//    • Better for imbalanced data than AUC-ROC
//
// APPLICATIONS:
// -------------
//
// 1. FRAUD DETECTION:
//    • Credit card: Unusual transactions
//    • Insurance: Fraudulent claims
//    • Online: Fake accounts, bots
//
//    Challenges:
//      • Highly imbalanced (<0.1% fraud)
//      • Fraudsters adapt (adversarial)
//      • Real-time detection needed
//
// 2. SYSTEM MONITORING:
//    • Server: CPU/memory spikes
//    • Network: DDoS attacks, intrusions
//    • IoT: Sensor failures
//
//    Challenges:
//      • High-dimensional (many metrics)
//      • Time series patterns
//      • Concept drift (normal changes over time)
//
// 3. MANUFACTURING/QUALITY CONTROL:
//    • Defective products
//    • Equipment failures
//    • Process anomalies
//
//    Challenges:
//      • Different defect types
//      • Small training datasets
//      • Real-time inspection
//
// 4. HEALTHCARE:
//    • Disease outbreak detection
//    • Unusual patient vitals
//    • Medical imaging (tumors)
//
//    Challenges:
//      • Patient privacy
//      • High stakes (life/death)
//      • Interpretability required
//
// 5. CYBERSECURITY:
//    • Malware detection
//    • Intrusion detection
//    • User behavior analytics
//
//    Challenges:
//      • Adversarial attacks
//      • Zero-day exploits
//      • Low latency requirements
//
// PRACTICAL TIPS:
// ---------------
//
// 1. Feature engineering is critical:
//    • Domain knowledge helps
//    • Try different representations
//    • Normalize/standardize features
//
// 2. Use ensemble of methods:
//    • Combine Isolation Forest + Autoencoder
//    • More robust than single method
//
// 3. Set threshold carefully:
//    • Consider cost of false positives vs false negatives
//    • Use validation set
//
// 4. Handle concept drift:
//    • Retrain periodically
//    • Online learning
//    • Monitor performance
//
// 5. Incorporate human feedback:
//    • Active learning
//    • Refine models based on expert labels
//
// 6. Explain detections:
//    • Which features contributed?
//    • Similar historical anomalies?
//    • Builds trust
//
// KEY TAKEAWAYS:
// --------------
//
// ✓ Anomaly detection finds unusual patterns in data
// ✓ Isolation Forest: Anomalies easier to isolate (fewer splits)
// ✓ Autoencoder: Anomalies have high reconstruction error
// ✓ One-Class SVM: Learn boundary around normal data
// ✓ Applications: Fraud, monitoring, quality control, security
// ✓ Challenges: Imbalanced data, concept drift, threshold tuning
// ✓ Ensemble methods often best in practice
//
// ============================================================================

use ndarray::Array1;
use rand::Rng;

/// Simple statistical anomaly detection using z-score
fn statistical_anomaly_detection(data: &[f32], threshold: f32) -> Vec<bool> {
    // Compute mean and std
    let n = data.len() as f32;
    let mean: f32 = data.iter().sum::<f32>() / n;
    let variance: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std_dev = variance.sqrt();

    // Flag points with |z-score| > threshold
    data.iter()
        .map(|&x| {
            let z_score = (x - mean).abs() / std_dev;
            z_score > threshold
        })
        .collect()
}

/// Simple distance-based anomaly detection (k-NN)
fn knn_anomaly_detection(data: &[f32], k: usize, threshold: f32) -> Vec<bool> {
    data.iter()
        .enumerate()
        .map(|(i, &x)| {
            // Compute distances to all other points
            let mut distances: Vec<f32> = data.iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, &y)| (x - y).abs())
                .collect();

            // Sort and take k nearest
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let avg_distance: f32 = distances.iter().take(k).sum::<f32>() / k as f32;

            avg_distance > threshold
        })
        .collect()
}

/// Simplified isolation tree node
enum IsolationTreeNode {
    Leaf { size: usize },
    Split { threshold: f32, left: Box<IsolationTreeNode>, right: Box<IsolationTreeNode> },
}

impl IsolationTreeNode {
    /// Build isolation tree recursively
    fn build(data: &[f32], max_depth: usize, current_depth: usize) -> Self {
        if current_depth >= max_depth || data.len() <= 1 {
            return IsolationTreeNode::Leaf { size: data.len() };
        }

        // Random split
        let mut rng = rand::thread_rng();
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        if (max - min).abs() < 1e-6 {
            return IsolationTreeNode::Leaf { size: data.len() };
        }

        let threshold = rng.gen_range(min..max);

        // Split data
        let left_data: Vec<f32> = data.iter().filter(|&&x| x < threshold).copied().collect();
        let right_data: Vec<f32> = data.iter().filter(|&&x| x >= threshold).copied().collect();

        if left_data.is_empty() || right_data.is_empty() {
            return IsolationTreeNode::Leaf { size: data.len() };
        }

        IsolationTreeNode::Split {
            threshold,
            left: Box::new(Self::build(&left_data, max_depth, current_depth + 1)),
            right: Box::new(Self::build(&right_data, max_depth, current_depth + 1)),
        }
    }

    /// Compute path length for a point
    fn path_length(&self, x: f32, current_depth: usize) -> f32 {
        match self {
            IsolationTreeNode::Leaf { size } => {
                current_depth as f32 + Self::average_path_length(*size)
            }
            IsolationTreeNode::Split { threshold, left, right } => {
                if x < *threshold {
                    left.path_length(x, current_depth + 1)
                } else {
                    right.path_length(x, current_depth + 1)
                }
            }
        }
    }

    /// Average path length in BST of size n
    fn average_path_length(n: usize) -> f32 {
        if n <= 1 {
            0.0
        } else {
            let n_f = n as f32;
            2.0 * ((n_f - 1.0).ln() + 0.5772156649) - 2.0 * (n_f - 1.0) / n_f
        }
    }
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("Anomaly Detection");
    println!("{}", "=".repeat(70));
    println!();

    // Generate data: mostly normal, few anomalies
    let mut rng = rand::thread_rng();
    let mut data: Vec<f32> = (0..100)
        .map(|_| rng.gen_range(45.0..55.0)) // Normal: mean=50, range=±5
        .collect();

    // Add anomalies
    data.push(80.0); // Anomaly 1
    data.push(15.0); // Anomaly 2
    data.push(90.0); // Anomaly 3

    println!("DATASET:");
    println!("--------");
    println!("Total points: {}", data.len());
    println!("Normal points: ~100 (45-55 range)");
    println!("Anomalies: 3 (15, 80, 90)");
    println!();

    // 1. Statistical (Z-score)
    println!("1. STATISTICAL ANOMALY DETECTION (Z-score):");
    println!("--------------------------------------------");
    let threshold_z = 2.5;
    let stat_anomalies = statistical_anomaly_detection(&data, threshold_z);
    let stat_count = stat_anomalies.iter().filter(|&&x| x).count();
    println!("Threshold: {} standard deviations", threshold_z);
    println!("Detected anomalies: {}", stat_count);

    let detected_values: Vec<f32> = data.iter()
        .zip(&stat_anomalies)
        .filter(|(_, &is_anomaly)| is_anomaly)
        .map(|(&val, _)| val)
        .collect();
    println!("Values: {:?}", detected_values);
    println!();

    // 2. Distance-based (k-NN)
    println!("2. DISTANCE-BASED (K-NEAREST NEIGHBORS):");
    println!("-----------------------------------------");
    let k = 5;
    let threshold_dist = 10.0;
    let knn_anomalies = knn_anomaly_detection(&data, k, threshold_dist);
    let knn_count = knn_anomalies.iter().filter(|&&x| x).count();
    println!("k: {}, Distance threshold: {}", k, threshold_dist);
    println!("Detected anomalies: {}", knn_count);

    let detected_values: Vec<f32> = data.iter()
        .zip(&knn_anomalies)
        .filter(|(_, &is_anomaly)| is_anomaly)
        .map(|(&val, _)| val)
        .collect();
    println!("Values: {:?}", detected_values);
    println!();

    // 3. Isolation Forest (simplified)
    println!("3. ISOLATION FOREST:");
    println!("--------------------");
    let n_trees = 10;
    let max_depth = 8;
    println!("Trees: {}, Max depth: {}", n_trees, max_depth);

    // Build trees
    let trees: Vec<IsolationTreeNode> = (0..n_trees)
        .map(|_| {
            // Sample subset
            let mut sample: Vec<f32> = data.iter()
                .filter(|_| rng.gen_bool(0.5))
                .copied()
                .collect();
            if sample.is_empty() {
                sample = data.clone();
            }
            IsolationTreeNode::build(&sample, max_depth, 0)
        })
        .collect();

    // Compute anomaly scores
    let scores: Vec<f32> = data.iter()
        .map(|&x| {
            let avg_path: f32 = trees.iter()
                .map(|tree| tree.path_length(x, 0))
                .sum::<f32>() / n_trees as f32;

            // Anomaly score: 2^(-avg_path / c(n))
            let c_n = IsolationTreeNode::average_path_length(data.len());
            2.0_f32.powf(-avg_path / c_n)
        })
        .collect();

    // Anomalies have scores close to 1
    let score_threshold = 0.6;
    let if_anomalies: Vec<bool> = scores.iter().map(|&s| s > score_threshold).collect();
    let if_count = if_anomalies.iter().filter(|&&x| x).count();
    println!("Score threshold: {}", score_threshold);
    println!("Detected anomalies: {}", if_count);

    let detected_with_scores: Vec<(f32, f32)> = data.iter()
        .zip(&scores)
        .zip(&if_anomalies)
        .filter(|((_, _), &is_anomaly)| is_anomaly)
        .map(|((&val, &score), _)| (val, score))
        .collect();
    println!("Values (score): {:?}", detected_with_scores.iter()
        .map(|(v, s)| format!("{:.1} ({:.2})", v, s))
        .collect::<Vec<_>>());
    println!();

    println!("{}", "=".repeat(70));
    println!("KEY CONCEPTS DEMONSTRATED:");
    println!("{}", "=".repeat(70));
    println!("✓ Statistical: Z-score based detection");
    println!("✓ Distance-based: K-nearest neighbors");
    println!("✓ Isolation Forest: Anomalies easier to isolate");
    println!();
    println!("ANOMALY DETECTION METHODS:");
    println!("  • Statistical: Z-score, Grubbs' test");
    println!("  • Distance: KNN, LOF (Local Outlier Factor)");
    println!("  • Isolation: Isolation Forest");
    println!("  • Clustering: K-Means, DBSCAN");
    println!("  • Reconstruction: Autoencoders");
    println!("  • One-class: One-Class SVM, SVDD");
    println!();
    println!("APPLICATIONS:");
    println!("  • Fraud detection: Credit cards, insurance");
    println!("  • System monitoring: Servers, networks, IoT");
    println!("  • Quality control: Manufacturing defects");
    println!("  • Healthcare: Disease outbreak, patient vitals");
    println!("  • Cybersecurity: Intrusion, malware detection");
    println!();
    println!("CHALLENGES:");
    println!("  • Imbalanced data (<1% anomalies typical)");
    println!("  • Threshold selection (trade-off FP vs FN)");
    println!("  • Concept drift (normal changes over time)");
    println!("  • Interpretability (why is it anomalous?)");
    println!();
}

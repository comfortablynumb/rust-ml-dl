# Anomaly Detection

Identify unusual patterns that differ from the majority of the data.

## Overview

Anomaly detection is critical for fraud detection, system monitoring, quality control, and cybersecurity. It identifies rare events, outliers, and unusual behavior that may indicate problems or opportunities.

## Running

```bash
cargo run --package anomaly-detection
```

## What is an Anomaly?

**Anomaly (outlier)**: Data point significantly different from normal patterns

### Types

**1. Point Anomalies**
```
Single unusual data point
Example: $10,000 charge when usually $50
```

**2. Contextual Anomalies**
```
Unusual in specific context
Example: 35°C normal in summer, anomaly in winter
```

**3. Collective Anomalies**
```
Collection of points unusual together
Example: Sequence of actions indicating intrusion
```

## Why Difficult?

```
• Normal behavior hard to define precisely
• Boundary between normal/anomalous fuzzy
• Anomalies rare (99.9% normal, 0.1% anomaly)
• Constantly evolving patterns
• High cost of errors (false positives/negatives)
```

## Main Approaches

### 1. Statistical

```
Assume normal data follows distribution

Method: Z-score
  z = (x - μ) / σ
  If |z| > threshold (typically 2.5-3): Anomaly

✅ Simple, fast, interpretable
❌ Assumes Gaussian distribution
❌ Doesn't work for complex patterns
```

### 2. Distance-Based (KNN)

```
Points far from neighbors are anomalies

Algorithm:
1. For each point, find k nearest neighbors
2. Compute average distance to neighbors
3. If distance > threshold: Anomaly

✅ No distribution assumption
✅ Intuitive
❌ O(n²) computation
❌ Sensitive to k and distance metric
```

### 3. Density-Based (LOF)

```
Local Outlier Factor

Points in low-density regions are anomalies
Compares local density to neighbors' density

✅ Handles varying densities
✅ Finds local anomalies
❌ Computationally expensive
❌ Parameter tuning needed
```

## Isolation Forest

**Key insight**: Anomalies are easier to isolate than normal points

### Intuition

```
Randomly partition data:
  • Normal point: Surrounded by similar points
    → Need many splits to isolate
  • Anomaly: Few similar points
    → Need few splits to isolate
```

### Algorithm

**Training**:
```
1. Build ensemble of isolation trees (100-200)
2. For each tree:
   a. Sample data randomly
   b. Recursively partition:
      - Pick random feature
      - Pick random split value
      - Split data
      - Repeat until isolated or max depth
```

**Scoring**:
```
1. Pass point through all trees
2. Record path length to isolate
3. Average across trees
4. Normalize:
   score = 2^(-avg_path / c(n))

   score ≈ 1: Anomaly (short paths)
   score ≈ 0: Normal (long paths)
```

### Advantages

✅ Fast: O(n log n) training, O(log n) prediction
✅ Handles high dimensions
✅ No distance computation
✅ Works without knowing distribution
✅ Few parameters

### Disadvantages

❌ Random (use ensemble for stability)
❌ Less effective when anomalies cluster
❌ Binary decision, not severity score

## Autoencoder-Based Detection

**Key insight**: Neural network trained on normal data will fail to reconstruct anomalies (high reconstruction error)

### Architecture

```
Input (n dims)
  ↓
Encoder: Compress (n → k dims)
  ↓
Bottleneck (k << n)
  ↓
Decoder: Reconstruct (k → n dims)
  ↓
Output (≈ Input if normal)
```

**Example**:
```
Input: 784 dims (28×28 image)
  ↓
Dense(128, ReLU)
  ↓
Dense(64, ReLU)
  ↓
Dense(32, ReLU)  ← Bottleneck
  ↓
Dense(64, ReLU)
  ↓
Dense(128, ReLU)
  ↓
Dense(784, Sigmoid)
  ↓
Reconstructed: 784 dims
```

### Training

```
Train on normal data only!

Loss: Reconstruction error
  L = ||x - x̂||²

Network learns:
  • Compress normal patterns
  • Reconstruct from compressed form
```

### Detection

```
1. Pass test point through autoencoder
2. Compute reconstruction error: e = ||x - x̂||²
3. If e > threshold: Anomaly

Why it works:
  • Normal: Low error (seen during training)
  • Anomaly: High error (can't reconstruct well)
```

### Threshold Selection

**1. Percentile**:
```
threshold = 95th percentile of validation errors
Flags top 5% as anomalies
```

**2. Statistical**:
```
threshold = μ + 3σ
Assumes Gaussian error distribution
```

**3. Domain knowledge**:
```
Set based on tolerable false positive rate
```

### Variants

**1. Variational Autoencoder (VAE)**
- Probabilistic latent space
- Better for capturing distributions

**2. Denoising Autoencoder**
- Add noise to input, train to remove
- More robust features

**3. LSTM Autoencoder**
- For time series
- Encode sequence → Decode sequence

### Advantages

✅ Learns complex patterns automatically
✅ Handles high-dimensional data
✅ Unsupervised (no anomaly labels)
✅ Scalable
✅ Non-linear relationships

### Disadvantages

❌ Requires lots of normal data
❌ Threshold tuning needed
❌ Black box (hard to interpret)
❌ Can overfit
❌ Computationally expensive

## One-Class SVM

**Key insight**: Learn boundary around normal data in feature space

### Concept

```
Map data to high-dimensional space (kernel)
Find smallest hypersphere containing normal data
Points outside hypersphere = anomalies
```

### Decision Function

```
f(x) = sign(w^T φ(x) - ρ)

φ(x): Feature map (kernel trick)
f(x) > 0: Normal
f(x) < 0: Anomaly
```

### Kernels

**RBF (Radial Basis Function)**:
```
K(x, y) = exp(-γ ||x - y||²)

Most common
γ controls influence radius
```

**Linear**:
```
K(x, y) = x^T y

Fastest
For linearly separable data
```

**Polynomial**:
```
K(x, y) = (x^T y + c)^d

For non-linear patterns
```

### Parameters

**ν (nu)**: Upper bound on outlier fraction
```
ν = 0.01: Allow 1% outliers
ν = 0.1: Allow 10% outliers
Typical: 0.01 - 0.1
```

**γ (gamma)**: RBF kernel parameter
```
High γ: Tight fit (may overfit)
Low γ: Loose fit (may underfit)
Typical: 1 / n_features
```

### Advantages

✅ Mathematically well-founded
✅ Kernel trick for non-linearity
✅ Works with small datasets
✅ Fewer parameters than deep learning

### Disadvantages

❌ Kernel/parameter selection critical
❌ O(n²) or O(n³) training
❌ Memory intensive
❌ Sensitive to feature scaling

## Evaluation Metrics

### Precision & Recall

```
Precision = TP / (TP + FP)
  Of flagged anomalies, how many are true?
  Important when false alarms costly

Recall = TP / (TP + FN)
  Of true anomalies, how many detected?
  Important when missing anomalies costly
```

### F1-Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Harmonic mean
Balances precision and recall
```

### AUC-ROC

```
Area Under ROC Curve
Threshold-independent
Good for comparing methods
```

### AUC-PR

```
Area Under Precision-Recall Curve
Better for imbalanced data than AUC-ROC
```

## Applications

### 1. Fraud Detection

```
• Credit cards: Unusual transactions
• Insurance: Fraudulent claims
• Online: Fake accounts, bots

Challenges:
  • Highly imbalanced (<0.1% fraud)
  • Adversarial (fraudsters adapt)
  • Real-time detection needed
```

### 2. System Monitoring

```
• Servers: CPU/memory spikes
• Network: DDoS, intrusions
• IoT: Sensor failures

Challenges:
  • High-dimensional (many metrics)
  • Time series patterns
  • Concept drift (normal changes)
```

### 3. Manufacturing/Quality Control

```
• Defective products
• Equipment failures
• Process anomalies

Challenges:
  • Different defect types
  • Small training datasets
  • Real-time inspection
```

### 4. Healthcare

```
• Disease outbreak detection
• Unusual patient vitals
• Medical imaging (tumors)

Challenges:
  • Patient privacy
  • High stakes (life/death)
  • Interpretability required
```

### 5. Cybersecurity

```
• Malware detection
• Intrusion detection
• User behavior analytics

Challenges:
  • Adversarial attacks
  • Zero-day exploits
  • Low latency requirements
```

## Practical Tips

### 1. Feature Engineering

```
Domain knowledge crucial
Try different representations
Normalize/standardize features
```

### 2. Ensemble Methods

```
Combine multiple approaches:
  Isolation Forest + Autoencoder

More robust than single method
```

### 3. Threshold Selection

```
Consider cost of FP vs FN
Use validation set
Adjust based on business needs
```

### 4. Handle Concept Drift

```
Retrain periodically
Online learning
Monitor performance over time
```

### 5. Incorporate Feedback

```
Active learning
Refine based on expert labels
Human-in-the-loop
```

### 6. Explain Detections

```
Which features contributed?
Similar historical anomalies?
Builds trust with users
```

## Method Comparison

| Method | Speed | Scalability | Interpretability | Data Needs |
|--------|-------|-------------|------------------|------------|
| **Statistical** | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| **KNN** | ⚡ | ⭐ | ⭐⭐ | ⭐ |
| **Isolation Forest** | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| **Autoencoder** | ⚡ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| **One-Class SVM** | ⚡⚡ | ⭐⭐ | ⭐⭐ | ⭐ |

## Papers

- [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) (Liu et al., 2008)
- [Autoencoders for Anomaly Detection](https://arxiv.org/abs/1802.03903) (An & Cho, 2015)
- [One-Class SVM](http://users.cecs.anu.edu.au/~williams/papers/P132.pdf) (Schölkopf et al., 2001)
- [LOF: Local Outlier Factor](https://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf) (Breunig et al., 2000)

## Key Takeaways

✓ **Anomaly detection**: Find unusual patterns in data
✓ **Isolation Forest**: Anomalies easier to isolate (fewer splits)
✓ **Autoencoder**: Anomalies have high reconstruction error
✓ **One-Class SVM**: Learn boundary around normal data
✓ **Evaluation**: Precision, Recall, F1, AUC-PR (not just accuracy!)
✓ **Applications**: Fraud, monitoring, quality, security, healthcare
✓ **Challenges**: Imbalanced data, concept drift, threshold tuning
✓ **Practice**: Ensemble methods, domain knowledge, human feedback
✓ **Trade-off**: False positives (alerts) vs false negatives (missed anomalies)

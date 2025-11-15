# Rust Machine Learning & Deep Learning Examples

A comprehensive collection of machine learning and deep learning examples implemented in Rust, designed for learning and experimentation.

## Overview

This repository contains fully documented, runnable examples covering fundamental ML/DL concepts, from basic linear regression to neural networks. Each example includes:

- **Detailed explanations** of concepts and algorithms
- **Runnable code** with clear comments
- **Mathematical foundations** with formulas and intuitions
- **Practical applications** and use cases
- **Comprehensive READMEs** for deeper learning

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/rust-ml-dl.git
cd rust-ml-dl

# Run any example
cargo run --package linear-regression
cargo run --package neural-network
cargo run --package gradient-descent
```

## Examples

### Supervised Learning

#### 1. Linear Regression
**Path:** `examples/linear-regression`
**Run:** `cargo run --package linear-regression`

Learn to predict continuous values using linear relationships.

**Key Concepts:**
- Least squares method
- Mean Squared Error (MSE)
- R² score evaluation
- Train/test splitting

**Use Cases:** House price prediction, sales forecasting, trend analysis

---

#### 2. Logistic Regression
**Path:** `examples/logistic-regression`
**Run:** `cargo run --package logistic-regression`

Binary classification using the sigmoid function.

**Key Concepts:**
- Sigmoid activation
- Binary cross-entropy loss
- Confusion matrix
- Precision, recall, F1-score

**Use Cases:** Spam detection, medical diagnosis, credit risk

---

#### 3. Decision Trees
**Path:** `examples/decision-trees`
**Run:** `cargo run --package decision-trees`

Interpretable classification using tree-based rules.

**Key Concepts:**
- Gini impurity
- Information gain
- Tree pruning
- Overfitting prevention

**Use Cases:** Medical diagnosis, credit approval, rule-based classification

---

#### 4. Support Vector Machines (SVM)
**Path:** `examples/svm`
**Run:** `cargo run --package svm`

Find maximum-margin hyperplanes for classification.

**Key Concepts:**
- Maximum margin
- Support vectors
- Kernel trick
- Soft vs hard margin

**Use Cases:** Text classification, image recognition, bioinformatics

### Unsupervised Learning

#### 5. K-Means Clustering
**Path:** `examples/k-means-clustering`
**Run:** `cargo run --package k-means-clustering`

Group similar data points without labels.

**Key Concepts:**
- Centroid calculation
- Elbow method
- Within-cluster sum of squares (WCSS)
- Cluster assignment

**Use Cases:** Customer segmentation, image compression, pattern discovery

---

#### 6. Principal Component Analysis (PCA)
**Path:** `examples/pca`
**Run:** `cargo run --package pca`

Reduce dimensionality while preserving variance.

**Key Concepts:**
- Eigenvalue decomposition
- Explained variance
- Feature extraction
- Dimensionality reduction

**Use Cases:** Visualization, noise reduction, feature engineering

### Optimization & Fundamentals

#### 7. Gradient Descent
**Path:** `examples/gradient-descent`
**Run:** `cargo run --package gradient-descent`

**Implementation from scratch!** Understand the core optimization algorithm powering ML.

**Key Concepts:**
- Batch gradient descent
- Stochastic gradient descent (SGD)
- Mini-batch gradient descent
- Learning rate effects

**Importance:** Foundation for training neural networks and most ML models

---

#### 8. Data Preprocessing
**Path:** `examples/data-preprocessing`
**Run:** `cargo run --package data-preprocessing`

Essential techniques for preparing data for ML.

**Key Concepts:**
- Feature scaling (standardization, normalization)
- Missing value imputation
- Outlier detection (IQR method)
- Train/test splitting
- Feature engineering

**Importance:** Often the most critical step in ML pipelines!

### Deep Learning

#### 9. Neural Networks
**Path:** `examples/neural-network`
**Run:** `cargo run --package neural-network`

**Implementation from scratch!** Build a feedforward network with backpropagation.

**Key Concepts:**
- Forward propagation
- Backpropagation
- Activation functions (ReLU, Sigmoid)
- Binary cross-entropy loss
- Weight initialization

**Architecture:** Input → Hidden (ReLU) → Output (Sigmoid)

---

#### 10. Deep Learning Basics
**Path:** `examples/deep-learning-basics`
**Run:** `cargo run --package deep-learning-basics`

Comprehensive guide to deep learning concepts and best practices.

**Topics Covered:**
- Activation functions comparison
- Loss functions (BCE, CCE, MSE, MAE)
- Regularization (L1/L2, Dropout)
- Optimization algorithms (SGD, Adam)
- Common architectures (CNN, RNN, Transformer)
- Training best practices

## Project Structure

```
rust-ml-dl/
├── Cargo.toml              # Workspace configuration
├── README.md               # This file
├── examples/
│   ├── linear-regression/
│   │   ├── src/main.rs    # Implementation
│   │   ├── Cargo.toml     # Dependencies
│   │   └── README.md      # Detailed docs
│   ├── logistic-regression/
│   ├── k-means-clustering/
│   ├── decision-trees/
│   ├── gradient-descent/
│   ├── svm/
│   ├── pca/
│   ├── neural-network/
│   ├── data-preprocessing/
│   └── deep-learning-basics/
└── LICENSE
```

## Learning Path

### Beginner Track
1. **Linear Regression** - Start here! Simple and intuitive
2. **Logistic Regression** - Introduction to classification
3. **Data Preprocessing** - Essential data preparation techniques
4. **K-Means Clustering** - Unsupervised learning basics

### Intermediate Track
5. **Gradient Descent** - Understand optimization fundamentals
6. **Decision Trees** - Learn tree-based models
7. **PCA** - Dimensionality reduction
8. **SVM** - Advanced classification

### Advanced Track
9. **Neural Networks** - Deep learning fundamentals
10. **Deep Learning Basics** - Modern DL concepts and architectures

## Libraries Used

- **[ndarray](https://github.com/rust-ndarray/ndarray)** - N-dimensional arrays (like NumPy)
- **[linfa](https://github.com/rust-ml/linfa)** - Comprehensive ML framework
  - `linfa-linear` - Linear models
  - `linfa-logistic` - Logistic regression
  - `linfa-clustering` - K-means and other clustering
  - `linfa-trees` - Decision trees and random forests
  - `linfa-svm` - Support vector machines
  - `linfa-reduction` - PCA and dimensionality reduction
- **[rand](https://github.com/rust-random/rand)** - Random number generation

## Key Features

- **Pedagogical Focus**: Code written for learning, not just performance
- **Comprehensive Documentation**: Every concept explained with math and intuition
- **Runnable Examples**: All code compiles and runs out of the box
- **From Scratch Implementations**: Neural networks and gradient descent implemented without high-level ML libraries
- **Real-World Context**: Use cases and applications for each technique
- **Best Practices**: Training tips, hyperparameter tuning, evaluation metrics

## Running Examples

Each example can be run independently:

```bash
# Run a specific example
cargo run --package <example-name>

# Examples
cargo run --package linear-regression
cargo run --package neural-network
cargo run --package gradient-descent
cargo run --package k-means-clustering
```

Build all examples:

```bash
cargo build --workspace
```

## Prerequisites

- Rust 1.70 or higher
- Basic understanding of:
  - Linear algebra (vectors, matrices)
  - Calculus (derivatives, gradients)
  - Probability and statistics
  - Programming fundamentals

## Learning Resources

### Books
- [Deep Learning Book](https://www.deeplearningbook.org/) by Goodfellow, Bengio, Courville
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) by Bishop
- [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) by Hastie, Tibshirani, Friedman

### Online Courses
- [CS229: Machine Learning](http://cs229.stanford.edu/) - Stanford
- [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/) - Stanford
- [Fast.ai Practical Deep Learning](https://www.fast.ai/)

### Rust-Specific
- [Linfa Documentation](https://rust-ml.github.io/linfa/)
- [ndarray Documentation](https://docs.rs/ndarray/)
- [Are We Learning Yet?](http://www.arewelearningyet.com/) - Rust ML ecosystem

## Contributing

Contributions are welcome! Here are some ways to contribute:

- Add new examples (Random Forests, XGBoost, GANs, etc.)
- Improve documentation and explanations
- Fix bugs or optimize implementations
- Add visualizations or plots
- Suggest new learning resources

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Rust ML community and [linfa](https://github.com/rust-ml/linfa) project
- The [ndarray](https://github.com/rust-ndarray/ndarray) developers
- Machine learning educators and researchers worldwide

---

**Happy Learning!** 🦀 🤖 📊

Start with `cargo run --package linear-regression` and work your way through the examples!

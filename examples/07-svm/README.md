# Support Vector Machine (SVM) Example

This example demonstrates binary classification using Support Vector Machines with the Linfa library.

## Overview

SVMs find the optimal hyperplane that maximizes the margin between different classes, making them powerful classifiers especially in high-dimensional spaces.

## Running the Example

```bash
cargo run --package svm
```

## Key Concepts

- **Hyperplane**: Decision boundary that separates classes
- **Support Vectors**: Data points closest to the hyperplane
- **Margin**: Distance from hyperplane to nearest points
- **Kernel Trick**: Transform to higher dimensions for non-linear separation

## Mathematical Foundation

SVM optimization problem:

```
minimize: (1/2)||w||² + C·Σξᵢ
subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
```

Where C controls the penalty for misclassifications.

## Further Reading

- [SVM Tutorial](https://en.wikipedia.org/wiki/Support-vector_machine)
- [The Kernel Trick](https://en.wikipedia.org/wiki/Kernel_method)

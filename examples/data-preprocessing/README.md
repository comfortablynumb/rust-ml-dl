# Data Preprocessing Example

This example demonstrates essential data preprocessing techniques for machine learning.

## Overview

Data preprocessing transforms raw data into a format suitable for machine learning algorithms. It's often the most important step in the ML pipeline!

## Running the Example

```bash
cargo run --package data-preprocessing
```

## Techniques Covered

### 1. Feature Scaling
- **Standardization**: Mean = 0, Std = 1
- **Min-Max Normalization**: Range [0, 1]
- **Robust Scaling**: Using median and IQR

### 2. Outlier Detection
- IQR method for identifying outliers

### 3. Missing Value Imputation
- Mean imputation
- Median imputation (exercise)

### 4. Train/Test Split
- Proper data splitting strategies

### 5. Feature Engineering
- Polynomial features
- Feature combinations

## Key Principles

1. **Always apply same preprocessing to train and test data**
2. **Fit on training data, transform both**
3. **Feature scaling is crucial for distance-based algorithms**
4. **Choose imputation method based on data distribution**
5. **Preserve domain knowledge when handling outliers**

## Further Reading

- [Feature Scaling](https://en.wikipedia.org/wiki/Feature_scaling)
- [Data Preprocessing Best Practices](https://scikit-learn.org/stable/modules/preprocessing.html)

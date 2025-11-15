# Logistic Regression Example

This example demonstrates binary classification using logistic regression in Rust with the Linfa library.

## Overview

Logistic regression is a supervised learning algorithm for classification. Despite its name, it's used for classification, not regression.

## Mathematical Foundation

The logistic regression model uses the sigmoid function to map any real-valued input to a probability between 0 and 1:

```
σ(z) = 1 / (1 + e^(-z))
```

Where:
```
z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

The model predicts class 1 if σ(z) ≥ 0.5, otherwise class 0.

## Cost Function

Logistic regression uses the **log-loss** (cross-entropy) cost function:

```
J(β) = -(1/n) Σ[yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

This penalizes wrong predictions more heavily when the model is confident.

## Running the Example

```bash
cargo run --package logistic-regression
```

## What the Example Does

1. **Generates synthetic binary classification data** with two well-separated classes
2. **Shuffles and splits** the data into training (80%) and testing (20%) sets
3. **Trains a logistic regression model** using gradient descent
4. **Makes predictions** on the test set
5. **Evaluates performance** using accuracy, precision, recall, F1 score, and confusion matrix

## Key Concepts

### Evaluation Metrics

- **Accuracy**: Percentage of correct predictions (TP + TN) / Total
- **Precision**: Of predicted positives, how many are correct? TP / (TP + FP)
- **Recall (Sensitivity)**: Of actual positives, how many did we find? TP / (TP + FN)
- **F1 Score**: Harmonic mean of precision and recall: 2 * (P * R) / (P + R)

### Confusion Matrix

```
                Predicted
              Neg    Pos
Actual Neg    TN     FP
       Pos    FN     TP
```

- **TP (True Positives)**: Correctly predicted positive
- **TN (True Negatives)**: Correctly predicted negative
- **FP (False Positives)**: Incorrectly predicted positive (Type I error)
- **FN (False Negatives)**: Incorrectly predicted negative (Type II error)

### When to Use Logistic Regression

- Binary classification problems (two classes)
- When you need probability estimates
- When you want an interpretable model
- As a baseline for more complex models
- When classes are linearly separable

### Advantages

1. **Simple and interpretable**: Easy to understand and explain
2. **Probabilistic output**: Provides probability scores
3. **Fast to train**: Efficient even on large datasets
4. **Works well with linearly separable data**
5. **Less prone to overfitting** (with regularization)

### Limitations

1. **Assumes linear decision boundary**: Cannot handle complex non-linear relationships
2. **Sensitive to outliers**
3. **Requires feature scaling** for optimal performance
4. **Limited to binary classification** (without extensions)

## Extensions

- **Multinomial Logistic Regression**: For multi-class classification
- **Regularization** (L1/L2): Prevents overfitting
- **Feature Engineering**: Polynomial features for non-linear boundaries

## Libraries Used

- `linfa`: Machine learning framework for Rust
- `linfa-logistic`: Logistic regression implementation
- `ndarray`: N-dimensional arrays for numerical computing

## Further Reading

- [Logistic Regression Theory](https://en.wikipedia.org/wiki/Logistic_regression)
- [Cross-Entropy Loss](https://en.wikipedia.org/wiki/Cross_entropy)

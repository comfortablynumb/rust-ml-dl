# Linear Regression Example

This example demonstrates how to implement and use linear regression in Rust using the Linfa library.

## Overview

Linear regression is a supervised learning algorithm that models the relationship between features and a continuous target variable using a linear equation.

## Mathematical Foundation

The linear regression model aims to find the best-fitting line (or hyperplane in higher dimensions):

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

Where:
- `y` is the predicted value
- `β₀` is the intercept (bias term)
- `β₁, β₂, ..., βₙ` are the coefficients (weights)
- `x₁, x₂, ..., xₙ` are the input features

The model finds these parameters by minimizing the **Mean Squared Error (MSE)**:

```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```

## Running the Example

```bash
cargo run --package linear-regression
```

## What the Example Does

1. **Generates synthetic data** with a known linear relationship
2. **Splits the data** into training (80%) and testing (20%) sets
3. **Trains a linear regression model** on the training data
4. **Makes predictions** on the test set
5. **Evaluates performance** using MSE, MAE, and R² score

## Key Concepts

### Evaluation Metrics

- **MSE (Mean Squared Error)**: Average of squared differences between predictions and actual values. Lower is better.
- **MAE (Mean Absolute Error)**: Average of absolute differences. More interpretable than MSE.
- **R² Score**: Proportion of variance explained by the model. Ranges from 0 to 1, where 1 is perfect.

### When to Use Linear Regression

- Predicting continuous numerical values
- Understanding feature importance
- When you need an interpretable model
- When the relationship between features and target is approximately linear

### Assumptions

1. **Linearity**: Relationship between features and target is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of errors
4. **Normality**: Errors are normally distributed (for inference)

## Libraries Used

- `linfa`: Machine learning framework for Rust
- `linfa-linear`: Linear models implementation
- `ndarray`: N-dimensional arrays for numerical computing
- `ndarray-rand`: Random array generation

## Further Reading

- [Linfa Documentation](https://rust-ml.github.io/linfa/)
- [Linear Regression Theory](https://en.wikipedia.org/wiki/Linear_regression)

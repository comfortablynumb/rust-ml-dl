# Gradient Descent Example

This example implements gradient descent from scratch to demonstrate the fundamental optimization algorithm used throughout machine learning.

## Overview

Gradient descent is an iterative optimization algorithm that finds the minimum of a function by following the direction of steepest descent.

## Mathematical Foundation

The update rule for gradient descent:

```
θₜ₊₁ = θₜ - α · ∇J(θₜ)
```

Where:
- `θ` = parameters (weights and bias)
- `α` = learning rate (step size)
- `∇J(θ)` = gradient of the cost function
- `t` = iteration number

## The Intuition

Imagine being on a mountainside in fog:
1. Feel the slope under your feet (compute gradient)
2. Take a step downhill (update parameters)
3. Repeat until you reach the valley (minimum)

The gradient points in the direction of steepest **ascent**, so we subtract it to descend.

## Running the Example

```bash
cargo run --package gradient-descent
```

## What the Example Does

1. **Generates synthetic linear data** with known parameters
2. **Implements three variants** of gradient descent from scratch:
   - Batch Gradient Descent
   - Stochastic Gradient Descent (SGD)
   - Mini-batch Gradient Descent
3. **Compares performance** of all three methods
4. **Shows convergence** by tracking loss over iterations

## Gradient Descent Variants

### 1. Batch Gradient Descent

Uses the **entire dataset** to compute gradient:

```rust
for iteration in 0..n_iterations {
    gradient = compute_gradient(all_data)
    parameters = parameters - learning_rate * gradient
}
```

**Pros:**
- Stable, smooth convergence
- Guaranteed to converge to global minimum (convex functions)
- Deterministic

**Cons:**
- Slow for large datasets
- Requires all data in memory
- Can get stuck in local minima (non-convex functions)

### 2. Stochastic Gradient Descent (SGD)

Uses **one sample at a time**:

```rust
for epoch in 0..n_epochs {
    shuffle(data)
    for sample in data {
        gradient = compute_gradient(sample)
        parameters = parameters - learning_rate * gradient
    }
}
```

**Pros:**
- Fast updates (one sample at a time)
- Can escape shallow local minima
- Works with streaming/online data
- Memory efficient

**Cons:**
- Noisy gradient estimates
- Fluctuating loss (doesn't smoothly decrease)
- May oscillate around minimum
- Requires learning rate decay

### 3. Mini-batch Gradient Descent

Uses **small batches** of samples:

```rust
for epoch in 0..n_epochs {
    shuffle(data)
    for batch in data.batches(batch_size) {
        gradient = compute_gradient(batch)
        parameters = parameters - learning_rate * gradient
    }
}
```

**Pros:**
- Balance between batch and SGD
- More stable than SGD
- GPU-friendly (parallel computation)
- Good gradient estimates with less noise

**Cons:**
- Hyperparameter tuning (batch size)
- Not as smooth as batch GD

**Common batch sizes:** 32, 64, 128, 256

## The Learning Rate

The learning rate `α` is the most important hyperparameter:

### Too Small (α = 0.0001)
```
Loss: 100 → 99.9 → 99.8 → 99.7 → ...
Problem: Slow convergence, may never reach minimum
```

### Just Right (α = 0.01)
```
Loss: 100 → 80 → 60 → 40 → 20 → 10 → 5 → 2 → 1
Result: Efficient convergence
```

### Too Large (α = 1.0)
```
Loss: 100 → 200 → 400 → 800 → ...
Problem: Overshooting, divergence
```

## Advanced Techniques

### Learning Rate Schedules

Decrease learning rate over time:

1. **Step Decay**: Reduce by factor every N epochs
   ```
   α = α₀ * 0.1^(epoch / 100)
   ```

2. **Exponential Decay**:
   ```
   α = α₀ * e^(-kt)
   ```

3. **1/t Decay**:
   ```
   α = α₀ / (1 + kt)
   ```

### Adaptive Learning Rates

Modern optimizers automatically adjust learning rates:

- **Momentum**: Accumulates gradient direction
  ```
  v = β * v + ∇J(θ)
  θ = θ - α * v
  ```

- **AdaGrad**: Adapts rate per parameter based on history
  ```
  g = g + (∇J(θ))²
  θ = θ - (α / √(g + ε)) * ∇J(θ)
  ```

- **RMSprop**: Uses moving average of squared gradients
  ```
  g = β * g + (1-β) * (∇J(θ))²
  θ = θ - (α / √(g + ε)) * ∇J(θ)
  ```

- **Adam**: Combines momentum and RMSprop (most popular)
  ```
  m = β₁ * m + (1-β₁) * ∇J(θ)
  v = β₂ * v + (1-β₂) * (∇J(θ))²
  θ = θ - α * m / √(v + ε)
  ```

## Convergence Criteria

When to stop training:

1. **Fixed iterations/epochs**: Simple but arbitrary
2. **Loss threshold**: Stop when loss < ε
3. **Gradient magnitude**: Stop when ||∇J|| < ε
4. **Relative improvement**: Stop when |lossₜ - lossₜ₋₁| < ε
5. **Validation loss**: Stop when validation loss stops improving (early stopping)

## Common Issues

### Problem: Slow Convergence
- **Solution**: Increase learning rate, use momentum, or adaptive optimizer

### Problem: Divergence (loss increasing)
- **Solution**: Decrease learning rate, check for bugs, normalize features

### Problem: Oscillation
- **Solution**: Decrease learning rate, use momentum, or learning rate decay

### Problem: Stuck in Local Minimum
- **Solution**: Use SGD's noise, add momentum, try different initialization

## Applications

Gradient descent is used to optimize:
- Linear regression (minimize MSE)
- Logistic regression (minimize cross-entropy)
- Neural networks (backpropagation)
- Support vector machines
- Matrix factorization
- Almost any differentiable model

## Key Takeaways

1. **Gradient descent is fundamental**: Powers most ML training algorithms
2. **Mini-batch is standard**: Best balance of speed and stability
3. **Learning rate matters**: Most important hyperparameter to tune
4. **Modern optimizers help**: Adam is a safe default choice
5. **Monitor convergence**: Always plot loss curves

## Further Reading

- [Gradient Descent Optimization Algorithms](http://ruder.io/optimizing-gradient-descent/)
- [An Overview of Gradient Descent Optimization Algorithms](https://arxiv.org/abs/1609.04747)
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

# Neural Network Example

This example implements a feedforward neural network from scratch to demonstrate core deep learning concepts.

## Overview

A neural network is composed of layers of neurons that learn to transform input data into predictions through training.

## Running the Example

```bash
cargo run --package neural-network
```

## Architecture

```
Input (2) → Hidden (4, ReLU) → Output (1, Sigmoid)
```

## Key Concepts

- **Forward Propagation**: Data flows through layers
- **Activation Functions**: Non-linear transformations (ReLU, Sigmoid)
- **Backpropagation**: Compute gradients using chain rule
- **Gradient Descent**: Update weights to minimize loss

## Universal Approximation

With enough neurons, a neural network can approximate any continuous function!

## Further Reading

- [Neural Networks](https://en.wikipedia.org/wiki/Artificial_neural_network)
- [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
- [Deep Learning Book](https://www.deeplearningbook.org/)

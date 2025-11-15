# Convolutional Neural Network (CNN) Example

This example demonstrates the fundamental concepts of CNNs, the architecture that revolutionized computer vision.

## Overview

CNNs use convolution operations to automatically learn spatial hierarchies of features from images, making them the go-to architecture for computer vision.

## Running the Example

```bash
cargo run --package cnn
```

## Key Operations

### 1. Convolution
Slides a small filter (kernel) over the image to detect features:
- Edges (vertical, horizontal, diagonal)
- Textures
- Patterns
- Complex features in deeper layers

### 2. Pooling
Reduces spatial dimensions while retaining important information:
- **Max Pooling**: Takes maximum value in each region
- **Average Pooling**: Takes average value
- Benefits: Reduces parameters, translation invariance

### 3. Multiple Feature Maps
CNNs use many filters in each layer:
- Each filter learns a different feature
- Early layers: Simple edges and colors
- Deep layers: Complex objects and patterns

## Architecture Pattern

```
Input Image
    ↓
[Conv + ReLU + Pool] × N
    ↓
Flatten
    ↓
Dense Layers
    ↓
Softmax Output
```

## Why CNNs for Images?

1. **Local Connectivity**: Neurons connect only to nearby pixels
2. **Parameter Sharing**: Same filter applies everywhere
3. **Translation Invariance**: Features work regardless of position
4. **Hierarchical Learning**: Builds complex features from simple ones

## Famous Architectures

- **LeNet-5 (1998)**: Pioneering CNN for digit recognition
- **AlexNet (2012)**: Sparked the deep learning revolution
- **VGG (2014)**: Very deep with simple 3×3 filters
- **ResNet (2015)**: Skip connections enable 100+ layers
- **Inception**: Multi-scale feature extraction
- **EfficientNet**: Optimal scaling

## Applications

- Image classification
- Object detection
- Semantic segmentation
- Face recognition
- Medical image analysis
- Self-driving cars

## Further Reading

- [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)
- [Understanding CNNs](https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)

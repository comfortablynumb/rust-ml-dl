# Rust Machine Learning & Deep Learning Examples

A comprehensive collection of **16 fully documented machine learning and deep learning examples** implemented in Rust, organized in a clear learning progression.

## Overview

This repository contains fully documented, runnable examples covering fundamental ML/DL concepts, from data preprocessing to advanced deep learning architectures. Each example includes:

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

# Run any example (use the package name without the number prefix)
cargo run --package data-preprocessing
cargo run --package neural-network
cargo run --package cnn
```

## 📚 Examples (In Learning Order)

### 01. Data Preprocessing
**Path:** `examples/01-data-preprocessing`
**Run:** `cargo run --package data-preprocessing`

Essential techniques for preparing data for machine learning.

**Topics:**
- Feature scaling (standardization, normalization)
- Missing value imputation
- Outlier detection (IQR method)
- Train/test splitting
- Feature engineering

**Why Start Here:** Data preprocessing is often the most important step in ML pipelines!

---

## Supervised Learning

### 02. Linear Regression
**Path:** `examples/02-linear-regression`
**Run:** `cargo run --package linear-regression`

Predict continuous values using linear relationships.

**Key Concepts:**
- Least squares method
- Mean Squared Error (MSE)
- R² score evaluation
- Model interpretation

**Use Cases:** House price prediction, sales forecasting, trend analysis

---

### 03. Logistic Regression
**Path:** `examples/03-logistic-regression`
**Run:** `cargo run --package logistic-regression`

Binary classification using the sigmoid function.

**Key Concepts:**
- Sigmoid activation
- Binary cross-entropy loss
- Confusion matrix
- Precision, recall, F1-score

**Use Cases:** Spam detection, medical diagnosis, credit risk assessment

---

## Unsupervised Learning

### 04. K-Means Clustering
**Path:** `examples/04-k-means-clustering`
**Run:** `cargo run --package k-means-clustering`

Group similar data points without labels.

**Key Concepts:**
- Centroid calculation
- Elbow method for choosing K
- Within-cluster sum of squares (WCSS)
- Cluster assignment

**Use Cases:** Customer segmentation, image compression, pattern discovery

---

### 05. Principal Component Analysis (PCA)
**Path:** `examples/05-pca`
**Run:** `cargo run --package pca`

Reduce dimensionality while preserving variance.

**Key Concepts:**
- Eigenvalue decomposition
- Explained variance
- Feature extraction
- Dimensionality reduction

**Use Cases:** Visualization, noise reduction, feature engineering

---

## Advanced Supervised Learning

### 06. Decision Trees
**Path:** `examples/06-decision-trees`
**Run:** `cargo run --package decision-trees`

Interpretable classification using tree-based rules.

**Key Concepts:**
- Gini impurity
- Information gain
- Tree pruning strategies
- Overfitting prevention

**Use Cases:** Medical diagnosis, credit approval, rule-based systems

---

### 07. Support Vector Machines (SVM)
**Path:** `examples/07-svm`
**Run:** `cargo run --package svm`

Find maximum-margin hyperplanes for classification.

**Key Concepts:**
- Maximum margin principle
- Support vectors
- Kernel trick (non-linear classification)
- Soft vs hard margin

**Use Cases:** Text classification, image recognition, bioinformatics

---

## Optimization Fundamentals

### 08. Gradient Descent ⭐
**Path:** `examples/08-gradient-descent`
**Run:** `cargo run --package gradient-descent`

**Implementation from scratch!** The core optimization algorithm powering ML.

**Key Concepts:**
- Batch gradient descent
- Stochastic gradient descent (SGD)
- Mini-batch gradient descent
- Learning rate effects

**Why Important:** Foundation for training neural networks and most ML models

---

## Deep Learning

### 09. Neural Networks ⭐
**Path:** `examples/09-neural-network`
**Run:** `cargo run --package neural-network`

**Implementation from scratch!** Feedforward network with backpropagation.

**Key Concepts:**
- Forward propagation
- Backpropagation algorithm
- Activation functions (ReLU, Sigmoid)
- Binary cross-entropy loss
- Weight initialization

**Architecture:** Input → Hidden (ReLU) → Output (Sigmoid)

---

### 10. Deep Learning Basics
**Path:** `examples/10-deep-learning-basics`
**Run:** `cargo run --package deep-learning-basics`

Comprehensive guide to deep learning concepts and best practices.

**Topics Covered:**
- Activation functions (Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax)
- Loss functions (BCE, CCE, MSE, MAE)
- Regularization techniques (L1/L2, Dropout, Early Stopping)
- Optimization algorithms (SGD, Momentum, Adam)
- Common architectures (CNN, RNN, Transformer, GAN, Autoencoder)
- Training best practices
- Hyperparameter tuning

---

## Deep Learning Architectures

### 11. Convolutional Neural Networks (CNN) 🆕
**Path:** `examples/11-cnn`
**Run:** `cargo run --package cnn`

The architecture that revolutionized computer vision.

**Key Operations:**
- Convolution (feature detection with filters)
- Pooling (dimensionality reduction)
- Multiple feature maps
- Hierarchical feature learning

**Key Concepts:**
- Local connectivity
- Parameter sharing
- Translation invariance
- Edge detection, texture recognition

**Applications:** Image classification, object detection, face recognition, medical imaging

**Famous Architectures:** LeNet, AlexNet, VGG, ResNet, EfficientNet

---

### 12. Recurrent Neural Networks (RNN) 🆕
**Path:** `examples/12-rnn`
**Run:** `cargo run --package rnn`

Designed for sequential data like text, time series, and speech.

**Key Concepts:**
- Hidden state (memory across time steps)
- Sequence processing patterns
- Vanishing/exploding gradients
- LSTM and GRU variants

**Architectures:**
- Many-to-One (sentiment analysis)
- One-to-Many (image captioning)
- Many-to-Many (translation, video labeling)
- Seq2Seq (encoder-decoder)

**Applications:** Machine translation, text generation, speech recognition, time series forecasting

**Modern Note:** Transformers have largely replaced RNNs for NLP, but RNNs remain useful for streaming/online processing

---

### 13. Autoencoders 🆕
**Path:** `examples/13-autoencoder`
**Run:** `cargo run --package autoencoder`

Unsupervised learning for compression, denoising, and generation.

**Architecture:**
- Encoder: Compresses input to latent representation
- Bottleneck: Low-dimensional latent space
- Decoder: Reconstructs from latent code

**Variants:**
- Vanilla Autoencoder (basic compression)
- Denoising Autoencoder (noise removal)
- Variational Autoencoder (VAE - generation)
- Sparse Autoencoder (interpretable features)

**Applications:** Dimensionality reduction, anomaly detection, image denoising, data generation, compression

**Modern Impact:** VAEs power Stable Diffusion and DALL-E!

---

### 14. Generative Adversarial Networks (GANs) 🔥
**Path:** `examples/14-gan`
**Run:** `cargo run --package gan`

Revolutionary architecture for generating realistic data through adversarial training.

**Key Concepts:**
- Generator vs Discriminator competition
- Adversarial training process
- Non-saturating loss functions
- Mode collapse and training instability

**Architecture:**
- Generator: Noise → Fake Data
- Discriminator: Real/Fake → Probability
- Min-max game: G tries to fool D, D tries to detect fakes

**Variants:**
- DCGAN (Convolutional for images)
- CGAN (Conditional generation)
- CycleGAN (Domain translation)
- StyleGAN (High-quality faces)
- WGAN (Wasserstein distance)

**Applications:** Image generation, style transfer, data augmentation, drug discovery, deepfakes

**Modern Context:** While diffusion models (Stable Diffusion, DALL-E) have overtaken GANs for image generation, GANs remain important for real-time applications and video games

**Famous:** ThisPersonDoesNotExist.com, StyleGAN face generation

---

### 15. Transformer Architecture 🚀
**Path:** `examples/15-transformer`
**Run:** `cargo run --package transformer`

The architecture that revolutionized AI - powers GPT, BERT, ChatGPT, and most modern AI systems.

**Core Innovation:**
- Self-attention mechanism (learns which words relate)
- Multi-head attention (learns different relationships)
- Positional encoding (handles word order)
- Parallel processing (unlike sequential RNNs)

**Architecture:**
- Encoder: Bidirectional processing
- Decoder: Autoregressive generation
- Attention(Q, K, V) = softmax(QK^T / √d_k) · V

**Famous Variants:**
- **BERT** (Encoder-only): Text understanding, Q&A
- **GPT** (Decoder-only): Text generation, ChatGPT
- **T5** (Full encoder-decoder): Translation, summarization
- **Vision Transformer** (ViT): Image classification
- **CLIP**: Vision-language connections

**Why Transformers Won:**
- Parallelization: 100× faster than RNNs
- Long-range dependencies: Direct connections
- Scalability: More params + data = better performance

**Applications:** NLP (translation, generation), code (GitHub Copilot), vision (ViT, DALL-E), speech (Whisper), science (AlphaFold 2)

**Impact:** From GPT-1 (117M params, 2018) → GPT-4 (~1.7T params, 2023)

---

### 16. Residual Networks (ResNet) 🏆
**Path:** `examples/16-resnet`
**Run:** `cargo run --package resnet`

The architecture that solved the degradation problem and enabled extremely deep networks (100+ layers).

**Problem Solved:**
- Before ResNet: Deeper networks performed **worse** (degradation problem)
- After ResNet: 152-layer networks outperform shallow ones

**Core Innovation: Skip Connections**
```
Output = F(x) + x
```
- F(x): Learned residual (what to add)
- x: Input passed through unchanged (identity shortcut)

**Benefits:**
- Easier gradient flow through skip connections
- Easy to learn identity (just set F(x) = 0)
- Each layer refines representation

**Variants:**
- ResNet-18/34: Basic blocks, 11-21M params
- ResNet-50/101/152: Bottleneck blocks, 25-60M params
- ResNeXt: Adds cardinality dimension
- Wide ResNet: Wider layers, better speed/accuracy

**Historical Impact:**
- Won ImageNet 2015: 3.57% error (superhuman!)
- First model to beat human performance (5.1%)
- Most cited computer vision paper (100,000+ citations)

**Applications:** Image classification, object detection (Faster R-CNN), semantic segmentation, face recognition, medical imaging

**Modern Status:** Still widely used as backbone despite Vision Transformers; excellent for transfer learning

---

## Project Structure

```
rust-ml-dl/
├── Cargo.toml                    # Workspace configuration
├── README.md                     # This file
└── examples/
    ├── 01-data-preprocessing/    # Data preparation
    ├── 02-linear-regression/     # Supervised: Regression
    ├── 03-logistic-regression/   # Supervised: Classification
    ├── 04-k-means-clustering/    # Unsupervised: Clustering
    ├── 05-pca/                   # Unsupervised: Dimensionality reduction
    ├── 06-decision-trees/        # Supervised: Tree-based
    ├── 07-svm/                   # Supervised: Margin-based
    ├── 08-gradient-descent/      # Optimization ⭐
    ├── 09-neural-network/        # Deep Learning basics ⭐
    ├── 10-deep-learning-basics/  # DL concepts & theory
    ├── 11-cnn/                   # DL Architecture: Images
    ├── 12-rnn/                   # DL Architecture: Sequences
    ├── 13-autoencoder/           # DL Architecture: Unsupervised
    ├── 14-gan/                   # DL Architecture: Generative 🔥
    ├── 15-transformer/           # DL Architecture: Attention 🚀
    └── 16-resnet/                # DL Architecture: Very Deep 🏆
```

⭐ = Implemented from scratch
🔥🚀🏆 = Advanced deep learning architectures

## Learning Paths

### 🟢 Beginner Track (Start Here!)
1. **01-data-preprocessing** - Essential data preparation
2. **02-linear-regression** - Simplest ML algorithm
3. **03-logistic-regression** - Introduction to classification
4. **04-k-means-clustering** - Unsupervised learning basics

### 🟡 Intermediate Track
5. **08-gradient-descent** - Understand optimization
6. **05-pca** - Dimensionality reduction
7. **06-decision-trees** - Tree-based models
8. **07-svm** - Advanced classification

### 🔴 Advanced Track
9. **09-neural-network** - Deep learning fundamentals
10. **10-deep-learning-basics** - Modern DL concepts
11. **11-cnn** - Computer vision architecture
12. **12-rnn** - Sequential data architecture
13. **13-autoencoder** - Unsupervised deep learning
14. **14-gan** - Generative models & adversarial training
15. **15-transformer** - Attention mechanisms & modern NLP
16. **16-resnet** - Very deep networks & skip connections

## Libraries Used

- **[ndarray](https://github.com/rust-ndarray/ndarray)** - N-dimensional arrays (like NumPy)
- **[linfa](https://github.com/rust-ml/linfa)** - Comprehensive ML framework for Rust
  - `linfa-linear` - Linear models
  - `linfa-logistic` - Logistic regression
  - `linfa-clustering` - K-means and clustering algorithms
  - `linfa-trees` - Decision trees and random forests
  - `linfa-svm` - Support vector machines
  - `linfa-reduction` - PCA and dimensionality reduction
- **[rand](https://github.com/rust-random/rand)** - Random number generation

## Key Features

- **📖 Pedagogical Focus**: Code written for learning, not just performance
- **📚 Comprehensive Documentation**: Every concept explained with math and intuition
- **▶️ Runnable Examples**: All code compiles and runs out of the box
- **🔧 From-Scratch Implementations**: Core algorithms (gradient descent, neural networks) built without high-level ML libraries
- **🌍 Real-World Context**: Use cases and applications for each technique
- **✅ Best Practices**: Training tips, hyperparameter tuning, evaluation metrics
- **🎯 Clear Progression**: Organized from fundamentals to advanced topics

## Running Examples

Each example can be run independently using the **package name** (without number prefix):

```bash
# Fundamentals
cargo run --package data-preprocessing
cargo run --package gradient-descent

# Traditional ML
cargo run --package linear-regression
cargo run --package k-means-clustering
cargo run --package decision-trees

# Deep Learning
cargo run --package neural-network
cargo run --package deep-learning-basics
cargo run --package cnn
cargo run --package rnn
cargo run --package autoencoder
cargo run --package gan
cargo run --package transformer
cargo run --package resnet
```

Build all examples:

```bash
cargo build --workspace
```

## What's New in This Version

### 🎯 Organized Structure
- Examples now numbered 01-16 in logical learning order
- Clear progression from basics to state-of-the-art architectures

### 🆕 Six Deep Learning Architectures
- **CNN**: Convolutional networks for computer vision
- **RNN**: Recurrent networks for sequences (text, time series)
- **Autoencoder**: Unsupervised learning for compression and generation
- **GAN**: Generative adversarial networks for data generation
- **Transformer**: Attention mechanisms powering GPT, BERT, ChatGPT
- **ResNet**: Residual networks enabling very deep architectures

### 📖 Enhanced Documentation
- Each example includes comprehensive theory
- Mathematical foundations with clear explanations
- Real-world applications and use cases
- Historical context and modern impact
- Comparisons between techniques

## Prerequisites

- **Rust 1.70+**
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
- [CS229: Machine Learning](http://cs229.stanford.edu/) - Stanford (Traditional ML)
- [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/) - Stanford (Deep Learning for Vision)
- [CS224n: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/) - Stanford (RNNs, Transformers)
- [Fast.ai Practical Deep Learning](https://www.fast.ai/)

### Rust-Specific
- [Linfa Documentation](https://rust-ml.github.io/linfa/)
- [ndarray Documentation](https://docs.rs/ndarray/)
- [Are We Learning Yet?](http://www.arewelearningyet.com/) - Rust ML ecosystem overview

## Contributing

Contributions are welcome! Ways to contribute:

- Add new examples (Random Forests, XGBoost, Transformers, GANs, etc.)
- Improve documentation and explanations
- Fix bugs or optimize implementations
- Add visualizations or plots
- Suggest new learning resources
- Improve example clarity

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Rust ML community and [linfa](https://github.com/rust-ml/linfa) project
- The [ndarray](https://github.com/rust-ndarray/ndarray) developers
- Machine learning educators and researchers worldwide
- The deep learning community for advancing the field

---

**Happy Learning!** 🦀 🤖 📊 🧠

**Suggested Learning Path:**
1. Start with `01-data-preprocessing`
2. Work through examples 02-07 (traditional ML)
3. Master `08-gradient-descent` (optimization fundamentals)
4. Progress to `09-neural-network` (DL basics)
5. Study `10-deep-learning-basics` (theory)
6. Explore core architectures: `11-cnn`, `12-rnn`, `13-autoencoder`
7. Master advanced architectures: `14-gan`, `15-transformer`, `16-resnet`

Each example builds on previous concepts, so following the numbered order is recommended!

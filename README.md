# Rust Machine Learning & Deep Learning Examples

A comprehensive collection of **36 fully documented machine learning and deep learning examples** implemented in Rust, covering the complete spectrum from fundamentals to cutting-edge AI architectures.

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

### 17. LSTM & GRU (Gated Recurrent Units) 🔗
**Path:** `examples/17-lstm-gru`
**Run:** `cargo run --package lstm-gru`

The RNN variants that solved the vanishing gradient problem and enabled long-term memory.

**Key Innovation:**
- LSTM: Explicit memory cell with three gates (forget, input, output)
- GRU: Simplified variant with two gates (update, reset)
- Gradient highway through cell state

**How They Work:**
- **Gates** control information flow (what to remember/forget)
- **Cell state** (LSTM) provides direct path for gradients
- **Reparameterization** enables learning 100+ timesteps

**LSTM vs GRU:**
- LSTM: More parameters, slightly better on complex tasks
- GRU: 25% faster, similar performance, fewer parameters

**Applications:** Machine translation, speech recognition, time series forecasting, music generation

**Modern Context:** Transformers replaced LSTM/GRU for NLP, but they're still used for streaming data, online processing, resource-constrained settings, and time series

---

### 18. U-Net (Semantic Segmentation) 🎨
**Path:** `examples/18-unet`
**Run:** `cargo run --package unet`

The encoder-decoder architecture that revolutionized medical image segmentation with very few training images.

**Key Innovation:**
- Symmetric encoder-decoder with skip connections
- Works with < 30 training images (heavy data augmentation)
- Pixel-wise predictions for segmentation

**Architecture:**
- **Encoder (contracting path)**: Downsample, extract features
- **Bottleneck**: Most abstract features
- **Decoder (expanding path)**: Upsample, recover spatial detail
- **Skip connections**: Concatenate encoder features to decoder

**Why Skip Connections?:**
- Encoder provides precise spatial information (where)
- Bottleneck provides semantic understanding (what)
- Decoder gets both for accurate segmentation

**Applications:** Medical imaging (tumor detection, organ segmentation), autonomous driving (road segmentation), satellite imagery (land use), portrait mode (background separation)

**Impact:** 40,000+ citations, foundation for Segment Anything (SAM 2023)

---

### 19. VAE (Variational Autoencoder) 📊
**Path:** `examples/19-vae`
**Run:** `cargo run --package vae`

A probabilistic generative model that learns smooth latent representations for generation and anomaly detection.

**Key Innovation:**
- Probabilistic encoding: `x → (μ, σ)` instead of deterministic `x → z`
- Reparameterization trick: `z = μ + σ⊙ε` enables backpropagation
- Smooth, continuous latent space enables generation

**Loss Function:**
```
L = Reconstruction Loss + KL Divergence
```
- **Reconstruction**: How well can we rebuild the input?
- **KL Divergence**: Regularize latent space to N(0,1)

**Generation Methods:**
- Random sampling: `z ~ N(0,1)` → decode → new image
- Interpolation: Smooth morphing between images
- Latent arithmetic: "smile vector" + face = smiling face

**Applications:** Image generation, anomaly detection, data compression, drug discovery (molecular VAE)

**Modern Role:** Component in Stable Diffusion (VAE encoder/decoder), still used for anomaly detection and when fast generation matters

---

### 20. Diffusion Models (DDPM) 🌟
**Path:** `examples/20-diffusion`
**Run:** `cargo run --package diffusion`

**State-of-the-art generative models** powering Stable Diffusion, DALL-E 2, Midjourney, and Imagen.

**Core Idea:**
- **Forward**: Gradually add noise to images (1000 steps)
- **Reverse**: Learn to denoise step-by-step
- **Generation**: Start with pure noise → denoise → clean image

**Training:**
```
1. Sample image x_0 and noise ε
2. Create noisy image x_t
3. Predict noise: ε_pred = network(x_t, t)
4. Loss: ||ε - ε_pred||²
```

**Latent Diffusion (Stable Diffusion):**
- VAE compresses image 512×512 → 64×64 latent
- Diffusion works in latent space (64× faster!)
- VAE decoder: latent → final image

**Text-to-Image:**
- CLIP text encoder: "A cat..." → text embedding
- Cross-attention: Image features attend to text
- Classifier-free guidance: Control text strength

**Why Diffusion Won:**
- Better quality than GANs
- Stable training (no mode collapse)
- Higher diversity
- Controllable generation

**Applications:** Text-to-image (DALL-E, Midjourney), image editing (inpainting), super-resolution, video generation (Sora)

**Impact:** Replaced GANs as #1 generative model, enabled AI art revolution, billion-dollar industry

---

### 21. Graph Neural Networks (GNN) 🕸️
**Path:** `examples/21-gnn`
**Run:** `cargo run --package gnn`

Neural networks for **non-Euclidean data**: graphs with irregular structure like social networks, molecules, and knowledge graphs.

**Core Concept: Message Passing**
- Nodes communicate with neighbors
- Aggregate neighbor information
- Update node representations
- After L layers: Know L-hop neighborhood

**Popular Architectures:**
- **GCN**: Graph convolutions with normalized adjacency
- **GraphSAGE**: Sampling for scalability (millions of nodes)
- **GAT**: Attention mechanism (learn neighbor importance)
- **GIN**: Theoretically most expressive

**Graph Tasks:**
- **Node classification**: Predict node labels (user interests)
- **Link prediction**: Predict missing edges (friend recommendations)
- **Graph classification**: Classify entire graphs (molecule toxicity)

**Applications:**
- **Social networks**: Pinterest (PinSage), friend recommendations
- **Drug discovery**: Molecular property prediction, toxicity
- **Knowledge graphs**: Google Knowledge Graph, fact completion
- **Recommendations**: YouTube, Amazon (user-item graphs)
- **AlphaFold 2**: Protein structure prediction (Nobel Prize-worthy)
- **Traffic**: Road network prediction (Uber, Google Maps)

**Challenges:**
- Over-smoothing (use 2-3 layers, not deep)
- Scalability (sampling, clustering for large graphs)

**Modern Developments:** Graph Transformers, foundation models for graphs

---

### 22. Attention Mechanisms 🔍
**Path:** `examples/22-attention`
**Run:** `cargo run --package attention`

The foundational mechanism powering Transformers, BERT, GPT, and all modern NLP.

**Core Innovation: Query, Key, Value Framework**
- Dynamically focus on relevant parts of input
- No fixed context window like RNNs
- Parallelizable (no sequential dependency)

**Key Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Variants:**
- **Self-Attention**: Sequence attends to itself
- **Cross-Attention**: Attend to different sequence (encoder-decoder)
- **Multi-Head Attention**: Multiple parallel attention patterns
- **Masked Attention**: Prevent looking ahead (GPT-style)

**Why It Won:**
- Replaced RNNs as primary sequence model
- Enables parallelization (100× faster training)
- Better long-range dependencies
- Interpretable (visualize attention weights)

**Applications:** Machine translation, image captioning, question answering, document classification

**Historical Impact:** Foundation of Transformer (2017), which led to BERT, GPT, and the modern AI revolution

---

### 23. Vision Transformer (ViT) 👁️
**Path:** `examples/23-vision-transformer`
**Run:** `cargo run --package vision-transformer`

Pure Transformer architecture for images - no convolutions needed!

**Core Idea: Images as Sequences**
- Split image into patches (16×16)
- Linear embedding of patches
- Add positional encodings
- Standard Transformer encoder
- Classification from [CLS] token

**Architecture:**
```
Image 224×224 → 196 patches (16×16 each) → Transformer → Classification
```

**Why It Works:**
- Global receptive field from layer 1
- Minimal inductive bias (learns from data)
- Scales better than CNNs with massive data

**Data Requirements:**
- ImageNet (1.3M): CNN > ViT
- JFT-300M (300M): ViT > CNN

**Variants:**
- **DeiT**: Data-efficient (distillation from CNNs)
- **Swin Transformer**: Hierarchical, shifted windows
- **BEiT**: Masked image modeling (like BERT for images)

**Applications:** Image classification, object detection (DETR), segmentation, CLIP (vision-language), DALL-E, Stable Diffusion components

**Impact:** Proved Transformers work beyond NLP, unified architecture across modalities

---

### 24. Object Detection (YOLO) 📦
**Path:** `examples/24-object-detection`
**Run:** `cargo run --package object-detection`

Real-time object detection: "You Only Look Once"

**Innovation: Single-Shot Detection**
- Traditional: 2000+ region proposals → slow
- YOLO: Single forward pass → 45+ FPS

**How It Works:**
- Divide image into S×S grid (7×7)
- Each cell predicts B bounding boxes + class probabilities
- Output: 7×7×30 tensor (all predictions at once)

**Architecture:**
- CNN backbone (24 conv layers)
- Fully connected layers
- Direct prediction of bbox coordinates + classes

**YOLO Evolution:**
- YOLOv1 (2015): 45 FPS, original
- YOLOv3 (2018): Multi-scale, better accuracy
- YOLOv5 (2020): PyTorch, easy deployment
- YOLOv8 (2023): Anchor-free, current SOTA

**Applications:** Autonomous driving (Tesla), surveillance, robotics, sports analytics, retail inventory

**Impact:** Enabled real-time object detection, critical for autonomous vehicles and robotics

---

### 25. Siamese Networks 👯
**Path:** `examples/25-siamese-networks`
**Run:** `cargo run --package siamese-networks`

Learn similarity between inputs using twin networks with shared weights.

**Core Concept:**
```
Input 1 → Network \
                   → Compare embeddings → Similar/Different
Input 2 → Network /
  (shared weights)
```

**Key Innovation: Metric Learning**
- Learn embedding space where similar inputs are close
- Dissimilar inputs are far apart
- Generalizes to new classes (one-shot learning)

**Loss Functions:**
- **Contrastive Loss**: Pull similar together, push dissimilar apart
- **Triplet Loss**: Anchor closer to positive than negative

**Applications:**
- **Face verification**: FaceNet (99.63% accuracy), phone unlock
- **One-shot learning**: Character recognition with few examples
- **Image retrieval**: Google Images, Pinterest
- **Signature verification**: Banking, legal

**Modern Context:**
- **Contrastive Learning**: SimCLR, MoCo (self-supervised)
- **CLIP**: Dual encoder (image + text), powers Stable Diffusion/DALL-E
- **Face recognition**: Security systems, photo organization

**Impact:** Pioneered metric learning, foundation for modern contrastive learning and multi-modal AI

---

### 26. Reinforcement Learning (DQN) 🎮
**Path:** `examples/26-reinforcement-learning`
**Run:** `cargo run --package reinforcement-learning`

Learn through interaction: Agent learns from trial and error using Deep Q-Networks.

**Different Paradigm:**
- Supervised: (input, label) pairs
- RL: Agent + Environment + Rewards → Learn policy

**The RL Loop:**
```
Observe state → Take action → Receive reward → Update policy → Repeat
Goal: Maximize cumulative reward
```

**DQN Innovations:**
1. **Experience Replay**: Store transitions, sample randomly
2. **Target Network**: Stabilize training (frozen copy)
3. **Function Approximation**: Neural network for Q(state, action)

**Architecture:**
- Input: 84×84×4 game frames
- 3 conv layers + 2 fully connected
- Output: Q-value for each action

**Exploration vs Exploitation:**
- Epsilon-greedy: Balance trying new actions vs using knowledge
- Decay epsilon over time (1.0 → 0.01)

**Applications:**
- **Game Playing**: Human-level Atari, AlphaGo, StarCraft II, Dota 2
- **Robotics**: Manipulation, navigation, locomotion
- **Resource Management**: Data center cooling (Google, 40% energy reduction), traffic control
- **Recommendations**: YouTube, Netflix (long-term engagement)

**DQN Improvements:**
- Double DQN, Dueling DQN, Prioritized replay
- Rainbow DQN: Combines 6 improvements, SOTA Atari

**Modern RL:** PPO, SAC (policy gradient), model-based RL, offline RL

**Impact:** Proved deep learning works for sequential decision making, enabled AI for robotics and complex strategy games

---

## Essential Training Techniques

### 27. Normalization Techniques 🔧
**Path:** `examples/27-normalization`
**Run:** `cargo run --package normalization`

Critical techniques enabling deep network training: BatchNorm, LayerNorm, GroupNorm, InstanceNorm.

**Problem Solved: Internal Covariate Shift**
- Activations shift during training
- Deep networks become unstable
- Normalization stabilizes and accelerates training

**Key Techniques:**
- **BatchNorm**: Normalize across batch (CNNs, large batches)
- **LayerNorm**: Normalize across features (Transformers, RNNs)
- **GroupNorm**: Normalize within groups (small batches, CNNs)
- **InstanceNorm**: Normalize per sample (style transfer, GANs)

**Benefits:**
- 10-100× faster training convergence
- Higher learning rates possible
- Reduces sensitivity to initialization
- Acts as regularization

**When to Use:**
- CNNs + large batch → BatchNorm
- CNNs + small batch → GroupNorm
- Transformers/RNNs → LayerNorm
- Style transfer/GANs → InstanceNorm

**Impact:** Enabled ResNet-152 (2015), GPT-3 (LayerNorm), all modern deep networks rely on normalization

---

### 28. Regularization & Dropout 🛡️
**Path:** `examples/28-regularization`
**Run:** `cargo run --package regularization`

Essential techniques to prevent overfitting and improve model generalization.

**Core Problem: Overfitting**
- Model memorizes training data
- Poor performance on new data
- Need to constrain model complexity

**Key Techniques:**
- **L2 Regularization** (Weight Decay): `Loss + λ × Σw²` → Prevents large weights
- **L1 Regularization** (Lasso): `Loss + λ × Σ|w|` → Sparse models, feature selection
- **Dropout**: Randomly drop neurons during training → Ensemble effect
- **DropConnect**: Drop weights instead of activations
- **Early Stopping**: Stop when validation loss stops improving
- **Data Augmentation**: Increase effective dataset size

**Dropout Rates:**
- Fully connected layers: 0.5
- CNNs: 0.1-0.3
- Input layer: 0.1-0.2

**Typical Configurations:**
- **ResNet**: L2=0.0001, no dropout (BatchNorm provides regularization)
- **Transformer**: L2=0.01, Dropout=0.1
- **Small dataset**: L2=0.01, Dropout=0.5, heavy augmentation

**Applications:** All production models use multiple regularization techniques to prevent overfitting

**Impact:** Dropout (2012) became standard in neural networks, essential for training with limited data

---

### 29. Transfer Learning & Fine-Tuning 🔄
**Path:** `examples/29-transfer-learning`
**Run:** `cargo run --package transfer-learning`

**The most practical deep learning workflow**: Start from pre-trained models and adapt to your specific task.

**Why Transfer Learning?**
- Train with 10-100× less data
- Converge 10× faster
- Better final performance
- Enable DL for small datasets

**Core Idea:**
- Pre-train on large dataset (ImageNet, Wikipedia)
- Transfer knowledge to your task
- Fine-tune on your data (100s vs millions of samples)

**Two Approaches:**

**1. Feature Extraction (Freeze Early Layers)**
```
Input → [Frozen Backbone] → [New Trainable Head]
```
- When: Small dataset (<1K), similar to pre-training
- Training time: Minutes instead of hours

**2. Fine-Tuning (Train All Layers)**
```
Stage 1: Train head only (2-5 epochs)
Stage 2: Unfreeze all, small LR (10-20 epochs)
```
- When: Larger dataset (>10K), different from pre-training
- Use discriminative learning rates per layer

**Popular Models:**
- **Vision**: ResNet-50 (ImageNet), EfficientNet, ViT
- **NLP**: BERT (Wikipedia), RoBERTa, GPT-2, T5
- **Multi-modal**: CLIP (image + text)

**Key Insights:**
- Early layers learn universal features (edges, textures)
- Later layers learn task-specific features
- Use 10-100× smaller learning rate than training from scratch
- **Critical**: Match pre-training normalization exactly!

**Real-World Results:**
- Medical imaging: 85% → 92% accuracy vs from scratch
- Sentiment analysis: 89% with 5K samples (would need 50K+ from scratch)
- Object detection: Works with 1K images (would need 10K+ from scratch)

**Applications:** Production deep learning, medical imaging, custom object detection, domain adaptation, few-shot learning

**Impact:** How practitioners actually use deep learning - transfer learning is the default workflow in industry

---

## Advanced Architectures & Techniques

### 30. Sequence-to-Sequence Models (Seq2Seq) 💬
**Path:** `examples/30-seq2seq`
**Run:** `cargo run --package seq2seq`

Encoder-decoder architecture for variable-length sequence transformation, revolutionized neural machine translation.

**Core Architecture:**
- **Encoder**: Compresses input sequence into context
- **Attention**: Decoder "looks back" at all encoder states
- **Decoder**: Generates output sequence token-by-token

**Key Innovation: Attention Mechanism (2015)**
```
Without attention: Bottleneck (single context vector)
With attention: Decoder accesses all encoder states
Result: +10-20 BLEU points on translation
```

**Decoding Strategies:**
- **Greedy**: Pick highest-probability word (fast, suboptimal)
- **Beam Search**: Keep top-k hypotheses (better quality)
- **Sampling**: Creative generation (chat, stories)

**Training: Teacher Forcing**
```
Use ground truth (not predictions) as next input
Prevents error compounding during training
```

**Applications:**
- Machine translation (English → French)
- Text summarization (article → summary)
- Dialogue/chatbots (question → answer)
- Image captioning (image → description)
- Speech recognition (audio → text)

**Evolution:** Seq2Seq (2014) → +Attention (2015) → Transformer (2017)

**Modern Context**: Largely replaced by Transformers for NLP, but still used for streaming tasks and resource-constrained deployment

---

### 31. Mixture of Experts (MoE) 🧠
**Path:** `examples/31-mixture-of-experts`
**Run:** `cargo run --package mixture-of-experts`

**Sparse activation architecture enabling trillion-parameter models** - used in GPT-4, Switch Transformer, and other modern large models.

**Problem Solved:**
```
Dense model: 10B params = 10B FLOPs per token
MoE model: 100B params, but only 10B activated per token
Result: 10× bigger model with same compute!
```

**Architecture:**
```
Input → Router (which experts?) → Top-k Experts → Weighted Combination
```

**Routing Strategies:**
- **Top-1** (Switch): Single expert (fastest)
- **Top-2**: Balance capacity vs efficiency (common)
- **Top-k**: More experts for complex inputs

**Expert Specialization** (emergent during training):
```
Expert 1: Punctuation and grammar
Expert 2: Named entities (people, places)
Expert 3: Numbers and dates
Expert 4: Technical/scientific terms
Expert 5: Common words
Expert 6: Rare words
```

**Load Balancing:**
- Auxiliary loss encourages uniform distribution
- Capacity limits prevent overloading
- Random routing ensures all experts trained

**Famous MoE Models:**
- **Switch Transformer** (Google, 1.6T params): 7× speedup over T5-XXL
- **GLaM** (Google, 1.2T params): Beats GPT-3 with 1/3 energy
- **GPT-4** (OpenAI, rumored ~1.8T params): 8 experts
- **Mixtral 8x7B** (Mistral, 47B params): Open-source, beats GPT-3.5

**Scaling:**
```
Dense GPT-3: 175B params, 175B FLOPs
MoE Switch: 1.6T params, ~200B FLOPs (same compute!)
```

**Impact:** Enables trillion-scale models, likely standard for future large models

---

### 32. Meta-Learning: "Learning to Learn" 🎓
**Path:** `examples/32-meta-learning`
**Run:** `cargo run --package meta-learning`

Fast adaptation to new tasks with minimal examples, mimicking human learning.

**Problem:**
```
Traditional ML: New task → Need 1000s of examples
Human learning: See 1-5 examples → Generalize
Meta-learning: Train on task distribution → Adapt with 1-10 examples!
```

**N-way K-shot Classification:**
```
5-way 1-shot: 5 classes, 1 example per class (5 total examples)
Classify new examples into one of 5 classes
```

**Core Concept:**
```
Meta-Train: Learn on tasks T₁, T₂, ..., Tₙ
Meta-Test: New task → Adapt quickly
Key: Learn "how to learn" rather than specific task
```

**Major Approaches:**

**1. Prototypical Networks (Metric Learning)**
```
1. Embed support examples
2. Compute class prototypes (mean per class)
3. Classify query by nearest prototype
```

**2. MAML (Optimization-Based)**
```
Learn initialization θ for fast fine-tuning
One gradient step adapts to new task
Second-order optimization through adaptation
```

**3. Matching Networks (Attention-Based)**
```
Attention over support set
Weighted vote for classification
Differentiable nearest neighbor
```

**Benchmarks:**
- **Omniglot**: 1,623 characters, "MNIST of few-shot learning"
- **Mini-ImageNet**: 100 classes, standard benchmark
- Performance: 60-85% (5-way 1-shot), vs 20% random

**Applications:**
- Drug discovery (few examples of effective compounds)
- Medical diagnosis (rare diseases, few training examples)
- Robotics (fast task adaptation with 10-20 demonstrations)
- Personalization (cold-start users, few ratings)
- Low-resource NLP (few parallel sentences)

**vs Transfer Learning:**
```
Transfer: Single task → Fine-tune on new task (100s examples)
Meta: Task distribution → Adapt to new task (1-10 examples)
```

**Modern:** In-context learning in GPT-3/GPT-4 is form of meta-learning

**Impact:** Enables human-like rapid learning from few examples

---

### 33. Neural Architecture Search (NAS) 🔍
**Path:** `examples/33-neural-architecture-search`
**Run:** `cargo run --package neural-architecture-search`

**AutoML for automatically discovering optimal neural architectures** - often finds better designs than human experts.

**Problem:**
```
Manual design: Expert trial-and-error, months/years
NAS: Automated search, systematic exploration
Result: Often beats human designs!
```

**Three Components:**

**1. Search Space** (What architectures to consider)
```
Operations: Conv3x3, Conv5x5, MaxPool, Identity
Connections: Skip, sequential
Depth: 10-50 layers

Cell-based: Design reusable "cell", stack to form network
```

**2. Search Strategy** (How to explore)
```
• Random search (surprisingly effective baseline)
• Reinforcement learning (NASNet)
• Evolutionary algorithms (mutation, crossover)
• Gradient-based (DARTS) - differentiable search!
```

**3. Performance Estimation** (How to evaluate)
```
• Full training (accurate, expensive)
• Low fidelity (few epochs, faster)
• Weight sharing (one-shot NAS)
• Predictors (learned performance models)
```

**Famous Results:**

**NASNet (Google, 2017)**
```
Method: RL-based search
Cost: 800 GPU days
Result: Beat human designs on ImageNet
Transfer: NASNet cells used in detection, segmentation
```

**DARTS (2018)**
```
Method: Gradient-based (differentiable)
Cost: 4 GPU days (200× faster than NASNet!)
Result: Competitive performance
Impact: Simple, reproducible, widely adopted
```

**EfficientNet (Google, 2019)**
```
Method: NAS + compound scaling
Result: SOTA ImageNet (84.3% top-1)
Efficiency: Fewer params than previous SOTA
```

**Discovered Patterns:**
- Depthwise separable convolutions (now widely used)
- Skip connections (validates ResNet)
- Irregular patterns (no human bias for symmetry)

**Variations:**
- **Hardware-aware NAS**: Optimize latency, energy (MobileNetV3)
- **Transferable NAS**: Search on CIFAR-10, transfer to ImageNet
- **Once-for-all**: Single network → multiple configurations

**When to Use:**
```
✅ Significant compute (10+ GPUs)
✅ Need SOTA performance
✅ Specific constraints (mobile, edge)

❌ Limited resources (<10 GPUs)
❌ Well-solved problem (use existing)
❌ Quick prototype needed
```

**Future:** Essential AutoML tool, democratizes deep learning for non-experts

**Impact:** Transforms architecture design from manual art to automated science

---

## Practical Machine Learning Applications

### 34. Time Series Forecasting 📈
**Path:** `examples/34-time-series-forecasting`
**Run:** `cargo run --package time-series-forecasting`

Predicting future values based on historical patterns in sequential data.

**Core Concepts:**
- Time series components (trend, seasonality, cycles, noise)
- Stationarity and differencing
- Moving averages and exponential smoothing
- AutoRegressive (AR) models

**Classical Methods:**
- **ARIMA/SARIMA**: Statistical models for univariate time series
- **Prophet** (Facebook): Additive model with trend, seasonality, holidays
- **Exponential Smoothing**: Weighted averages with recent emphasis

**Deep Learning:**
- **LSTM/GRU**: Capture long-term dependencies in sequences
- **Seq2Seq**: Multi-step ahead forecasting
- **Attention-based**: Focus on relevant historical periods

**Feature Engineering:**
- Lag features (previous values)
- Rolling statistics (moving avg, std)
- Date features (day of week, month, seasonality)
- Cyclical encoding (sin/cos for periodic patterns)

**Evaluation:**
- MAE, RMSE, MAPE, SMAPE
- Time-based cross-validation (walk-forward)

**Applications:**
- **Finance**: Stock prices, portfolio optimization, risk management
- **Retail**: Demand forecasting, inventory optimization, sales planning
- **Energy**: Electricity demand, renewable energy prediction, grid management
- **Weather**: Temperature, precipitation, extreme events
- **Operations**: Website traffic, server load, capacity planning

**Modern Context:** Essential for business planning and operations; combines classical statistical methods with deep learning for complex patterns

---

### 35. Recommendation Systems ⭐
**Path:** `examples/35-recommendation-systems`
**Run:** `cargo run --package recommendation-systems`

Predict user preferences and suggest relevant items to enhance user experience and engagement.

**Core Approaches:**

**1. Collaborative Filtering**
- **User-based**: Find similar users → recommend what they liked
- **Item-based**: Find similar items → recommend those (Amazon uses this)
- **Similarity metrics**: Cosine, Pearson correlation, Jaccard

**2. Matrix Factorization**
- Decompose ratings matrix: `R ≈ U × I^T`
- Learn latent factors (50-200 dimensions)
- **Training**: SVD, ALS (parallelizable), SGD
- Add biases: `r̂ = μ + b_user + b_item + user·item`

**3. Neural Collaborative Filtering**
- Replace dot product with neural network
- Learn non-linear user-item interactions
- **NeuMF**: Combines GMF + MLP paths

**Evaluation:**
- Prediction: RMSE, MAE
- Ranking: Precision@K, Recall@K, NDCG
- Beyond accuracy: Diversity, serendipity, coverage, fairness

**Cold Start Solutions:**
- Content-based filtering for new items
- Demographic features for new users
- Hybrid approaches
- Ask users to rate initial items

**Real-World Systems:**
- **YouTube**: Two-stage (candidate generation → ranking), optimizes watch time
- **Netflix**: Hybrid (collaborative + content + context), personalized thumbnails
- **Amazon**: Item-based CF, "Customers who bought X also bought Y"
- **Spotify**: Collaborative + content-based audio features + NLP on playlists

**Applications:** E-commerce, streaming (video/music), social media, news, job matching, ad targeting

**Impact:** Powers modern platforms, drives engagement and revenue, essential for personalization at scale

---

### 36. Anomaly Detection 🚨
**Path:** `examples/36-anomaly-detection`
**Run:** `cargo run --package anomaly-detection`

Identify unusual patterns that differ from the majority of the data - critical for fraud detection, system monitoring, and security.

**Anomaly Types:**
- **Point anomalies**: Single unusual data point ($10,000 charge vs usual $50)
- **Contextual anomalies**: Unusual in specific context (35°C normal in summer, anomaly in winter)
- **Collective anomalies**: Collection unusual together (intrusion attack sequence)

**Main Approaches:**

**1. Statistical Methods**
- **Z-score**: `z = (x - μ) / σ`, flag if |z| > 2.5-3
- ✅ Simple, fast, interpretable
- ❌ Assumes Gaussian distribution

**2. Distance-Based (KNN)**
- Points far from k nearest neighbors are anomalies
- ✅ No distribution assumption, intuitive
- ❌ O(n²) computation, sensitive to k

**3. Isolation Forest**
- **Key insight**: Anomalies easier to isolate (fewer splits)
- Build ensemble of random isolation trees
- Short path = anomaly, long path = normal
- ✅ Fast O(n log n), handles high dimensions, few parameters
- ❌ Less effective when anomalies cluster

**4. Autoencoder-Based**
- Train on normal data only
- Anomalies have high reconstruction error
- **Threshold**: 95th percentile, μ + 3σ, or domain-based
- ✅ Learns complex patterns, handles high dimensions, unsupervised
- ❌ Requires lots of normal data, black box, threshold tuning

**5. One-Class SVM**
- Learn boundary around normal data
- Points outside = anomalies
- **Kernels**: RBF (most common), linear, polynomial
- ✅ Mathematically well-founded, kernel trick
- ❌ O(n²-n³) training, kernel selection critical

**Evaluation:**
- Precision, Recall, F1-score
- **AUC-PR** (better for imbalanced data than AUC-ROC)
- Trade-off: False positives vs false negatives

**Applications:**
- **Fraud detection**: Credit cards, insurance claims, fake accounts (<0.1% fraud rate)
- **System monitoring**: CPU/memory spikes, network intrusions, DDoS
- **Manufacturing**: Defective products, equipment failures, quality control
- **Healthcare**: Disease outbreaks, unusual vitals, medical imaging
- **Cybersecurity**: Malware, intrusions, user behavior analytics

**Practical Tips:**
- Feature engineering crucial (domain knowledge)
- Ensemble methods (combine multiple approaches)
- Handle concept drift (retrain periodically)
- Explain detections (build trust)

**Challenges:** Highly imbalanced data (99.9% normal), adversarial adaptation (fraudsters evolve), real-time detection

**Impact:** Essential for security, quality, and system reliability across industries

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
    ├── 16-resnet/                # DL Architecture: Very Deep 🏆
    ├── 17-lstm-gru/              # DL Architecture: Sequences 🔗
    ├── 18-unet/                  # DL Architecture: Segmentation 🎨
    ├── 19-vae/                   # DL Architecture: Probabilistic Gen 📊
    ├── 20-diffusion/             # DL Architecture: State-of-the-Art 🌟
        ├── 21-gnn/                   # DL Architecture: Graph Data 🕸️
    ├── 22-attention/             # Core Concept: Attention 🔍
    ├── 23-vision-transformer/    # Transformers for Vision 👁️
    ├── 24-object-detection/      # Real-time Detection 📦
    ├── 25-siamese-networks/      # Similarity Learning 👯
    ├── 26-reinforcement-learning/ # RL & DQN 🎮
    ├── 27-normalization/         # Training Techniques: Normalization 🔧
    ├── 28-regularization/        # Training Techniques: Regularization 🛡️
    ├── 29-transfer-learning/     # Training Techniques: Transfer Learning 🔄
    ├── 30-seq2seq/               # Advanced: Seq2Seq Translation 💬
    ├── 31-mixture-of-experts/    # Advanced: MoE (Trillion-scale) 🧠
    ├── 32-meta-learning/         # Advanced: Learning to Learn 🎓
    ├── 33-neural-architecture-search/ # Advanced: AutoML/NAS 🔍
    ├── 34-time-series-forecasting/ # Practical ML: Time Series 📈
    ├── 35-recommendation-systems/ # Practical ML: Recommendations ⭐
    └── 36-anomaly-detection/     # Practical ML: Anomaly Detection 🚨
```

⭐ = Implemented from scratch
🔥🚀🏆🔗🎨📊🌟🕸️🔍👁️📦👯🎮🔧🛡️🔄💬🧠🎓🔍 = Advanced deep learning architectures & training techniques

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

### 🔴 Advanced Track (Deep Learning Mastery)
9. **09-neural-network** - Deep learning fundamentals
10. **10-deep-learning-basics** - Modern DL concepts
11. **11-cnn** - Computer vision architecture
12. **12-rnn** - Sequential data architecture
13. **13-autoencoder** - Unsupervised deep learning
14. **14-gan** - Generative models & adversarial training
15. **15-transformer** - Attention mechanisms & modern NLP
16. **16-resnet** - Very deep networks & skip connections

### 🟣 Expert Track (State-of-the-Art)
17. **17-lstm-gru** - Advanced sequence modeling & long-term memory
18. **18-unet** - Semantic segmentation & medical imaging
19. **19-vae** - Probabilistic generative models
20. **20-diffusion** - State-of-the-art generation (Stable Diffusion, DALL-E)
21. **21-gnn** - Graph neural networks & non-Euclidean data
22. **22-attention** - Attention mechanisms powering Transformers
23. **23-vision-transformer** - Pure Transformer for vision
24. **24-object-detection** - Real-time object detection (YOLO)
25. **25-siamese-networks** - Similarity learning & metric learning
26. **26-reinforcement-learning** - Deep Q-Networks & RL

### ⚡ Essential Training Techniques
27. **27-normalization** - BatchNorm, LayerNorm (enables deep networks)
28. **28-regularization** - Dropout, L1/L2 (prevents overfitting)
29. **29-transfer-learning** - Fine-tuning pre-trained models (most practical!)

### 🚀 Advanced Modern Techniques
30. **30-seq2seq** - Encoder-decoder + attention (machine translation)
31. **31-mixture-of-experts** - Sparse activation, trillion-scale models (GPT-4)
32. **32-meta-learning** - Learning to learn, few-shot adaptation (MAML)
33. **33-neural-architecture-search** - AutoML, automated architecture discovery (NAS)

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
cargo run --package lstm-gru
cargo run --package unet
cargo run --package vae
cargo run --package diffusion
cargo run --package gnn
```

Build all examples:

```bash
cargo build --workspace
```

## What's New in This Version

### 🎯 Comprehensive Coverage
- **36 examples** covering the entire ML/DL landscape from fundamentals to cutting-edge production techniques
- Clear progression: Fundamentals → Traditional ML → Deep Learning → State-of-the-Art → Training Techniques → Advanced Modern → Practical Applications
- Six learning tracks: Beginner 🟢 → Intermediate 🟡 → Advanced 🔴 → Expert 🟣 → Training Techniques ⚡ → Advanced Modern 🚀 → Practical ML 📊

### 🆕 Deep Learning Architectures (11 Core + 5 Advanced)
**Core Architectures:**
- **CNN**: Convolutional networks for computer vision
- **RNN**: Recurrent networks for sequences (text, time series)
- **Autoencoder**: Unsupervised learning for compression and generation

**Advanced Architectures:**
- **GAN**: Generative adversarial networks for data generation
- **Transformer**: Attention mechanisms powering GPT, BERT, ChatGPT
- **ResNet**: Residual networks enabling very deep architectures

**State-of-the-Art Architectures:**
- **LSTM/GRU**: Advanced sequence modeling with long-term memory
- **U-Net**: Semantic segmentation for medical imaging
- **VAE**: Probabilistic generative models with smooth latent spaces
- **Diffusion Models**: State-of-the-art generation (Stable Diffusion, DALL-E)
- **GNN**: Graph neural networks for non-Euclidean data (social networks, molecules)
- **Attention Mechanisms**: Query-Key-Value framework, foundation of Transformers
- **Vision Transformer**: Pure Transformer for images, no convolutions
- **YOLO**: Real-time object detection (45+ FPS)
- **Siamese Networks**: Similarity learning, one-shot learning, metric learning
- **Deep Q-Networks**: Reinforcement learning for sequential decisions

### ⚡ Essential Training Techniques
- **Normalization**: BatchNorm, LayerNorm, GroupNorm, InstanceNorm - enables training deep networks
- **Regularization**: L1/L2, Dropout, DropConnect, early stopping - prevents overfitting
- **Transfer Learning**: Fine-tuning pre-trained models - the most practical DL workflow

### 🚀 Advanced Modern Techniques
- **Seq2Seq Models**: Encoder-decoder + attention for translation, summarization, dialogue
- **Mixture of Experts**: Sparse activation enabling trillion-parameter models (GPT-4, Switch Transformer)
- **Meta-Learning**: "Learning to learn" - rapid adaptation with 1-10 examples (MAML, Prototypical Networks)
- **Neural Architecture Search**: AutoML for discovering optimal architectures (NASNet, DARTS, EfficientNet)

### 📊 Practical Machine Learning Applications (NEW!)
- **Time Series Forecasting**: ARIMA, Prophet, LSTM for sequential predictions (stock prices, weather, sales)
- **Recommendation Systems**: Collaborative filtering, matrix factorization, neural CF (YouTube, Netflix, Amazon-style)
- **Anomaly Detection**: Isolation Forest, autoencoders, One-Class SVM for fraud and system monitoring

### 📖 Enhanced Documentation
- Each example includes comprehensive theory
- Mathematical foundations with clear explanations
- Real-world applications and use cases
- Historical context and modern impact
- Comparisons between techniques
- Production deployment considerations

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
1. **Foundations** (01-08): Start with `01-data-preprocessing`, work through traditional ML, master `08-gradient-descent`
2. **Deep Learning Basics** (09-10): Understand `09-neural-network` from scratch, study `10-deep-learning-basics` theory
3. **Core Architectures** (11-13): Learn `11-cnn` (vision), `12-rnn` (sequences), `13-autoencoder` (unsupervised)
4. **Advanced Architectures** (14-16): Master `14-gan` (generation), `15-transformer` (attention), `16-resnet` (very deep networks)
5. **State-of-the-Art** (17-26):
   - `17-lstm-gru`: Advanced sequences with memory
   - `18-unet`: Semantic segmentation
   - `19-vae`: Probabilistic generation
   - `20-diffusion`: Modern AI art (Stable Diffusion, DALL-E)
   - `21-gnn`: Graph-structured data (molecules, social networks)
   - `22-attention`: Attention mechanisms (foundation of Transformers)
   - `23-vision-transformer`: Transformers for images
   - `24-object-detection`: Real-time detection (YOLO)
   - `25-siamese-networks`: Similarity learning
   - `26-reinforcement-learning`: Deep Q-Networks
6. **Essential Training Techniques** (27-29):
   - `27-normalization`: BatchNorm, LayerNorm - enables deep training
   - `28-regularization`: Dropout, L1/L2 - prevents overfitting
   - `29-transfer-learning`: Fine-tuning pre-trained models - **most practical!**
7. **Advanced Modern Techniques** (30-33):
   - `30-seq2seq`: Encoder-decoder + attention for translation
   - `31-mixture-of-experts`: Sparse activation, trillion-scale models (GPT-4 architecture)
   - `32-meta-learning`: Learning to learn, few-shot adaptation
   - `33-neural-architecture-search`: AutoML, automated architecture discovery
8. **Practical Machine Learning Applications** (34-36):
   - `34-time-series-forecasting`: ARIMA, Prophet, LSTM for sequential data (stocks, weather, sales)
   - `35-recommendation-systems`: Collaborative filtering, matrix factorization (YouTube, Netflix, Amazon)
   - `36-anomaly-detection`: Isolation Forest, autoencoders, One-Class SVM (fraud, monitoring)

Each example builds on previous concepts, so following the numbered order is recommended! The complete path takes you from basics to cutting-edge AI, production techniques, practical applications, and the future of deep learning.

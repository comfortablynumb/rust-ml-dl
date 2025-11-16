# Model Compression & Deployment 📦

**Making deep learning models smaller, faster, and efficient enough for production deployment** on mobile, edge devices, and resource-constrained environments.

## Overview

Modern deep learning models are massive (GPT-3: 175B parameters, 700GB). Model compression techniques reduce size by **10-100×** and speed up inference by **2-10×** with minimal accuracy loss.

## The Production Problem

**Challenge:** Deploy trained models to production

| Model | Parameters | Size | Inference Time |
|-------|-----------|------|----------------|
| GPT-3 | 175B | 700GB | Too slow for real-time |
| BERT-base | 110M | 440MB | 100ms (GPU) / 1s (CPU) |
| ResNet-50 | 25M | 100MB | 50ms (GPU) / 500ms (CPU) |
| MobileNetV2 | 3.5M | 14MB | 10ms (mobile) |

**Goals:**
- ✅ **Smaller models**: 10-100× compression
- ✅ **Faster inference**: 2-10× speedup
- ✅ **Edge deployment**: Run on mobile/embedded
- ✅ **Lower latency**: Real-time applications
- ✅ **Reduced cost**: Less cloud compute

## Compression Techniques

### 1. Pruning ✂️ (Remove Unnecessary Weights)

**Idea:** Remove weights with small magnitudes - they contribute little to predictions.

**Types:**

#### Unstructured Pruning
```
Remove individual weights based on magnitude
Result: Sparse weight matrix (irregular pattern)
```

**Example:**
```
Original weights:
[0.82  0.01  -0.15  0.93]
[0.03  0.76   0.88  -0.02]

After 50% pruning (|w| < threshold):
[0.82  0.00  0.00  0.93]
[0.00  0.76  0.88  0.00]
```

**Benefits:**
- 90% sparsity possible with <1% accuracy loss
- Works for any architecture

**Drawbacks:**
- Irregular sparsity pattern
- Doesn't speed up inference on standard hardware
- Needs specialized sparse matrix libraries

#### Structured Pruning
```
Remove entire channels, filters, or neurons
Result: Smaller, dense network
```

**Example:**
```
Original: Conv layer with 128 filters
After pruning 50% channels: 64 filters
Actual speedup: 2× (dense operations)
```

**Benefits:**
- Real speedup on any hardware
- Smaller model size
- No specialized libraries needed

**Drawbacks:**
- Less compression than unstructured
- More accuracy degradation

#### Iterative Magnitude Pruning (IMP)

**Algorithm:**
1. Train network to convergence
2. Prune p% of weights with smallest |magnitude|
3. Fine-tune remaining weights
4. Repeat steps 2-3 (prune 20% → 50% → 70% → 90%)

**Results:**
- **ResNet-50:** 90% pruning, 0.5% accuracy loss
- **BERT:** 70% pruning, <1% task degradation
- **Lottery Ticket Hypothesis:** Pruned networks can match or beat original

**When to Use:**
- Post-training compression
- Model size > inference speed priority
- Have GPU for fine-tuning

### 2. Quantization 🔢 (Reduce Precision)

**Idea:** Use fewer bits to represent weights and activations.

**Precision Levels:**

| Type | Bits | Range | Accuracy Loss | Speedup |
|------|------|-------|---------------|---------|
| FP32 (Float) | 32 | Full | 0% (baseline) | 1× |
| FP16 (Half) | 16 | ±65k | ~0% | 2× |
| INT8 | 8 | -128 to 127 | 0.5-2% | 2-4× |
| INT4 | 4 | -8 to 7 | 2-5% | 4-8× |

#### Post-Training Quantization (PTQ)

**Algorithm:**
1. Train model normally (FP32)
2. Calibrate: Run sample data, collect activation statistics
3. Quantize weights and activations to INT8
4. (Optional) Fine-tune for accuracy recovery

**Quantization Formula:**
```
FP32 value: x
Quantized: q = round(x / scale) + zero_point
Dequantized: x' = (q - zero_point) × scale

scale = (max - min) / (2^bits - 1)
```

**Example:**
```
FP32 weight: 0.37
Range: [-1.0, 1.0]
INT8 range: [-128, 127]

scale = 2.0 / 255 = 0.00784
quantized = round(0.37 / 0.00784) = 47
dequantized = 47 × 0.00784 = 0.368 ≈ 0.37
```

**Benefits:**
- **4× smaller models** (FP32 → INT8)
- **2-4× faster inference** on CPU/edge devices
- No retraining needed (PTQ)
- Supported by all hardware

**Specialized Hardware:**
- **Apple Neural Engine**: INT8 operations
- **Google Edge TPU**: INT8 only
- **NVIDIA Tensor Cores**: INT8/INT4
- **Qualcomm Hexagon DSP**: INT8

#### Quantization-Aware Training (QAT)

**Idea:** Simulate quantization during training.

**Algorithm:**
1. Insert fake quantization nodes in forward pass
2. Train with quantization noise
3. Backward pass in FP32 (straight-through estimator)
4. Export quantized model

**Benefits:**
- Better accuracy than PTQ (0.5% vs 2% loss)
- Learns to be robust to quantization
- Used for aggressive quantization (INT4, binary)

**When to Use:**
- Post-training quantization loses >1% accuracy
- Target: INT4 or lower
- Have time/resources for retraining

#### Mixed Precision

**Idea:** Critical layers in FP32, others in INT8.

**Strategy:**
```
First layer: FP32 (sensitive to input)
Middle layers: INT8 (bulk of compute)
Last layer: FP32 (affects final output)
Attention layers: FP16 (preserve accuracy)
```

**Benefits:**
- Optimize accuracy/speed tradeoff
- 3× smaller, <0.5% accuracy loss

### 3. Knowledge Distillation 🎓 (Teacher-Student)

**Idea:** Train small "student" network to mimic large "teacher" network.

**Algorithm:**
```
1. Train large teacher model (e.g., BERT-large)
2. Create smaller student model (e.g., DistilBERT)
3. Train student to match:
   - Teacher's soft outputs (probabilities)
   - Teacher's intermediate representations
4. Student learns from teacher's knowledge
```

**Distillation Loss:**
```
L = α · L_student + (1-α) · L_distillation

L_student = CrossEntropy(student_output, true_labels)
L_distillation = KL_Divergence(student_softmax / T, teacher_softmax / T)

T = temperature (softens probabilities, typically 2-4)
α = balance factor (0.5 typical)
```

**Temperature Effect:**
```
Original logits: [3.0, 1.0, 0.1]

T=1 (hard):   [0.88, 0.09, 0.03]  ← One-hot-like
T=3 (soft):   [0.61, 0.29, 0.10]  ← Reveals relative confidences
```

**Why Soft Targets Help:**
- Reveals relative similarities (dog vs cat vs airplane)
- More information than hard labels
- Regularization effect

**Famous Examples:**

| Teacher | Student | Compression | Accuracy Retained |
|---------|---------|-------------|-------------------|
| BERT-base (110M) | DistilBERT (66M) | 1.6× | 97% |
| BERT-large (340M) | TinyBERT (14M) | 24× | 96% |
| ResNet-152 | ResNet-18 | 8.4× | 95% |
| GPT-2 (1.5B) | DistilGPT-2 (82M) | 18× | ~90% |

**Advanced Distillation:**

#### Feature-Based Distillation
```
Match intermediate layer activations
Student learns internal representations
Better than output-only distillation
```

#### Attention Transfer
```
Transfer attention maps (Transformers)
Student learns what teacher focuses on
Used in DistilBERT
```

#### Self-Distillation
```
Model is its own teacher
Use ensemble or earlier checkpoint
Improves accuracy even without compression
```

**When to Use:**
- Need significant compression (>5×)
- Can afford training time
- Teacher model already trained
- Deployment to resource-constrained devices

### 4. Low-Rank Factorization 📐

**Idea:** Decompose weight matrices into smaller matrices.

**Example: Fully Connected Layer**
```
Original: W ∈ ℝ^{1000×1000} (1M parameters)
Factorized: W ≈ U × V
  U ∈ ℝ^{1000×100}
  V ∈ ℝ^{100×1000}
  Total: 200K parameters (5× compression)
```

**Methods:**
- **SVD (Singular Value Decomposition)**: Keep top-k singular values
- **Tucker Decomposition**: For convolutional layers
- **LoRA** (Low-Rank Adaptation): Fine-tuning with low-rank updates

**Benefits:**
- Automatic compression
- Principled (SVD is optimal low-rank approximation)
- No retraining needed

**Drawbacks:**
- Limited compression (<5×)
- May lose accuracy
- Better for fully-connected than convolutional layers

### 5. Architecture Search & Efficient Designs

**Mobile-Optimized Architectures:**

#### MobileNetV1/V2/V3
```
Depthwise separable convolutions
9× fewer parameters than standard conv
Inverted residuals + linear bottlenecks
Optimized for mobile CPUs
```

#### EfficientNet
```
Compound scaling (depth, width, resolution)
AutoML-discovered architecture
Better accuracy/FLOPS tradeoff
```

#### SqueezeNet
```
Fire modules (squeeze + expand)
AlexNet accuracy with 50× fewer parameters
```

**When to Use:**
- Designing new model
- Deploy to mobile/edge
- Start with efficient architecture rather than compress large one

## Compression Strategies

### Sequential Combination
```
1. Prune: Remove 70% of weights
2. Quantize: FP32 → INT8 (4× smaller)
3. Distill: Train compact student
Total: 50-100× compression
```

### Which Technique When?

| Goal | Best Technique | Compression | Accuracy Loss |
|------|---------------|-------------|---------------|
| Smallest model | Distillation + Quantization | 50-100× | 2-5% |
| Fastest inference | Quantization (INT8) | 4× | <1% |
| Edge deployment | Prune + Quantize | 20-40× | 1-3% |
| Minimal accuracy loss | Pruning (structured) | 2-5× | <0.5% |
| Quick compression | PTQ (INT8) | 4× | 1-2% |

## Real-World Examples

### BERT → DistilBERT (Hugging Face)
```
Compression: 40% smaller, 60% faster
Method: Knowledge distillation
Accuracy: 97% of BERT-base
Use case: Production NLP on CPUs
```

### GPT-3 → GPT-3.5-turbo
```
Method: Distillation + quantization + architectural improvements
Speedup: 10× faster, 90% cheaper
Quality: Similar or better on most tasks
```

### MobileNetV2 (Google)
```
From scratch: Efficient architecture design
Size: 3.5M params (vs ResNet-50: 25M)
Speedup: 5× on mobile
Use case: On-device ML (phones, cameras)
```

### YOLOv8-nano
```
Method: Efficient architecture + pruning
Size: 3MB (vs YOLOv5-large: 90MB)
Speed: 60 FPS on mobile
Use case: Real-time object detection on edge
```

## Deployment Considerations

### Hardware-Specific Optimization

#### CPU Deployment
```
Priority: Quantization (INT8)
Tools: ONNX Runtime, Intel MKL
Speedup: 2-4×
```

#### GPU Deployment
```
Priority: Mixed precision (FP16)
Tools: TensorRT, CUDA
Speedup: 2×
```

#### Mobile (iOS/Android)
```
Priority: Quantization + pruning
Tools: Core ML, TensorFlow Lite
Target: <10MB, <100ms
```

#### Edge (Raspberry Pi, Jetson)
```
Priority: Aggressive quantization (INT8/INT4)
Tools: TensorFlow Lite, ONNX
Target: <5MB, real-time
```

### Deployment Frameworks

| Framework | Best For | Quantization | Pruning | Platforms |
|-----------|----------|--------------|---------|-----------|
| TensorFlow Lite | Mobile/Edge | ✅ INT8 | ✅ | Android, iOS, RPi |
| Core ML | Apple devices | ✅ INT8 | ✅ | iPhone, iPad, Mac |
| ONNX Runtime | Cross-platform | ✅ INT8 | ⚠️ | All |
| TensorRT | NVIDIA GPUs | ✅ INT8/FP16 | ✅ | NVIDIA hardware |
| OpenVINO | Intel CPUs | ✅ INT8 | ✅ | Intel CPUs |

## Best Practices

### Development Workflow
```
1. Train large model (FP32, full size)
2. Evaluate compression techniques:
   - Quick: Post-training quantization
   - Best: Pruning + QAT + distillation
3. Measure: accuracy, latency, size
4. Deploy: Convert to deployment framework
5. Monitor: Production accuracy
```

### Accuracy-Efficiency Tradeoff
```
Start conservative (minimal compression)
Measure production impact
Iteratively increase compression
Stop when accuracy drop unacceptable
```

### Testing
```
✅ Accuracy on validation set
✅ Latency on target hardware (not dev machine!)
✅ Memory footprint
✅ Battery usage (mobile)
✅ Edge cases and failure modes
```

## Key Takeaways

1. **Quantization (INT8)** is the easiest win: 4× smaller, 2-4× faster, <1% loss
2. **Pruning** can remove 70-90% of weights with proper fine-tuning
3. **Distillation** achieves highest compression (>10×) but requires retraining
4. **Combine techniques** for maximum compression (50-100×)
5. **Test on target hardware** - dev machine performance ≠ production
6. **Mobile-first design** - start with MobileNet/EfficientNet rather than compress ResNet

## Running the Example

```bash
cargo run --package model-compression
```

This demonstrates:
- Weight pruning (magnitude-based)
- Quantization (FP32 → INT8 simulation)
- Knowledge distillation basics
- Model size and compression metrics

## References

- **Pruning:** Han et al. (2015) - "Learning both Weights and Connections"
- **Quantization:** Jacob et al. (2018) - "Quantization and Training of Neural Networks"
- **Distillation:** Hinton et al. (2015) - "Distilling the Knowledge in a Neural Network"
- **DistilBERT:** Sanh et al. (2019) - "DistilBERT, a distilled version of BERT"
- **TinyBERT:** Jiao et al. (2020) - "TinyBERT: Distilling BERT for Natural Language Understanding"

## Impact

Model compression enables:
- ✅ **On-device AI** (privacy, latency)
- ✅ **Democratization** (run on cheap hardware)
- ✅ **Sustainability** (less energy, carbon)
- ✅ **Cost reduction** (10× less cloud compute)
- ✅ **Real-time applications** (robotics, AR/VR)

**Without compression, modern AI would be restricted to data centers!**

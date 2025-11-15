# Meta-Learning: "Learning to Learn"

Fast adaptation to new tasks with minimal examples.

## Overview

Meta-learning enables models to learn from a distribution of tasks, allowing rapid adaptation to new tasks with just 1-10 examples — mimicking how humans learn.

## Running

```bash
cargo run --package meta-learning
```

## Core Problem

### Traditional Machine Learning
```
Task A: Need 10,000 examples → Train model
Task B: Need another 10,000 examples → Train new model
```
❌ Slow, expensive, doesn't scale

### Human Learning
```
See 1-5 examples → Immediately generalize
Learn from past experience
```
✅ Fast, efficient, scalable

### Meta-Learning
```
Train on tasks T₁, T₂, ..., Tₙ (task distribution)
Learn "how to learn"
New task: Adapt with 1-10 examples!
```
✅ Human-like adaptation speed

## Key Concepts

### N-way K-shot Classification

**N-way**: Number of classes
**K-shot**: Examples per class

**Example: 5-way 1-shot**
```
Support set: 1 example per class (5 total)
Query: Classify new examples into 1 of 5 classes
```

**Example: 2-way 5-shot**
```
Support set: 5 examples per class (10 total)
Query: Binary classification with more support
```

### Task Structure

```
Support Set (Training for new task):
  • Few examples (1-10 per class)
  • Like tiny "train set"

Query Set (Testing for new task):
  • Test examples
  • Evaluate adaptation quality
```

### Meta-Training vs Meta-Testing

**Meta-Training**
```
Learn on many tasks:
  Task 1: Classify 5 Greek characters
  Task 2: Classify 5 Hebrew characters
  ...
  Task 1000: Classify 5 Arabic characters
```

**Meta-Testing**
```
New unseen task:
  Classify 5 Cyrillic characters (never seen before)
  Model adapts quickly using learned strategy!
```

## Major Approaches

### 1. Metric Learning (Prototypical Networks)

**Concept**: Learn embedding where similar items are close

**Algorithm**:
```
1. Embed support examples: e₁, e₂, ..., eₖ
2. Compute prototypes (mean per class):
   c_k = mean({e_i | class k})
3. Embed query: e_query
4. Classify by nearest prototype
```

**Example (3-way 1-shot)**:
```
Support:
  Class A: cat image → e_cat
  Class B: dog image → e_dog
  Class C: bird image → e_bird

Query: New cat image
  Embed → e_query
  Distances:
    dist(e_query, e_cat)  = 0.2  ← Nearest!
    dist(e_query, e_dog)  = 0.8
    dist(e_query, e_bird) = 1.2
  Prediction: Class A (cat)
```

✅ Simple, intuitive
✅ Works well for few-shot
✅ Scalable to many classes

### 2. Optimization-Based (MAML)

**Concept**: Learn initialization for fast fine-tuning

**Model-Agnostic Meta-Learning (MAML)**:
```
Goal: Learn weights θ such that one gradient step adapts well

For each task:
  1. Start from θ
  2. Adapt: θ' = θ - α·∇L_support(θ)
  3. Evaluate: L_query(θ')
  4. Meta-update: θ ← θ - β·∇L_query(θ')
                        ↑ Gradient THROUGH adaptation!
```

**Key Insight**: Second-order optimization
- Optimize initialization to be "one step away" from good solutions
- Learn how to learn efficiently

**Visualization** (weight space):
```
     Task 1 optimum ★
       /
      /
   θ  ◉  ← Meta-learned initialization
      \
       \
     Task 2 optimum ★

θ positioned for fast adaptation to any task!
```

✅ Model-agnostic (any gradient-based model)
✅ Fast adaptation (1-5 steps)
✅ Strong theoretical foundation
❌ Expensive (second-order gradients)

### 3. Attention-Based (Matching Networks)

**Concept**: Attention over support set

**Algorithm**:
```
1. Embed support and query
2. Compute attention: a_i = softmax(similarity(query, support_i))
3. Weighted vote: P(class) = Σ a_i × label_i
```

✅ Differentiable nearest neighbor
✅ Handles variable support sizes
✅ Interpretable attention

## Benchmarks

### Omniglot ("MNIST of few-shot learning")
```
1,623 characters from 50 alphabets
20 examples per character
Standard: 20-way 1-shot, 20-way 5-shot
```

### Mini-ImageNet
```
100 classes, 600 images per class
Subset of ImageNet
Standard: 5-way 1-shot, 5-way 5-shot
More challenging than Omniglot
```

### Performance (5-way 1-shot accuracy)
```
Random guess:          20%
Baseline (k-NN):       40-50%
Prototypical Networks: 60-70%
MAML:                  60-70%
State-of-the-art 2024: 75-85%
```

## Applications

### 1. Drug Discovery
```
Predict properties of new molecules
Few examples of effective compounds
Meta-learn from similar drug families
```

### 2. Robotics
```
Adapt to new environments/objects
10-20 demonstrations → Master new task
Faster than RL from scratch
```

### 3. Personalization
```
Recommendation systems
Adapt to new users with few ratings
Solve cold-start problem
```

### 4. Medical Diagnosis
```
Rare diseases (few training examples)
New diseases (COVID-19)
Transfer from common to rare conditions
```

### 5. Low-Resource NLP
```
Low-resource languages
Few parallel sentences
Transfer from high-resource languages
```

### 6. Character Recognition
```
New alphabets/scripts
1-5 examples per character
Original Omniglot motivation
```

## Meta-Learning vs Transfer Learning

| Aspect | Transfer Learning | Meta-Learning |
|--------|------------------|---------------|
| **Training** | Single task (ImageNet) | Task distribution |
| **Adaptation** | Fine-tune (100s-1000s examples) | Fast (1-10 examples) |
| **Goal** | Transfer features | Learn learning process |
| **Tasks** | One source → one target | Many tasks → new task |
| **Philosophy** | "What to learn" | "How to learn" |

**Complementary**: Can combine both!
- Pre-train with transfer learning
- Meta-train for few-shot adaptation

## Inductive vs Transductive

### Inductive (Standard)
```
Adapt using only support set
Classify queries one by one
Query set unlabeled during adaptation
```

### Transductive
```
Use support + query sets (unlabeled queries)
Semi-supervised adaptation
Better performance (+5-10% accuracy)
```

## Challenges

### 1. Task Distribution
```
Meta-test tasks must be similar to meta-train
If too different → Poor performance
Example: Train on animals, test on vehicles ❌
```

### 2. Compute Cost
```
MAML: Second-order gradients expensive
Train on thousands of tasks
Longer than standard training
```

### 3. Hyperparameters
```
Sensitive to:
  • Inner learning rate (α)
  • Outer learning rate (β)
  • Number of adaptation steps
  • Task sampling strategy
```

### 4. Evaluation Variance
```
High variance across task samples
Need many test tasks for reliable evaluation
Report confidence intervals
```

## Modern Developments (2024)

### 1. In-Context Learning (GPT-3/GPT-4)
```
Large language models as meta-learners
"Few-shot prompting" = meta-learning
No gradient updates, just conditioning on examples
```

### 2. Task-Adaptive Pre-training
```
Combine transfer learning + meta-learning
Pre-train on large corpus
Meta-train on task distribution
```

### 3. Meta-Learning for RL
```
Quickly adapt to new environments
Model-based + meta-learning
Applications: Robotics, games
```

### 4. Cross-Domain Meta-Learning
```
Meta-train on diverse domains
Generalize to very different domains
```

### 5. Continual Meta-Learning
```
Continuously update meta-knowledge
Don't forget old tasks while learning new
```

## Papers

- [Matching Networks for One Shot Learning](https://arxiv.org/abs/1606.04080) (Vinyals et al., 2016)
- [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175) (Snell et al., 2017)
- [Model-Agnostic Meta-Learning (MAML)](https://arxiv.org/abs/1703.03400) (Finn et al., 2017)
- [Meta-Learning: A Survey](https://arxiv.org/abs/1810.03548) (Hospedales et al., 2020)

## Key Takeaways

✓ **Meta-learning**: "Learning to learn" from task distribution
✓ **Few-shot learning**: Adapt to new tasks with 1-10 examples
✓ **Main approaches**:
  - Metric learning (Prototypical, Matching)
  - Optimization-based (MAML)
  - Model-based (Memory networks)
✓ **Applications**: Drug discovery, robotics, personalization, rare diseases
✓ **Different from transfer**: Multi-task training + fast adaptation
✓ **Modern**: In-context learning in LLMs is form of meta-learning
✓ **Key insight**: Instead of learning one task, learn the learning process itself

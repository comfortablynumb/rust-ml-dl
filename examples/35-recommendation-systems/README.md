# Recommendation Systems

Predict user preferences and suggest relevant items.

## Overview

Recommendation systems power YouTube, Netflix, Amazon, Spotify, and most modern platforms. They help users discover relevant content from millions of options through personalization.

## Running

```bash
cargo run --package recommendation-systems
```

## Types of Recommendation Systems

### 1. Collaborative Filtering

```
"Users who liked X also liked Y"

Uses ratings from similar users/items
No content information needed
Most popular approach
```

**User-based**: Find similar users → recommend what they liked
**Item-based**: Find similar items → recommend those

### 2. Content-Based

```
"Recommend items similar to what user liked"

Uses item features (genre, director, keywords)
No need for other users' data
Works for new users
```

### 3. Hybrid

```
Combine collaborative + content-based
Best of both worlds
Used by Netflix, YouTube
```

## Collaborative Filtering

### Similarity Metrics

**Cosine Similarity**:
```
sim(u, v) = (u · v) / (||u|| × ||v||)

Measures angle between rating vectors
Range: -1 (opposite) to 1 (identical)
```

**Pearson Correlation**:
```
Centers ratings first
Accounts for rating scale differences
```

**Jaccard Similarity**:
```
For binary data (liked/not liked)
|A ∩ B| / |A ∪ B|
```

### User-Based CF

```
Algorithm:
1. Find k most similar users to target user
2. For unrated item i:
   Predict = weighted average of neighbors' ratings
   r̂_{u,i} = Σ sim(u,v) × r_{v,i} / Σ sim(u,v)
3. Recommend items with highest predictions
```

**Example**:
```
Alice wants movie recommendations
Similar users: Bob (sim=0.8), Carol (sim=0.6)

Movie "Inception":
  Bob: 5 stars
  Carol: 4 stars
  Predicted for Alice: (0.8×5 + 0.6×4) / 1.4 = 4.57
```

**Pros**:
✅ Simple, intuitive
✅ No item features needed
✅ Explainable

**Cons**:
❌ Sparsity (most ratings missing)
❌ Scalability (O(users²))
❌ Cold start (new users)

### Item-Based CF

```
More popular than user-based (Amazon uses this)

Algorithm:
1. Pre-compute item-item similarities
2. For user u:
   For each rated item → Find similar items
   Predict ratings for similar items
3. Recommend highest predicted
```

**Why better**:
- Items more stable than users
- Can pre-compute similarities (offline)
- Better explanations ("because you liked X")

## Matrix Factorization

Modern collaborative filtering approach

### Concept

```
R ≈ U × I^T

R: Ratings matrix (n_users × n_items)
U: User factors (n_users × k)
I: Item factors (n_items × k)
k: Latent factors (50-200 typical)
```

**Latent factors** might represent:
- Movies: Action, Romance, Sci-fi, Comedy
- Music: Rock, Pop, Classical, Electronic
- **Learned automatically, not predefined!**

### Example (k=2)

```
User factors:
           Action  Romance
Alice        0.8      0.1
Bob          0.2      0.9

Item factors:
             Action  Romance
Die Hard       0.9      0.1
Titanic        0.1      0.9

Prediction (Alice, Titanic):
  r̂ = [0.8, 0.1] · [0.1, 0.9] = 0.17 (low!)

Prediction (Bob, Titanic):
  r̂ = [0.2, 0.9] · [0.1, 0.9] = 0.83 (high!)
```

### Training

**Loss**:
```
L = Σ (r - û·î)² + λ(||û||² + ||î||²)

Minimize prediction error
λ: Regularization
```

**Optimization**:

**1. SVD (Singular Value Decomposition)**
- Closed-form solution
- Modified for sparse data (Funk SVD)

**2. ALS (Alternating Least Squares)**
- Fix users → optimize items
- Fix items → optimize users
- Repeat until convergence
- Parallelizable, scales to billions

**3. SGD (Stochastic Gradient Descent)**
```
For each rating (u, i, r):
  error = r - û·î
  û += α(error·î - λ·û)
  î += α(error·û - λ·î)
```

### Adding Biases

```
r̂_{u,i} = μ + b_u + b_i + û·î

μ: Global average
b_u: User bias
b_i: Item bias
û·î: Interaction
```

**Example**:
```
μ = 3.5 (global average)
b_Alice = +0.5 (Alice rates higher)
b_Inception = +0.8 (Inception rated higher)
û·î = 0.3 (weak preference match)

r̂ = 3.5 + 0.5 + 0.8 + 0.3 = 5.1
```

## Neural Collaborative Filtering

Use deep learning instead of dot product

### Architecture

```
User ID → Embedding → ┐
                       ├→ Concatenate → MLP → Prediction
Item ID → Embedding → ┘
```

**Why neural?**
- Dot product is linear
- Neural nets learn non-linear interactions
- Can incorporate features

### GMF (Generalized Matrix Factorization)

```
Prediction: σ(h^T (e_u ⊙ e_i))

e_u: User embedding
e_i: Item embedding
⊙: Element-wise product
σ: Sigmoid
```

### MLP Approach

```
Input: [e_u, e_i] (concatenation)
  ↓
Dense(128, ReLU)
  ↓
Dense(64, ReLU)
  ↓
Dense(1, Sigmoid) → Prediction
```

### NeuMF

```
Combines GMF + MLP:
  GMF path: e_u ⊙ e_i
  MLP path: MLP([e_u, e_i])
  Final: Concat → Dense → Prediction
```

## Evaluation Metrics

### RMSE & MAE

```
RMSE = √[Σ (r - r̂)² / n]
MAE = Σ |r - r̂| / n

Measure prediction accuracy
Lower is better
```

### Precision@K & Recall@K

```
Precision@10 = relevant in top 10 / 10
Recall@10 = relevant in top 10 / total relevant

Measure ranking quality
```

### NDCG (Normalized Discounted Cumulative Gain)

```
Accounts for ranking order
Higher-ranked relevant items contribute more
Industry standard
```

## Cold Start Problem

New users or items with no ratings

### Solutions

**1. Content-based**
```
Use item features for new items
Use user demographics for new users
```

**2. Hybrid**
```
Collaborative for existing
Content-based for new
```

**3. Popularity**
```
Recommend popular items
Non-personalized baseline
```

**4. Ask questions**
```
New users rate a few items upfront
Rapid personalization
```

## Real-World Systems

### YouTube

```
Two-stage:
  1. Candidate generation: 100s from millions
  2. Ranking: Rank top 100s → return top 10

Features: History, video features, context
Model: Deep neural networks
Objective: Watch time (not just clicks)
```

### Netflix

```
Hybrid: Collaborative + content + context
Baseline: Matrix factorization
Ranking: Neural networks
Personalization: Thumbnails, artwork
Testing: A/B test everything
```

### Amazon

```
Item-based collaborative filtering
"Customers who bought X also bought Y"
Real-time updates
Complementary items (batteries with toys)
```

### Spotify

```
Collaborative for known songs
Content-based (audio) for new songs
NLP on playlists
Bandits for exploration
```

## Beyond Accuracy

### 1. Diversity
```
Don't recommend only similar items
Expose users to variety
```

### 2. Serendipity
```
Recommend unexpected items
Balance safe bets with surprises
```

### 3. Novelty
```
Recommend items user doesn't know
Not just popular items
```

### 4. Coverage
```
Recommend from full catalog
Avoid filter bubbles
```

### 5. Fairness
```
Don't over-recommend popular items
Give niche items a chance
Avoid bias (gender, age, race)
```

### 6. Explainability
```
"Because you watched X"
"Users who liked X also liked Y"
Trust and transparency
```

## Papers

- [Matrix Factorization Techniques](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) (Koren et al., 2009) - Netflix Prize
- [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) (He et al., 2017)
- [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf) (Covington et al., 2016)
- [The Netflix Recommender System](https://dl.acm.org/doi/10.1145/2843948) (Gomez-Uribe & Hunt, 2015)

## Key Takeaways

✓ **Recommendation systems** personalize content for users
✓ **Collaborative filtering**: Use ratings from similar users/items
✓ **Matrix factorization**: Learn latent factors (SVD, ALS, SGD)
✓ **Neural CF**: Deep learning for non-linear interactions
✓ **Evaluation**: RMSE, Precision@K, Recall@K, NDCG
✓ **Cold start**: Combine with content-based, ask questions
✓ **Real-world**: YouTube, Netflix, Amazon, Spotify
✓ **Beyond accuracy**: Diversity, serendipity, fairness, explainability
✓ **Scalability**: Billions of users/items, real-time updates

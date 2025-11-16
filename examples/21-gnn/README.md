# Graph Neural Networks (GNN) Example

This example explains **Graph Neural Networks**, deep learning architectures designed for graph-structured data.

## Overview

GNNs enable learning on irregular, non-Euclidean data:
- **Social networks**: Users and friendships
- **Molecules**: Atoms and bonds
- **Knowledge graphs**: Entities and relations
- **3D meshes**: Vertices and edges

## Running the Example

```bash
cargo run --package gnn
```

## Why GNNs?

Standard neural networks need regular structure:

```
CNNs: Require grid structure (images)
RNNs: Require sequence structure (text)

But many real-world data are graphs:
• Social networks (irregular connections)
• Molecules (variable structure)
• Road networks (arbitrary topology)
• Knowledge bases (complex relations)

→ Need GNNs!
```

## Graph Basics

```
Graph G = (V, E)
• V: Nodes (vertices)
• E: Edges (connections)

Example: Social Network
    Alice ─── Bob
      │        │
    Carol ─── Dave

Adjacency matrix A:
A[i,j] = 1 if edge between i,j

Node features X:
X[i] = feature vector for node i
```

## Graph Tasks

### 1. Node Classification

```
Predict label for each node

Example:
• Social network: Predict user interests
• Citation network: Classify paper topics
• Protein: Predict protein function
```

### 2. Link Prediction

```
Predict missing or future edges

Example:
• Friend recommendations
• Drug interaction prediction
• Knowledge graph completion
```

### 3. Graph Classification

```
Classify entire graphs

Example:
• Molecule: Toxic or safe?
• Protein: What function?
• Code: Buggy or clean?
```

## Message Passing

**Core idea:** Nodes communicate with neighbors

```
For L layers:
  For each node:
    1. Receive messages from neighbors
    2. Aggregate messages
    3. Update own representation

Result: Each node knows L-hop neighborhood
• Layer 1: Direct neighbors
• Layer 2: 2-hop neighbors
• Layer 3: 3-hop neighbors
```

## GCN (Graph Convolutional Network)

### Formula

```
H^(l+1) = σ(Ã H^(l) W^(l))

Where:
• H^(l): Node features at layer l
• Ã: Normalized adjacency (with self-loops)
• W^(l): Learnable weights
• σ: Activation (ReLU)

For each node i:
h_i^(l+1) = σ(W · average(h_j for j in neighbors(i)))
```

### Intuition

```
Like convolution, but on graphs:
1. Aggregate neighbor features
2. Transform with weights
3. Apply activation

"Graph convolution"!
```

## GNN Variants

### GraphSAGE (Scalable)

```
Problem: GCN uses all neighbors (slow for large graphs)

Solution: Sample fixed number of neighbors

h_i = AGG(sample(neighbors))

Benefits:
✅ Scalable to millions of nodes
✅ Works on new nodes (inductive)
✅ Faster training
```

### GAT (Graph Attention)

```
Learn importance of neighbors:

h_i = Σ_j α_ij W h_j

where α_ij = attention(i, j)

Benefits:
✅ Adaptive neighbor weighting
✅ Interpretable (attention scores)
✅ Often better performance
```

### GIN (Theoretically Powerful)

```
Provably maximally expressive:

h_i = MLP((1+ε) h_i + Σ_j h_j)

Benefits:
✅ Distinguishes more graph structures
✅ Good for graph classification
```

## Applications

### Molecular Property Prediction

```
Graph:
• Nodes: Atoms (C, H, O, N, ...)
• Edges: Chemical bonds

Tasks:
• Solubility
• Toxicity
• Drug efficacy

Impact: Accelerate drug discovery
```

### Social Networks

```
Facebook, Pinterest, LinkedIn:
• Friend recommendations (link prediction)
• Community detection (node clustering)
• Influence prediction

PinSage (Pinterest):
• 3 billion pins
• GraphSAGE-based
• Production system
```

### Knowledge Graphs

```
Google Knowledge Graph, Wikidata:
• Nodes: Entities
• Edges: Relations

Tasks:
• Complete missing facts
• Question answering
• Reasoning
```

### Recommendation Systems

```
User-Item graph:
• Nodes: Users and Items
• Edges: Interactions

YouTube, Amazon, Uber Eats:
• Learn user/item embeddings
• Better than matrix factorization
```

### AlphaFold 2

```
Protein structure prediction:
• Nodes: Amino acids
• Edges: Spatial proximity

GNN updates node features:
→ 3D structure prediction
→ Nobel Prize-worthy breakthrough!
```

## Training

### Node Classification

```
Semi-supervised:
• Only some nodes labeled
• Train on labeled
• Predict all nodes

Loss: Cross-entropy on labeled nodes
```

### Graph Classification

```
Batch of graphs:
1. GNN processes each graph
2. Pool to graph-level representation
3. Classify with MLP

Pooling:
• Global mean/max/sum
• Attention-weighted
```

### Link Prediction

```
1. Get node embeddings: h_i, h_j
2. Compute edge score:
   • Dot product: h_i · h_j
   • MLP: MLP([h_i || h_j])
   • Distance: -||h_i - h_j||

Loss: Binary cross-entropy
• Positive: Existing edges
• Negative: Random non-edges
```

## Challenges

### Over-smoothing

```
Problem: Deep GNNs make nodes too similar

Solutions:
✅ Use 2-3 layers (often enough!)
✅ Skip connections
✅ Jumping knowledge
```

### Scalability

```
Problem: Millions of nodes

Solutions:
✅ Sampling (GraphSAGE)
✅ Clustering
✅ Mini-batching
```

### Heterogeneous Graphs

```
Multiple node/edge types:
• Academic: Authors, Papers, Venues
• E-commerce: Users, Products, Shops

Solution: Type-specific parameters
```

## Implementation Tips

```
Layers: 2-3 (most tasks)
Hidden dim: 64-256
Learning rate: 0.001
Dropout: 0.5
Optimizer: Adam

More layers → over-smoothing!
```

## Modern Developments

### Graph Transformers

```
Apply attention to all nodes:
✅ More expressive
❌ O(N²) complexity

Examples: Graphormer, GraphGPS
```

### Foundation Models

```
Pre-train on large graphs:
• Self-supervised learning
• Transfer to downstream tasks

Examples: GraphMAE, GraphCL
```

## Historical Timeline

**2016:** GCN (Kipf & Welling)
- Practical graph convolutions
- Breakthrough paper

**2017:** GraphSAGE
- Sampling for scalability
- Inductive learning

**2018:** GAT (Graph Attention)
- Attention mechanism

**2019:** GIN
- Theoretical expressiveness

**2020:** AlphaFold 2
- Uses GNNs for protein folding
- Demonstrates real-world impact

**2021+:** Widespread adoption
- Recommendation systems
- Drug discovery
- Knowledge graphs

## Impact

```
Enabled learning on non-Euclidean data
Powers modern recommendation systems
Accelerates scientific discovery
Key component in AlphaFold (protein folding)
```

## Further Reading

- [GCN paper](https://arxiv.org/abs/1609.02907) (Kipf & Welling, 2016)
- [GraphSAGE](https://arxiv.org/abs/1706.02216) (Hamilton et al., 2017)
- [GAT](https://arxiv.org/abs/1710.10903) (Veličković et al., 2018)
- [GIN](https://arxiv.org/abs/1810.00826) (Xu et al., 2019)
- [Graph Neural Networks book](https://graph-neural-networks.github.io/)

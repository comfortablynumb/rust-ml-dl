//! # Graph Neural Networks (GNN)
//!
//! This example explains Graph Neural Networks, designed for learning on
//! non-Euclidean data structures like social networks, molecules, and knowledge graphs.
//!
//! ## The Problem: Irregular Data
//!
//! **Standard neural networks assume grid structure:**
//!
//! ```
//! Images: Regular 2D grid
//! [pixel] [pixel] [pixel]
//! [pixel] [pixel] [pixel]
//! [pixel] [pixel] [pixel]
//! → CNNs work great!
//!
//! Sequences: Regular 1D structure
//! [word] → [word] → [word] → [word]
//! → RNNs/Transformers work great!
//!
//! But what about:
//! Social networks: Irregular connections
//! Molecules: Variable number of atoms/bonds
//! Knowledge graphs: Arbitrary relationships
//! 3D meshes: Non-uniform structure
//! → Need GNNs!
//! ```
//!
//! ## What is a Graph?
//!
//! ```
//! Graph G = (V, E)
//! • V: Set of nodes (vertices)
//! • E: Set of edges (connections)
//!
//! Example: Social Network
//! Nodes: People
//! Edges: Friendships
//!
//!     Alice ---- Bob
//!       |         |
//!     Carol ---- Dave
//!
//! Adjacency matrix A:
//! A[i,j] = 1 if edge between node i and j
//! A[i,j] = 0 otherwise
//!
//! Node features X:
//! X[i] = feature vector for node i
//! Example: [age, city, interests, ...]
//! ```
//!
//! ## Graph Tasks
//!
//! ### 1. Node Classification
//! ```
//! Predict label for each node
//!
//! Example: Social network
//! • Nodes: Users
//! • Task: Predict interests/communities
//! • Use: Friend connections help predict
//!
//! Citation network:
//! • Nodes: Papers
//! • Edges: Citations
//! • Task: Classify paper topic
//! ```
//!
//! ### 2. Link Prediction
//! ```
//! Predict missing or future edges
//!
//! Example: Social network
//! • Given: Current friendships
//! • Predict: Future friendships
//! • Use: Friend recommendations
//!
//! Drug interactions:
//! • Nodes: Drugs
//! • Predict: Which drugs interact
//! ```
//!
//! ### 3. Graph Classification
//! ```
//! Classify entire graph
//!
//! Example: Molecule classification
//! • Input: Molecular graph
//! • Output: Toxic or safe?
//! • Use: Drug discovery
//!
//! Protein function:
//! • Input: Protein structure graph
//! • Output: Function class
//! ```
//!
//! ## Message Passing: The Core Idea
//!
//! **Nodes communicate with neighbors:**
//!
//! ```
//! Goal: Learn node representations that incorporate neighborhood information
//!
//! Iteration:
//! 1. Each node receives messages from neighbors
//! 2. Aggregate messages
//! 3. Update own representation
//! 4. Repeat for L layers
//!
//! Result: L-hop neighborhood information
//! • Layer 1: Direct neighbors
//! • Layer 2: 2-hop neighbors
//! • Layer 3: 3-hop neighbors
//! ```
//!
//! ### Example: Friendship Influence
//!
//! ```
//! Task: Predict if user likes sports
//!
//! Initial features: h_i^(0) = user_features[i]
//!
//! Layer 1:
//! • Bob gets messages from Alice, Carol, Dave
//! • Aggregate: sum/mean/max of their features
//! • Update: h_Bob^(1) = f(h_Bob^(0), aggregate_neighbors)
//! • Now Bob's representation includes friend info!
//!
//! Layer 2:
//! • Bob gets messages from friends-of-friends
//! • Now h_Bob^(2) knows 2-hop neighborhood
//!
//! Final: Classify h_Bob^(L)
//! ```
//!
//! ## Graph Convolutional Network (GCN)
//!
//! **Most popular GNN architecture**
//!
//! ### Forward Pass
//!
//! ```
//! H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
//!
//! Where:
//! • H^(l): Node features at layer l (N × d matrix)
//! • A: Adjacency matrix
//! • Ã = A + I (add self-loops)
//! • D̃: Degree matrix of Ã
//! • W^(l): Learnable weights
//! • σ: Activation (ReLU)
//!
//! Intuition:
//! 1. Multiply features by weights: H^(l) W^(l)
//! 2. Aggregate neighbors: Normalized adjacency × features
//! 3. Activation: σ(...)
//!
//! Self-loops: Node includes its own features
//! Normalization: Prevent vanishing/exploding based on degree
//! ```
//!
//! ### Simplified Explanation
//!
//! ```
//! For each node i:
//!
//! h_i^(l+1) = σ(W^(l) · (Σ_j∈N(i) h_j^(l) / √(deg(i) · deg(j))))
//!
//! In words:
//! 1. Sum neighbor features: Σ h_j
//! 2. Normalize by degree: / √(deg(i) · deg(j))
//! 3. Apply linear transformation: W · ...
//! 4. Apply activation: σ(...)
//!
//! This is a "graph convolution"!
//! Similar to image convolution but for irregular graphs
//! ```
//!
//! ## GNN Variants
//!
//! ### GraphSAGE (Sample and Aggregate)
//!
//! ```
//! Problem: GCN uses all neighbors (expensive for large graphs)
//!
//! Solution: Sample fixed number of neighbors
//!
//! h_i^(l+1) = σ(W^(l) · [h_i^(l) || AGG({h_j^(l) : j ∈ S_i})])
//!
//! Where:
//! • S_i: Sampled neighbors of i (e.g., 10 neighbors)
//! • AGG: Mean, max, or LSTM aggregator
//! • ||: Concatenation
//!
//! Benefits:
//! • Scalable to large graphs
//! • Inductive (works on new nodes)
//! • Faster training
//! ```
//!
//! ### GAT (Graph Attention Network)
//!
//! ```
//! Idea: Learn importance of neighbors
//!
//! h_i^(l+1) = σ(Σ_j∈N(i) α_ij W h_j^(l))
//!
//! Where α_ij: Attention weight for edge i→j
//!
//! Attention mechanism:
//! e_ij = LeakyReLU(a^T [W h_i || W h_j])
//! α_ij = softmax_j(e_ij)
//!
//! Intuition:
//! • Learn which neighbors are more important
//! • Different weights for different edges
//! • Multi-head attention (like Transformers)
//!
//! Benefits:
//! • Adaptively weight neighbors
//! • Interpretable (attention scores)
//! • Often better performance
//! ```
//!
//! ### GIN (Graph Isomorphism Network)
//!
//! ```
//! Maximally powerful GNN (theoretically)
//!
//! h_i^(l+1) = MLP((1 + ε^(l)) · h_i^(l) + Σ_j∈N(i) h_j^(l))
//!
//! Where:
//! • ε: Learnable parameter
//! • MLP: Multi-layer perceptron
//!
//! Key insight:
//! • Distinguishes more graph structures
//! • Provably as powerful as WL-test
//! • Good for graph classification
//! ```
//!
//! ## Pooling for Graph Classification
//!
//! **Aggregate node features to graph-level:**
//!
//! ### Global Pooling
//!
//! ```
//! Readout functions:
//!
//! 1. Global mean:
//! h_G = (1/N) Σ_i h_i
//!
//! 2. Global max:
//! h_G = max_i h_i (element-wise)
//!
//! 3. Global sum:
//! h_G = Σ_i h_i
//!
//! 4. Attention-based:
//! h_G = Σ_i α_i h_i  (learn α_i)
//! ```
//!
//! ### Hierarchical Pooling
//!
//! ```
//! Like max pooling in CNNs:
//!
//! 1. Cluster nodes into groups
//! 2. Create coarser graph
//! 3. Repeat
//!
//! Methods:
//! • DiffPool: Differentiable pooling
//! • TopKPool: Keep top-K nodes by score
//! • SAGPool: Self-attention pooling
//! ```
//!
//! ## Training GNNs
//!
//! ### Node Classification
//!
//! ```
//! # Semi-supervised learning
//! Only some nodes have labels
//!
//! Loss: Cross-entropy on labeled nodes
//! L = -Σ_{i∈labeled} y_i log(ŷ_i)
//!
//! Training:
//! 1. Forward pass on entire graph
//! 2. Compute loss only on labeled nodes
//! 3. Backpropagate through message passing
//! 4. Update weights
//!
//! Inference: Predict all nodes (including unlabeled)
//! ```
//!
//! ### Graph Classification
//!
//! ```
//! Batch of graphs:
//! 1. Process each graph with GNN
//! 2. Pool to graph representation
//! 3. MLP classifier
//! 4. Cross-entropy loss
//!
//! Batch construction:
//! • Combine multiple graphs into one big graph
//! • Track graph membership
//! • Pool within each graph
//! ```
//!
//! ### Link Prediction
//!
//! ```
//! Learn edge existence:
//!
//! 1. Get node embeddings: h_i, h_j = GNN(graph)
//! 2. Compute edge score: score(i,j) = decoder(h_i, h_j)
//!
//! Decoders:
//! • Dot product: h_i^T h_j
//! • Concatenation + MLP: MLP([h_i || h_j])
//! • Distance: -||h_i - h_j||
//!
//! Loss: Binary cross-entropy
//! • Positive edges: Existing edges
//! • Negative edges: Random non-edges (sampling)
//! ```
//!
//! ## Applications
//!
//! ### Molecular Property Prediction
//!
//! ```
//! Graph representation:
//! • Nodes: Atoms (features: atomic number, charge, ...)
//! • Edges: Chemical bonds (features: bond type, ...)
//!
//! Tasks:
//! • Solubility prediction
//! • Toxicity classification
//! • Drug-target binding
//!
//! Impact:
//! • Accelerate drug discovery
//! • Predict properties without experiments
//! • ChEMBL dataset: 2M molecules
//! ```
//!
//! ### Social Network Analysis
//!
//! ```
//! Community detection:
//! • Nodes: Users
//! • Edges: Friendships
//! • Task: Cluster into communities
//!
//! Influence prediction:
//! • Who will adopt a product?
//! • Spread of information
//! • Recommendation systems
//!
//! Facebook Friend Suggestions:
//! • Link prediction
//! • Based on mutual friends, interests
//! ```
//!
//! ### Knowledge Graphs
//!
//! ```
//! Reasoning over facts:
//! • Nodes: Entities (people, places, concepts)
//! • Edges: Relations (is_a, born_in, works_at)
//!
//! Tasks:
//! • Link prediction: Complete missing facts
//! • Entity classification
//! • Question answering
//!
//! Examples:
//! • Google Knowledge Graph
//! • Wikidata
//! • BioMed knowledge bases
//! ```
//!
//! ### Recommendation Systems
//!
//! ```
//! Bipartite graph:
//! • Nodes: Users and Items
//! • Edges: User-item interactions
//!
//! Pinterest, YouTube, Amazon:
//! • GNN learns user/item embeddings
//! • Captures higher-order connectivity
//! • Better than matrix factorization
//!
//! PinSage (Pinterest):
//! • 3 billion pins
//! • GraphSAGE-based
//! • Production system
//! ```
//!
//! ### Traffic Prediction
//!
//! ```
//! Road network:
//! • Nodes: Road segments
//! • Edges: Connections
//! • Features: Speed, volume, time
//!
//! Spatio-temporal GNN:
//! • Spatial: Graph convolution
//! • Temporal: RNN/Transformer
//!
//! Uber, Google Maps:
//! • Predict traffic congestion
//! • Optimize routes
//! ```
//!
//! ### 3D Shape Analysis
//!
//! ```
//! 3D mesh as graph:
//! • Nodes: Vertices
//! • Edges: Mesh edges
//! • Features: 3D coordinates, normals
//!
//! Tasks:
//! • 3D shape classification
//! • Segmentation (part labeling)
//! • Shape generation
//!
//! Applications:
//! • 3D modeling
//! • Autonomous driving (LIDAR)
//! • Robotics
//! ```
//!
//! ## Challenges & Solutions
//!
//! ### Over-smoothing
//!
//! ```
//! Problem: Deep GNNs make all nodes similar
//! • After many layers, features converge
//! • Lose discriminative power
//!
//! Solutions:
//! • Shallow networks (2-3 layers often enough)
//! • Skip connections (like ResNet)
//! • Initial residual: h^(l+1) = h^(l) + GNN(h^(l))
//! • Jumping knowledge: Concatenate all layer outputs
//! ```
//!
//! ### Scalability
//!
//! ```
//! Problem: Large graphs (millions of nodes)
//!
//! Solutions:
//! 1. Sampling (GraphSAGE):
//!    • Sample neighbors instead of using all
//!    • Mini-batch training
//!
//! 2. Clustering:
//!    • Partition graph into clusters
//!    • Process clusters separately
//!
//! 3. Simplification:
//!    • Pre-compute propagation (SGC)
//!    • Linear models on pre-processed features
//! ```
//!
//! ### Heterogeneous Graphs
//!
//! ```
//! Multiple node/edge types:
//!
//! Example: Academic graph
//! • Nodes: Authors, Papers, Venues
//! • Edges: Writes, Cites, PublishedIn
//!
//! Solution: Heterogeneous GNN
//! • Different parameters per edge type
//! • Aggregate by relation type
//! • R-GCN, HGT (Heterogeneous Graph Transformer)
//! ```
//!
//! ### Dynamic Graphs
//!
//! ```
//! Graphs that change over time:
//!
//! Example: Social network
//! • New users join
//! • Friendships form/break
//! • User features change
//!
//! Solutions:
//! • Temporal GNN: GNN + RNN
//! • Snapshot-based: Process snapshots independently
//! • Continuous-time: Event-based updates
//! ```
//!
//! ## GNN vs Other Methods
//!
//! ### GNN vs Graph Kernels
//!
//! ```
//! Graph Kernels (traditional):
//! • Hand-crafted similarity functions
//! • No learned representations
//! • Limited expressiveness
//!
//! GNN:
//! ✅ Learn features end-to-end
//! ✅ More expressive
//! ✅ Better performance
//! ✅ Scalable
//! ```
//!
//! ### GNN vs Matrix Factorization
//!
//! ```
//! For link prediction/recommendation:
//!
//! Matrix Factorization:
//! • Only uses edge information
//! • Linear model
//!
//! GNN:
//! ✅ Uses node features
//! ✅ Higher-order connectivity
//! ✅ Non-linear
//! ✅ Better accuracy
//! ```
//!
//! ## Implementation Tips
//!
//! ### Number of Layers
//!
//! ```
//! 2-3 layers: Most tasks
//! • Sufficient for local neighborhoods
//! • Avoid over-smoothing
//!
//! 4-6 layers: Specific cases
//! • Need long-range dependencies
//! • Use skip connections
//!
//! > 6 layers: Rarely helps
//! • Over-smoothing issue
//! • Use graph transformers instead
//! ```
//!
//! ### Hyperparameters
//!
//! ```
//! Hidden dimensions: 64-512
//! Learning rate: 0.001-0.01
//! Dropout: 0.5 (prevents overfitting)
//! Batch size: 32-128 (graph classification)
//! Optimizer: Adam
//! Epochs: 100-500
//! ```
//!
//! ### Data Splits
//!
//! ```
//! Node classification:
//! • Transductive: Train on partial graph
//! • Inductive: Train on separate graphs
//!
//! Graph classification:
//! • Standard train/val/test split
//! • Stratify by class
//! ```
//!
//! ## Modern Developments
//!
//! ### Graph Transformers
//!
//! ```
//! Apply transformer attention to graphs:
//! • Attention over all nodes (not just neighbors)
//! • Positional encodings for graph structure
//! • More expressive than GNNs
//!
//! Examples:
//! • Graph Transformer (GT)
//! • GraphGPS
//! • Graphormer
//!
//! Trade-off: O(N²) complexity
//! ```
//!
//! ### Graph Foundation Models
//!
//! ```
//! Pre-train on large graph datasets:
//! • Self-supervised learning
//! • Transfer to downstream tasks
//!
//! Examples:
//! • GraphMAE (masked autoencoders)
//! • GraphCL (contrastive learning)
//! • GROVER (molecular pre-training)
//! ```
//!
//! ## Historical Impact
//!
//! **2009:** Spectral graph convolutions
//! - Theoretical foundation
//! - Not practical
//!
//! **2014:** DeepWalk, Node2Vec
//! - Graph embeddings
//! - Random walk based
//!
//! **2016:** GCN (Graph Convolutional Network)
//! - Kipf & Welling
//! - Practical message passing
//! - Breakthrough paper
//!
//! **2017:** GraphSAGE
//! - Sampling for scalability
//! - Inductive learning
//!
//! **2018:** GAT (Graph Attention)
//! - Attention mechanism
//! - Better performance
//!
//! **2019:** GIN (Graph Isomorphism)
//! - Theoretical expressiveness
//! - WL-test equivalence
//!
//! **2020+:** Widespread adoption
//! - Pinterest (PinSage)
//! - Alibaba (recommendations)
//! - DeepMind (AlphaFold uses GNN)
//! - Drug discovery companies
//!
//! **Legacy:**
//! - Enabled learning on graph-structured data
//! - Key component in modern AI systems
//! - Active research area

fn main() {
    println!("=== Graph Neural Networks (GNN) ===\n");

    println!("This example explains GNNs, neural networks for graph-structured data");
    println!("like social networks, molecules, and knowledge graphs.\n");

    println!("📚 Key Concepts Covered:");
    println!("  • Graph representation and tasks");
    println!("  • Message passing framework");
    println!("  • GCN, GraphSAGE, GAT architectures");
    println!("  • Node, edge, and graph-level predictions");
    println!("  • Applications: molecules, social networks, recommendations");
    println!("  • Scalability and over-smoothing challenges\n");

    println!("🎯 Why This Matters:");
    println!("  • Handles non-Euclidean data (not grids or sequences)");
    println!("  • Powers modern recommendation systems (Pinterest, YouTube)");
    println!("  • Accelerates drug discovery (molecular property prediction)");
    println!("  • Enables knowledge graph reasoning");
    println!("  • Used in AlphaFold, traffic prediction, 3D analysis\n");

    println!("See the source code documentation for comprehensive explanations!");
}

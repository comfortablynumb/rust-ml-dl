// ============================================================================
// Neural Architecture Search (NAS)
// ============================================================================
//
// AutoML for automatically discovering optimal neural network architectures
// instead of manual expert design. The future of deep learning architecture design.
//
// PROBLEM SOLVED:
// ----------------
// Traditional approach (manual design):
//   • Expert designs architecture (LeNet, ResNet, Transformer, etc.)
//   • Trial and error process
//   • Requires deep expertise and intuition
//   • Time-consuming (months/years for breakthrough)
//   • Human bias limits exploration
//
// Neural Architecture Search:
//   • Automatically search for optimal architecture
//   • Explore huge search space systematically
//   • Often finds better architectures than human experts
//   • Democratizes deep learning (non-experts can get SOTA)
//
// Key Question: Can we automate the discovery of neural architectures?
//
// CORE COMPONENTS:
// ----------------
//
// 1. Search Space:
//    • Set of possible architectures to consider
//    • Example: "All CNNs with 10-50 layers, using conv3x3, conv5x5, pooling"
//
// 2. Search Strategy:
//    • Algorithm to explore search space
//    • Examples: Random search, RL, evolution, gradient-based
//
// 3. Performance Estimation:
//    • How to evaluate candidate architectures
//    • Full training (accurate but expensive) vs proxy (fast but noisy)
//
// NAS Loop:
// ```
// repeat:
//   1. Search strategy proposes architecture
//   2. Train and evaluate architecture
//   3. Update search strategy based on performance
// until: Compute budget exhausted or convergence
// ```
//
// SEARCH SPACE DESIGN:
// --------------------
//
// Defines "what architectures can be discovered"
//
// 1. Macro Search Space (Global structure):
//    • Chain of operations: Conv → Pool → Conv → FC
//    • Skip connections: Add or not?
//    • Number of layers: 10, 20, 50?
//
// 2. Micro Search Space (Cell-based):
//    • Design reusable "cell"
//    • Stack cells to form network
//    • Example: NASNet cell repeated N times
//
// 3. Hierarchical Search Space:
//    • Multiple levels: Network → Block → Layer → Operation
//    • Balance flexibility and search efficiency
//
// Example Micro Search Space (Cell):
// ```
// Cell has N nodes
// Each node: Apply operation to 2 inputs
//
// Operations: {
//   3x3 conv, 5x5 conv, 3x3 depthwise conv,
//   3x3 max pool, 3x3 avg pool, identity, zero
// }
//
// Connections: Choose 2 from previous nodes
//
// Search: Find best operations + connections
// ```
//
// Cell Stacking:
//   • Normal Cell: Preserves spatial dimensions
//   • Reduction Cell: Reduces spatial size (like pooling)
//   • Network: N × [Normal] + [Reduction] + N × [Normal] + [Reduction] + ...
//
// SEARCH STRATEGIES:
// ------------------
//
// 1. RANDOM SEARCH:
//    • Baseline: Sample architectures randomly
//    • Surprisingly effective! (good search space design crucial)
//    • No learning, just exploration
//
//    Advantages:
//      ✅ Simple, easy to parallelize
//      ✅ No bias
//    Disadvantages:
//      ❌ Inefficient, many samples needed
//      ❌ Doesn't learn from past evaluations
//
// 2. REINFORCEMENT LEARNING (NASNet, 2017):
//    • Treat architecture design as sequential decision problem
//    • Controller (RNN) generates architecture
//    • Reward: Validation accuracy
//    • Update controller with policy gradient (REINFORCE)
//
//    Example:
//    ```
//    Controller: RNN that outputs sequence of architecture decisions
//      Step 1: Output "conv3x3"
//      Step 2: Output "connect to node 0"
//      Step 3: Output "conv5x5"
//      ...
//      Architecture encoded as sequence
//
//    Train architecture → Get accuracy A
//    Reward R = A (or R = A - baseline)
//    Update controller: ∇J = E[R × ∇log P(architecture)]
//    ```
//
//    Advantages:
//      ✅ Learns from experience
//      ✅ Can handle complex search spaces
//    Disadvantages:
//      ❌ Expensive: Train thousands of architectures
//      ❌ High variance, requires many samples
//
// 3. EVOLUTIONARY ALGORITHMS:
//    • Population of architectures
//    • Mutate and crossover to create offspring
//    • Select best for next generation
//
//    Algorithm:
//    ```
//    Initialize population P of random architectures
//    repeat:
//      Evaluate fitness (accuracy) of each architecture in P
//      Select top K parents
//      Generate offspring by:
//        - Mutation: Randomly modify operation/connection
//        - Crossover: Combine two parent architectures
//      P = offspring
//    ```
//
//    Advantages:
//      ✅ Parallelizable (evaluate population in parallel)
//      ✅ Robust, handles discrete search spaces
//    Disadvantages:
//      ❌ Many evaluations needed
//      ❌ Requires good mutation/crossover operators
//
// 4. GRADIENT-BASED (DARTS, 2018):
//    • Make architecture search differentiable!
//    • Continuous relaxation of discrete choices
//    • Use gradient descent to search
//
//    Key Idea: Instead of choosing one operation, use weighted sum of all
//    ```
//    Discrete:
//      Choose operation i from {conv3x3, conv5x5, pool}
//
//    Continuous (DARTS):
//      output = Σ α_i × operation_i(input)
//      α_i = softmax(architecture parameters)
//
//    Search: Optimize α using gradient descent
//    Final: Choose argmax(α) for each edge
//    ```
//
//    Bilevel Optimization:
//    ```
//    min_α L_val(w*(α), α)           # Architecture loss
//    s.t. w*(α) = argmin_w L_train(w, α)  # Weight loss
//    ```
//
//    Approximate with alternating optimization:
//    ```
//    Step 1: Update w (weights) on train set
//    Step 2: Update α (architecture) on validation set
//    ```
//
//    Advantages:
//      ✅ Fast: 1-4 GPU days vs 1000s for RL
//      ✅ Efficient: Gradients guide search
//      ✅ Memory efficient with one-shot approach
//    Disadvantages:
//      ❌ Approximation: May not find truly discrete optimum
//      ❌ Performance gap: Continuous relaxation != discrete choice
//
// PERFORMANCE ESTIMATION:
// -----------------------
//
// Problem: Training each architecture to convergence too expensive!
//   • 1000 candidate architectures × 100 GPU hours = 100,000 GPU hours
//   • Need faster evaluation
//
// Solutions:
//
// 1. Lower Fidelity Estimates:
//    • Train for fewer epochs (10 instead of 200)
//    • Train on subset of data (10% instead of 100%)
//    • Smaller resolution (32×32 instead of 224×224)
//    • Approximate ranking, not exact performance
//
// 2. Learning Curve Extrapolation:
//    • Train for few epochs
//    • Extrapolate final performance from early learning curve
//    • Stop poor performers early
//
// 3. Weight Sharing / One-Shot NAS:
//    • Single "super-network" contains all candidate architectures
//    • Share weights between architectures
//    • Evaluate by sampling from super-network
//    • Train super-network once, evaluate many architectures
//
//    Example (ENAS, 2018):
//    ```
//    Super-network: Directed acyclic graph with all possible operations
//    Architecture: Path through super-network
//    Inherit weights → Fast evaluation (seconds, not hours!)
//    ```
//
// 4. Performance Predictors:
//    • Train model to predict architecture performance
//    • Input: Architecture encoding
//    • Output: Predicted accuracy
//    • Warm start: Train on small number of evaluated architectures
//    • Use predictor to filter before expensive evaluation
//
// FAMOUS NAS RESULTS:
// -------------------
//
// 1. NASNet (Google, 2017):
//    • RL-based search (800 GPU days)
//    • Cell-based search space
//    • Result: Beat human-designed architectures on ImageNet
//    • NASNet cell transferred to other tasks (detection, segmentation)
//
// 2. ENAS (Efficient NAS, 2018):
//    • Weight sharing for efficiency
//    • 1000× faster than NASNet (0.5 GPU days)
//    • Similar performance
//
// 3. DARTS (2018):
//    • Gradient-based search
//    • 4 GPU days on CIFAR-10
//    • Found competitive architectures
//    • Code simple, reproducible
//
// 4. EfficientNet (Google, 2019):
//    • Compound scaling: Depth, width, resolution
//    • NAS to find base architecture (EfficientNet-B0)
//    • Scale up using compound coefficient
//    • Result: SOTA ImageNet (84.3% top-1) with fewer params
//
// 5. Vision Transformer + NAS:
//    • AutoFormer, Evolved Transformer
//    • Search attention patterns, MLP dimensions
//    • Improved efficiency over vanilla ViT
//
// SEARCH SPACE INSIGHTS:
// ----------------------
//
// Discovered patterns (what NAS found):
//
// 1. Depthwise Separable Convolutions:
//    • NAS consistently selects over standard convolutions
//    • More efficient (fewer params, similar capacity)
//    • Human insight: Now widely used (MobileNet, EfficientNet)
//
// 2. Skip Connections:
//    • NAS re-discovers importance of skip connections
//    • Similar to human-designed ResNet
//    • Validates: Good ideas found by both humans and NAS
//
// 3. Mixed Operations:
//    • Combination of different kernel sizes (3×3, 5×5)
//    • Not obvious to human designers
//
// 4. Irregular Patterns:
//    • Not always symmetric or regular
//    • Human bias: Prefer clean, regular patterns
//    • NAS: No such bias
//
// NAS VARIATIONS:
// ---------------
//
// 1. Hardware-Aware NAS:
//    • Optimize for latency, energy, memory (not just accuracy)
//    • Search for: High accuracy + Low latency
//    • Example: MobileNetV3 (optimized for mobile devices)
//
//    Multi-objective:
//      Maximize: Accuracy
//      Minimize: Latency, energy, params
//      Result: Pareto front of architectures
//
// 2. Transferable NAS:
//    • Search on proxy task (CIFAR-10)
//    • Transfer to target task (ImageNet)
//    • Cheaper than searching directly on ImageNet
//
// 3. Once-For-All Networks:
//    • Single network supports multiple architectures
//    • Select sub-network at deployment based on constraints
//    • Train once, deploy many
//
// 4. Modular NAS:
//    • Search for modules (attention, convolution types)
//    • Compose into full architectures
//    • Interpretable, reusable
//
// CHALLENGES & LIMITATIONS:
// -------------------------
//
// 1. Computational Cost:
//    • Even with efficiency improvements, still expensive
//    • DARTS: 4 GPU days
//    • Full NAS: 100s-1000s GPU days
//    • Not accessible to most researchers/practitioners
//
// 2. Search Space Design:
//    • Good search space crucial
//    • Requires domain expertise
//    • Paradox: NAS for automation, but needs expert search space
//
// 3. Overfitting to Search Dataset:
//    • Architecture searched on validation set
//    • May overfit to that specific dataset
//    • Transfer to new datasets may disappoint
//
// 4. Reproducibility:
//    • High variance across runs
//    • Sensitive to random seed, hyperparameters
//    • Hard to compare methods fairly
//
// 5. Evaluation Noise:
//    • Proxy estimates (low fidelity) can be misleading
//    • Weight sharing biases evaluation
//    • Best on proxy ≠ best when trained from scratch
//
// PRACTICAL CONSIDERATIONS:
// -------------------------
//
// When to use NAS:
//   ✅ Have significant compute budget
//   ✅ Need SOTA performance on specific task
//   ✅ Specific constraints (latency, mobile, edge)
//   ✅ Research: Discover new architectural patterns
//
// When NOT to use NAS:
//   ❌ Limited compute (<10 GPUs)
//   ❌ Well-solved problem (use existing architectures)
//   ❌ Quick prototype needed
//   ❌ Transfer learning sufficient
//
// Practical Approach:
//   1. Start with existing architectures (ResNet, EfficientNet, ViT)
//   2. If not satisfactory, try architecture search
//   3. Use efficient methods (DARTS, weight sharing)
//   4. Search on proxy task, transfer to target
//   5. Use discovered architectures as starting point for manual refinement
//
// FUTURE DIRECTIONS:
// ------------------
//
// 1. Faster Search:
//    • Better performance predictors
//    • Zero-cost proxies (no training needed)
//    • Neural architecture ranking
//
// 2. Larger Search Spaces:
//    • Beyond CNNs: Transformers, MoE, hybrid architectures
//    • Cross-modality: Vision + language + audio
//    • Task-agnostic: Single search for multiple tasks
//
// 3. Automated ML Pipelines:
//    • Not just architecture: Data augmentation, learning rate, optimizers
//    • End-to-end AutoML
//
// 4. Neural Architecture Editing:
//    • Start from existing architecture
//    • Incrementally improve through search
//    • Interpretable modifications
//
// 5. Constitutional NAS:
//    • Incorporate human preferences and constraints
//    • Fairness, interpretability, robustness
//    • Not just accuracy
//
// KEY TAKEAWAYS:
// --------------
//
// ✓ NAS automates discovery of neural architectures
// ✓ Three components: Search space, search strategy, performance estimation
// ✓ Strategies: Random, RL, evolution, gradient-based (DARTS)
// ✓ Efficiency: Weight sharing, low fidelity estimates, predictors
// ✓ Success: NASNet, EfficientNet, ENAS, DARTS
// ✓ Discoveries: Depthwise separable convs, irregular patterns
// ✓ Future: Essential tool for architecture design, AutoML
// ✓ Democratization: Non-experts can discover competitive architectures
//
// ============================================================================

use rand::Rng;

/// Represents an operation in the search space
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Operation {
    Conv3x3,
    Conv5x5,
    MaxPool3x3,
    AvgPool3x3,
    Identity,
    Zero,
}

impl Operation {
    fn all() -> Vec<Operation> {
        vec![
            Operation::Conv3x3,
            Operation::Conv5x5,
            Operation::MaxPool3x3,
            Operation::AvgPool3x3,
            Operation::Identity,
            Operation::Zero,
        ]
    }

    fn name(&self) -> &str {
        match self {
            Operation::Conv3x3 => "Conv3x3",
            Operation::Conv5x5 => "Conv5x5",
            Operation::MaxPool3x3 => "MaxPool3x3",
            Operation::AvgPool3x3 => "AvgPool3x3",
            Operation::Identity => "Identity",
            Operation::Zero => "Zero",
        }
    }
}

/// Cell architecture (micro search space)
#[derive(Debug, Clone)]
struct CellArchitecture {
    num_nodes: usize,
    operations: Vec<(Operation, Operation)>, // Two operations per node
    connections: Vec<(usize, usize)>,        // Two input connections per node
}

impl CellArchitecture {
    fn random(num_nodes: usize) -> Self {
        let mut rng = rand::thread_rng();
        let ops = Operation::all();

        let mut operations = Vec::new();
        let mut connections = Vec::new();

        for node in 0..num_nodes {
            // Random operations
            let op1 = ops[rng.gen_range(0..ops.len())];
            let op2 = ops[rng.gen_range(0..ops.len())];
            operations.push((op1, op2));

            // Random connections to previous nodes (including inputs)
            let max_input = node + 2; // +2 for two input nodes
            let conn1 = rng.gen_range(0..max_input);
            let conn2 = rng.gen_range(0..max_input);
            connections.push((conn1, conn2));
        }

        Self {
            num_nodes,
            operations,
            connections,
        }
    }

    fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        let ops = Operation::all();

        // Randomly mutate one aspect
        match rng.gen_range(0..3) {
            0 => {
                // Mutate operation
                let node = rng.gen_range(0..self.num_nodes);
                if rng.gen_bool(0.5) {
                    self.operations[node].0 = ops[rng.gen_range(0..ops.len())];
                } else {
                    self.operations[node].1 = ops[rng.gen_range(0..ops.len())];
                }
            }
            1 => {
                // Mutate first connection
                let node = rng.gen_range(0..self.num_nodes);
                let max_input = node + 2;
                self.connections[node].0 = rng.gen_range(0..max_input);
            }
            _ => {
                // Mutate second connection
                let node = rng.gen_range(0..self.num_nodes);
                let max_input = node + 2;
                self.connections[node].1 = rng.gen_range(0..max_input);
            }
        }
    }

    fn describe(&self) -> String {
        let mut desc = format!("Cell with {} nodes:\n", self.num_nodes);
        for (i, ((op1, op2), (conn1, conn2))) in
            self.operations.iter().zip(&self.connections).enumerate()
        {
            desc.push_str(&format!(
                "  Node {}: {} from node {}, {} from node {}\n",
                i,
                op1.name(),
                conn1,
                op2.name(),
                conn2
            ));
        }
        desc
    }
}

/// Simulated performance evaluation
fn evaluate_architecture(arch: &CellArchitecture) -> f32 {
    // In reality: Train and evaluate on dataset
    // Here: Simulate with heuristic (favor certain operations)
    let mut rng = rand::thread_rng();

    let mut score: f32 = 0.5; // Base score

    // Reward convolutions
    for (op1, op2) in &arch.operations {
        if matches!(op1, Operation::Conv3x3 | Operation::Conv5x5) {
            score += 0.02;
        }
        if matches!(op2, Operation::Conv3x3 | Operation::Conv5x5) {
            score += 0.02;
        }
        // Penalize zeros (no information flow)
        if matches!(op1, Operation::Zero) {
            score -= 0.01;
        }
        if matches!(op2, Operation::Zero) {
            score -= 0.01;
        }
    }

    // Add noise to simulate evaluation variance
    score += rng.gen_range(-0.05..0.05);

    score.max(0.0).min(1.0)
}

/// Random search
fn random_search(num_trials: usize, num_nodes: usize) -> (CellArchitecture, f32) {
    let mut best_arch = CellArchitecture::random(num_nodes);
    let mut best_score = evaluate_architecture(&best_arch);

    for _ in 1..num_trials {
        let arch = CellArchitecture::random(num_nodes);
        let score = evaluate_architecture(&arch);

        if score > best_score {
            best_arch = arch;
            best_score = score;
        }
    }

    (best_arch, best_score)
}

/// Evolutionary search
fn evolutionary_search(
    population_size: usize,
    num_generations: usize,
    num_nodes: usize,
) -> (CellArchitecture, f32) {
    // Initialize population
    let mut population: Vec<CellArchitecture> = (0..population_size)
        .map(|_| CellArchitecture::random(num_nodes))
        .collect();

    let mut best_arch = population[0].clone();
    let mut best_score = 0.0;

    for gen in 0..num_generations {
        // Evaluate population
        let mut scores: Vec<(usize, f32)> = population
            .iter()
            .enumerate()
            .map(|(i, arch)| (i, evaluate_architecture(arch)))
            .collect();

        // Sort by score
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Track best
        if scores[0].1 > best_score {
            best_arch = population[scores[0].0].clone();
            best_score = scores[0].1;
        }

        if gen % 10 == 0 {
            println!("  Generation {}: Best score = {:.3}", gen, scores[0].1);
        }

        // Selection: Keep top half
        let num_parents = population_size / 2;
        let parents: Vec<CellArchitecture> = scores[..num_parents]
            .iter()
            .map(|(i, _)| population[*i].clone())
            .collect();

        // Generate offspring through mutation
        population = parents.clone();
        for _ in 0..(population_size - num_parents) {
            let mut offspring = parents[rand::thread_rng().gen_range(0..parents.len())].clone();
            offspring.mutate();
            population.push(offspring);
        }
    }

    (best_arch, best_score)
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("Neural Architecture Search (NAS)");
    println!("{}", "=".repeat(70));
    println!();

    println!("SEARCH SPACE:");
    println!("-------------");
    println!("Cell-based micro search space");
    println!("Operations: Conv3x3, Conv5x5, MaxPool, AvgPool, Identity, Zero");
    println!("Each node: 2 operations, 2 input connections");
    println!();

    let num_nodes = 4;
    let num_random_trials = 50;
    let population_size = 20;
    let num_generations = 30;

    println!("RANDOM SEARCH:");
    println!("--------------");
    println!("Trials: {}\n", num_random_trials);
    let (random_best, random_score) = random_search(num_random_trials, num_nodes);
    println!("Best architecture found:");
    println!("{}", random_best.describe());
    println!("Performance: {:.3}\n", random_score);

    println!("EVOLUTIONARY SEARCH:");
    println!("--------------------");
    println!("Population size: {}", population_size);
    println!("Generations: {}\n", num_generations);
    let (evo_best, evo_score) = evolutionary_search(population_size, num_generations, num_nodes);
    println!("\nBest architecture found:");
    println!("{}", evo_best.describe());
    println!("Performance: {:.3}\n", evo_score);

    println!("{}", "=".repeat(70));
    println!("KEY CONCEPTS DEMONSTRATED:");
    println!("{}", "=".repeat(70));
    println!("✓ Cell-based search space (micro architecture)");
    println!("✓ Random search baseline");
    println!("✓ Evolutionary search (mutation, selection)");
    println!("✓ Performance evaluation (simulated)");
    println!();
    println!("NAS STRATEGIES:");
    println!("  • Random Search (baseline, surprisingly effective)");
    println!("  • Reinforcement Learning (NASNet, controller RNN)");
    println!("  • Evolutionary Algorithms (mutation, crossover)");
    println!("  • Gradient-Based (DARTS, differentiable search)");
    println!();
    println!("FAMOUS NAS RESULTS:");
    println!("  • NASNet (Google): Beat human designs on ImageNet");
    println!("  • EfficientNet: SOTA ImageNet (84.3%) with NAS");
    println!("  • DARTS: 4 GPU days (1000× faster than NASNet)");
    println!("  • ENAS: Weight sharing for efficiency");
    println!();
    println!("APPLICATIONS:");
    println!("  • AutoML: Discover architectures for specific tasks");
    println!("  • Hardware-aware NAS: Optimize for mobile/edge devices");
    println!("  • Transferable NAS: Search on proxy, deploy on target");
    println!();
    println!("FUTURE:");
    println!("  • Essential tool for architecture design");
    println!("  • Democratizes deep learning (non-experts get SOTA)");
    println!("  • Automated ML pipelines (architecture + training + deployment)");
    println!();
}

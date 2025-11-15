// ============================================================================
// Meta-Learning: "Learning to Learn"
// ============================================================================
//
// The paradigm shift enabling models to rapidly adapt to new tasks with
// minimal examples. Core to few-shot learning, robotics, and personalization.
//
// PROBLEM SOLVED:
// ----------------
// Traditional machine learning:
//   • Train model on task A with 10,000 examples
//   • New task B arrives → Need another 10,000 examples
//   • Slow, expensive, doesn't scale to many tasks
//
// Human learning:
//   • See 1-5 examples of a new concept
//   • Immediately generalize
//   • "Learn to learn" from past experience
//
// Meta-Learning (Learning to Learn):
//   • Train on many tasks (task distribution)
//   • Learn how to learn quickly from few examples
//   • New task arrives → Adapt with 1-10 examples!
//
// Key Insight: Instead of learning one task, learn the learning process itself.
//
// CORE CONCEPTS:
// --------------
//
// 1. Task Distribution:
//    • Not training on single task
//    • Training on distribution of tasks
//    • Each task: Learn to classify new animals, new handwriting styles, etc.
//
// 2. Support Set (Training for new task):
//    • Few examples for new task (1-shot, 5-shot, etc.)
//    • Like "train set" but tiny (1-10 examples per class)
//
// 3. Query Set (Testing for new task):
//    • Test examples for new task
//    • Like "test set" for the new task
//
// 4. Meta-Training vs Meta-Testing:
//    • Meta-training: Learn on many tasks
//    • Meta-testing: Evaluate on new unseen tasks
//
// Example (Omniglot - handwritten characters):
//   Meta-train: Tasks from alphabets 1-30
//     Task 1: Classify 5 Greek characters (1 example each)
//     Task 2: Classify 5 Hebrew characters (1 example each)
//     ...
//     Task 100,000: ...
//
//   Meta-test: Tasks from alphabets 31-50
//     New Task: Classify 5 Cyrillic characters (1 example each)
//     Model adapts quickly using learned "learning strategy"!
//
// PROBLEM FORMULATION:
// --------------------
//
// N-way K-shot Classification:
//   • N classes
//   • K examples per class (support set)
//   • M query examples to classify
//
// Example: 5-way 1-shot
//   Support: 1 example each of 5 classes (5 total examples)
//   Query: Classify new examples into one of 5 classes
//
// Example: 2-way 5-shot
//   Support: 5 examples each of 2 classes (10 total examples)
//   Query: Binary classification with more support
//
// MAJOR APPROACHES:
// -----------------
//
// 1. Metric Learning:
//    • Learn embedding space where similar items are close
//    • Classify based on distance to support examples
//    • Examples: Siamese Networks, Prototypical Networks, Matching Networks
//
// 2. Model-Based (Memory):
//    • Use external memory to store task information
//    • Neural Turing Machines, Memory-Augmented Networks
//
// 3. Optimization-Based:
//    • Learn good initialization for fast adaptation
//    • MAML, Reptile
//
// ============================================================================
// APPROACH 1: PROTOTYPICAL NETWORKS
// ============================================================================
//
// Core Idea: Represent each class by prototype (mean of support examples)
//
// Algorithm:
// ```
// 1. Embed support examples: e_i = f_θ(x_i)
//
// 2. Compute class prototypes (mean per class):
//    c_k = mean({e_i | x_i belongs to class k})
//
// 3. Embed query example: e_query = f_θ(x_query)
//
// 4. Classify by nearest prototype:
//    P(y=k | x_query) ∝ exp(-distance(e_query, c_k))
// ```
//
// Example (5-way 1-shot, image classification):
// ----------------------------------------------
//
// Support Set:
//   Class "cat": 1 image → Embed → e_cat
//   Class "dog": 1 image → Embed → e_dog
//   Class "bird": 1 image → Embed → e_bird
//   Class "fish": 1 image → Embed → e_fish
//   Class "snake": 1 image → Embed → e_snake
//
// Prototypes (just the embeddings, since K=1):
//   c_cat = e_cat
//   c_dog = e_dog
//   ... (prototype = embedding when K=1)
//
// Query: New cat image
//   Embed query → e_query
//   Distances:
//     dist(e_query, c_cat)  = 0.2  ← Smallest!
//     dist(e_query, c_dog)  = 0.8
//     dist(e_query, c_bird) = 1.2
//     dist(e_query, c_fish) = 1.5
//     dist(e_query, c_snake) = 1.3
//   Prediction: "cat"
//
// Distance Metric:
//   • Euclidean: ||e_query - c_k||²
//   • Cosine: 1 - (e_query · c_k) / (||e_query|| ||c_k||)
//
// Benefits:
//   ✅ Simple, intuitive
//   ✅ Works well for 1-shot, few-shot
//   ✅ Scalable to many classes
//   ✅ No task-specific parameters (just prototypes from support set)
//
// ============================================================================
// APPROACH 2: MAML (Model-Agnostic Meta-Learning)
// ============================================================================
//
// Core Idea: Learn initialization θ such that one gradient step adapts well
//
// Intuition:
//   Instead of learning final weights, learn "pre-adapted" weights
//   that are good starting point for fast fine-tuning.
//
// Algorithm:
// ```
// Meta-training:
//   for each task T_i sampled from task distribution:
//     1. Start from meta-parameters θ
//     2. Adapt to task using support set:
//        θ'_i = θ - α ∇L_support(θ, T_i)  (one gradient step)
//     3. Evaluate adapted parameters on query set:
//        L_query(θ'_i, T_i)
//     4. Meta-update:
//        θ = θ - β ∇_θ Σ L_query(θ'_i, T_i)
//        (Update θ to make adapted θ'_i better on query sets)
// ```
//
// Key Insight: Second-order optimization!
//   • Compute gradients through the gradient step
//   • Optimize initialization to be "one step away" from good solutions
//
// Example (Sine wave regression):
// --------------------------------
//
// Task: Fit different sine waves with different amplitudes and phases
//
// Task 1: y = 2 sin(x + 0.5)
//   Support: 5 points → Fine-tune θ → θ'_1
//   Query: Evaluate θ'_1 on new points
//
// Task 2: y = 0.5 sin(x - 1.0)
//   Support: 5 points → Fine-tune θ → θ'_2
//   Query: Evaluate θ'_2 on new points
//
// MAML learns θ such that:
//   • One gradient step on Task 1 → θ'_1 fits Task 1 well
//   • One gradient step on Task 2 → θ'_2 fits Task 2 well
//   • θ is in "good neighborhood" for all tasks
//
// Visualization (2D weight space):
// ```
//        Task 1 optimum ★
//       /
//      /
//   θ  ◉  ← Meta-learned initialization
//      \
//       \
//        Task 2 optimum ★
//
// θ positioned so one step reaches any task optimum!
// ```
//
// Benefits:
//   ✅ Model-agnostic (works with any gradient-based model)
//   ✅ Fast adaptation (1-5 gradient steps)
//   ✅ Strong theoretical foundation
//
// Challenges:
//   ❌ Expensive: Second-order gradients
//   ❌ Memory: Need to store computation graph
//   ❌ Requires careful hyperparameter tuning (α, β)
//
// First-Order MAML (FOMAML):
//   • Ignore second-order gradients
//   • Treat θ'_i as constant when computing ∇_θ
//   • Much faster, similar performance
//
// ============================================================================
// APPROACH 3: MATCHING NETWORKS
// ============================================================================
//
// Core Idea: Attention over support set for classification
//
// Algorithm:
// ```
// 1. Embed support set: {e_1, ..., e_k}
// 2. Embed query: e_query
// 3. Compute attention weights:
//    a_i = softmax(cosine_similarity(e_query, e_i))
// 4. Predict using weighted combination:
//    P(y | x_query, S) = Σ a_i × y_i
// ```
//
// Example (3-way 2-shot):
// -----------------------
//
// Support:
//   Class A: [e_1, e_2] (2 examples)
//   Class B: [e_3, e_4]
//   Class C: [e_5, e_6]
//
// Query: New example
//   Embed → e_query
//   Attention: [0.05, 0.40, 0.30, 0.10, 0.10, 0.05]
//                ↑     ↑      ↑     ↑
//              A₁    A₂     B₁    B₂   (strong on A₂, B₁)
//
//   Weighted vote:
//     Class A: 0.05 + 0.40 = 0.45
//     Class B: 0.30 + 0.10 = 0.40
//     Class C: 0.10 + 0.05 = 0.15
//   Prediction: Class A
//
// Benefits:
//   ✅ Differentiable nearest neighbor
//   ✅ Handles variable support set sizes
//   ✅ Interpretable attention weights
//
// ============================================================================
// FEW-SHOT LEARNING BENCHMARKS
// ============================================================================
//
// 1. Omniglot:
//    • 1,623 characters from 50 alphabets
//    • 20 examples per character (drawn by different people)
//    • Standard: 20-way 1-shot, 20-way 5-shot
//    • "MNIST of few-shot learning"
//
// 2. Mini-ImageNet:
//    • 100 classes, 600 images per class
//    • Subset of ImageNet
//    • Standard: 5-way 1-shot, 5-way 5-shot
//    • More challenging than Omniglot
//
// 3. Tiered-ImageNet:
//    • 608 classes, hierarchical structure
//    • Harder than Mini-ImageNet
//    • Tests generalization to different class granularities
//
// 4. Meta-Dataset:
//    • Multiple datasets combined
//    • Tests cross-dataset generalization
//    • Very challenging
//
// Typical Performance (5-way 1-shot accuracy):
//   • Random guess: 20%
//   • Baseline (nearest neighbor): 40-50%
//   • Prototypical Networks: 60-70%
//   • MAML: 60-70%
//   • State-of-the-art (2024): 75-85%
//
// ============================================================================
// APPLICATIONS
// ============================================================================
//
// 1. Drug Discovery:
//    • Predict properties of new molecules
//    • Few examples of effective compounds
//    • Meta-learn from similar drug families
//
// 2. Robotics:
//    • Adapt to new environments/objects quickly
//    • 10-20 demonstrations → Master new task
//    • Faster than reinforcement learning from scratch
//
// 3. Personalization:
//    • Recommendation systems
//    • Adapt to new users with few ratings
//    • Cold-start problem
//
// 4. Medical Diagnosis:
//    • Rare diseases (few training examples)
//    • New diseases (COVID-19)
//    • Transfer from common to rare conditions
//
// 5. Language Learning:
//    • Low-resource languages
//    • Few parallel sentences
//    • Transfer from high-resource languages
//
// 6. Character Recognition:
//    • New alphabets/scripts
//    • 1-5 examples per character
//    • Original motivation (Omniglot)
//
// ============================================================================
// TRANSDUCTIVE vs INDUCTIVE
// ============================================================================
//
// Inductive (Standard):
//   • Adapt using only support set
//   • Classify query examples one by one
//   • Query set not used for adaptation
//
// Transductive:
//   • Use both support AND query sets (unlabeled)
//   • Semi-supervised adaptation
//   • Better performance (+5-10% accuracy)
//   • Example: Use query examples to refine prototypes
//
// ============================================================================
// META-LEARNING vs TRANSFER LEARNING
// ============================================================================
//
// Transfer Learning:
//   • Train on task A (ImageNet)
//   • Fine-tune on task B (medical images)
//   • Single task transfer
//   • Needs 100s-1000s examples for task B
//
// Meta-Learning:
//   • Train on tasks A₁, A₂, ..., Aₙ
//   • Adapt to task B with 1-10 examples
//   • Multi-task learning + fast adaptation
//   • Learns "how to learn" rather than specific features
//
// Complementary:
//   • Can combine: Pre-train with transfer learning,
//     then meta-train for few-shot adaptation
//
// ============================================================================
// CHALLENGES & LIMITATIONS
// ============================================================================
//
// 1. Task Distribution:
//    • Meta-test tasks must be similar to meta-train tasks
//    • If too different, meta-learning fails
//    • Example: Meta-train on animals, test on vehicles → Poor
//
// 2. Compute Cost:
//    • MAML: Second-order gradients expensive
//    • Need to train on many tasks (thousands)
//    • Longer meta-training than standard training
//
// 3. Hyperparameters:
//    • Sensitive to learning rates (inner α, outer β)
//    • Number of adaptation steps
//    • Task sampling strategy
//
// 4. Evaluation:
//    • High variance across different task samples
//    • Need many test tasks for reliable evaluation
//    • Confidence intervals important
//
// 5. Overfitting to Meta-Train Tasks:
//    • Can memorize rather than learn to learn
//    • Need diverse task distribution
//
// ============================================================================
// MODERN DEVELOPMENTS (2024)
// ============================================================================
//
// 1. In-Context Learning (GPT-3):
//    • Large language models as meta-learners
//    • "Few-shot prompting" = meta-learning
//    • No gradient updates, just conditioning on examples
//
// 2. Task-Adaptive Pre-training:
//    • Combine transfer learning + meta-learning
//    • Pre-train on large corpus, meta-train on task distribution
//
// 3. Meta-Learning for RL:
//    • Quickly adapt to new environments
//    • Model-based + meta-learning
//
// 4. Cross-Domain Meta-Learning:
//    • Meta-train on diverse domains
//    • Generalize to very different domains
//
// 5. Continual Meta-Learning:
//    • Continuously update meta-knowledge
//    • Don't forget old tasks while learning new ones
//
// KEY TAKEAWAYS:
// --------------
//
// ✓ Meta-learning: "Learning to learn" from task distribution
// ✓ Few-shot learning: Adapt to new tasks with 1-10 examples
// ✓ Main approaches: Metric learning, Optimization-based, Model-based
// ✓ Prototypical Networks: Simple, effective metric learning
// ✓ MAML: Learn initialization for fast fine-tuning
// ✓ Applications: Drug discovery, robotics, personalization, rare diseases
// ✓ Different from transfer learning: Multi-task + fast adaptation
// ✓ Modern: In-context learning in LLMs is form of meta-learning
//
// ============================================================================

use ndarray::Array1;
use rand::Rng;

/// Embedding network
struct EmbeddingNetwork {
    embed_dim: usize,
}

impl EmbeddingNetwork {
    fn new(embed_dim: usize) -> Self {
        Self { embed_dim }
    }

    /// Embed input into latent space
    fn embed(&self, input: &Array1<f32>) -> Array1<f32> {
        // Simulate embedding (in reality: neural network)
        // For demo: Simple transformation + normalization
        let mut embedding = input.mapv(|x| x.tanh());

        // Normalize to unit length
        let norm = embedding.iter().map(|x| x*x).sum::<f32>().sqrt();
        if norm > 0.0 {
            embedding = embedding / norm;
        }

        embedding
    }
}

/// Prototypical Networks implementation
struct PrototypicalNetwork {
    encoder: EmbeddingNetwork,
}

impl PrototypicalNetwork {
    fn new(embed_dim: usize) -> Self {
        println!("Prototypical Network:");
        println!("  Embedding dimension: {}", embed_dim);
        println!("  Classification: Nearest prototype\n");

        Self {
            encoder: EmbeddingNetwork::new(embed_dim),
        }
    }

    /// Compute class prototypes from support set
    fn compute_prototypes(
        &self,
        support_examples: &[Array1<f32>],
        support_labels: &[usize],
        num_classes: usize,
    ) -> Vec<Array1<f32>> {
        let mut prototypes = vec![Array1::zeros(self.encoder.embed_dim); num_classes];
        let mut counts = vec![0; num_classes];

        // Embed and accumulate
        for (example, &label) in support_examples.iter().zip(support_labels) {
            let embedding = self.encoder.embed(example);
            prototypes[label] = &prototypes[label] + &embedding;
            counts[label] += 1;
        }

        // Average to get prototypes
        for (prototype, count) in prototypes.iter_mut().zip(&counts) {
            if *count > 0 {
                *prototype = &*prototype / (*count as f32);
            }
        }

        prototypes
    }

    /// Classify query example using prototypes
    fn classify(
        &self,
        query: &Array1<f32>,
        prototypes: &[Array1<f32>],
    ) -> (usize, Vec<f32>) {
        let query_embedding = self.encoder.embed(query);

        // Compute distances to each prototype
        let distances: Vec<f32> = prototypes
            .iter()
            .map(|proto| {
                // Euclidean distance
                query_embedding.iter()
                    .zip(proto.iter())
                    .map(|(q, p)| (q - p).powi(2))
                    .sum::<f32>()
                    .sqrt()
            })
            .collect();

        // Convert to probabilities (softmax of negative distances)
        let neg_distances: Vec<f32> = distances.iter().map(|d| -d).collect();
        let max_neg_d = neg_distances.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_neg_d: Vec<f32> = neg_distances.iter().map(|d| (d - max_neg_d).exp()).collect();
        let sum_exp: f32 = exp_neg_d.iter().sum();
        let probabilities: Vec<f32> = exp_neg_d.iter().map(|e| e / sum_exp).collect();

        // Predicted class (argmin distance)
        let predicted_class = distances
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        (predicted_class, probabilities)
    }
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("Meta-Learning: Learning to Learn");
    println!("{}", "=".repeat(70));
    println!();

    // Configuration: 5-way 1-shot classification
    let num_classes = 5;
    let shots_per_class = 1;
    let num_queries = 10;
    let input_dim = 64;
    let embed_dim = 32;

    println!("TASK CONFIGURATION:");
    println!("-------------------");
    println!("{}-way {}-shot classification", num_classes, shots_per_class);
    println!("Support set: {} examples ({} per class)", num_classes * shots_per_class, shots_per_class);
    println!("Query set: {} examples\n", num_queries);

    println!("MODEL:");
    println!("------");
    let model = PrototypicalNetwork::new(embed_dim);

    // Generate synthetic support set
    println!("GENERATING SUPPORT SET:");
    println!("-----------------------");
    let mut rng = rand::thread_rng();
    let mut support_examples = Vec::new();
    let mut support_labels = Vec::new();

    for class_id in 0..num_classes {
        for _ in 0..shots_per_class {
            let example: Array1<f32> = Array1::from_vec(
                (0..input_dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
            );
            support_examples.push(example);
            support_labels.push(class_id);
        }
        println!("Class {}: {} example(s)", class_id, shots_per_class);
    }

    // Compute prototypes
    println!("\nCOMPUTING PROTOTYPES:");
    println!("---------------------");
    let prototypes = model.compute_prototypes(&support_examples, &support_labels, num_classes);
    println!("Computed {} prototypes (one per class)\n", prototypes.len());

    // Generate and classify queries
    println!("CLASSIFYING QUERY EXAMPLES:");
    println!("---------------------------");
    let mut correct = 0;

    for i in 0..num_queries {
        // Generate query (with known label for evaluation)
        let true_label = rng.gen_range(0..num_classes);
        let query: Array1<f32> = Array1::from_vec(
            (0..input_dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
        );

        // Classify
        let (predicted, probabilities) = model.classify(&query, &prototypes);

        if predicted == true_label {
            correct += 1;
        }

        if i < 5 {
            println!("Query {}: True={}, Predicted={}, Probs={:?}",
                     i + 1, true_label, predicted,
                     probabilities.iter().map(|p| format!("{:.2}", p)).collect::<Vec<_>>());
        }
    }

    let accuracy = (correct as f32 / num_queries as f32) * 100.0;
    println!("\n... ({} total queries)\n", num_queries);
    println!("RESULTS:");
    println!("--------");
    println!("Accuracy: {}/{} ({:.1}%)", correct, num_queries, accuracy);
    println!("Baseline (random): {:.1}%\n", 100.0 / num_classes as f32);

    println!("{}", "=".repeat(70));
    println!("KEY CONCEPTS DEMONSTRATED:");
    println!("{}", "=".repeat(70));
    println!("✓ Few-shot learning with minimal examples");
    println!("✓ Prototypical Networks: Classify by nearest prototype");
    println!("✓ Embedding space where similar items are close");
    println!("✓ {}-way {}-shot: {} support examples, classify new examples",
             num_classes, shots_per_class, num_classes * shots_per_class);
    println!();
    println!("META-LEARNING APPROACHES:");
    println!("  • Prototypical Networks (metric learning)");
    println!("  • MAML (optimization-based)");
    println!("  • Matching Networks (attention-based)");
    println!();
    println!("APPLICATIONS:");
    println!("  • Drug discovery (few examples of effective compounds)");
    println!("  • Medical diagnosis (rare diseases)");
    println!("  • Robotics (fast task adaptation)");
    println!("  • Personalization (cold-start users)");
    println!("  • Low-resource language translation");
    println!();
    println!("KEY INSIGHT:");
    println!("  Instead of learning one task, learn HOW to learn quickly!");
    println!();
}

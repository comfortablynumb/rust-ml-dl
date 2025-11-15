// ============================================================================
// Mixture of Experts (MoE)
// ============================================================================
//
// The sparse activation architecture enabling trillion-parameter models
// Used in GPT-4, Switch Transformer, GLaM, and other modern large models.
//
// PROBLEM SOLVED:
// ----------------
// Traditional dense models:
//   • Every parameter activated for every input
//   • Scaling: 10B params → 10B operations per token
//   • Expensive: More params = More compute per token
//
// Mixture of Experts (MoE):
//   • Different "experts" specialize in different patterns
//   • Route each input to subset of experts (sparse activation)
//   • Scaling: 100B params, but only 10B activated per token
//   • 10× bigger model with same compute!
//
// Key Insight: Not all parameters need to process all inputs.
//
// CORE CONCEPT:
// -------------
//
// Traditional Layer:
//   output = FeedForward(input)  // All parameters used
//
// MoE Layer:
//   experts = [Expert_1, Expert_2, ..., Expert_N]
//   routing = Gating(input)  // Which experts to use?
//   output = Σ routing[i] × Expert_i(input)  // Weighted combination
//
// Example (4 experts, top-2 routing):
//   Input: "The cat"
//   Routing: [0.7, 0.3, 0.0, 0.0]  // Use experts 1 and 2
//   Output: 0.7 × Expert_1(input) + 0.3 × Expert_2(input)
//
//   Expert 3 and 4: Not activated → No compute!
//
// ARCHITECTURE:
// -------------
//
// Standard Transformer Layer:
// ```
// x' = x + SelfAttention(x)
// y  = x' + FeedForward(x')
// ```
//
// MoE Transformer Layer:
// ```
// x' = x + SelfAttention(x)      // Dense (all params)
// y  = x' + MoE_FeedForward(x')  // Sparse (subset of params)
// ```
//
// Typical Configuration:
//   • Self-attention: Dense (all tokens interact)
//   • Feed-forward: Sparse (MoE, only k experts activated)
//   • Ratio: Replace every 2nd or 4th FFN with MoE
//
// MoE FEED-FORWARD LAYER:
// -----------------------
//
// Components:
//   1. Router/Gate: Decides which experts to use
//   2. Experts: Specialized feed-forward networks
//   3. Combination: Weighted sum of expert outputs
//
// ```python
// # Router: Input → Expert weights
// router_logits = W_gate @ x  # [num_experts]
// router_weights = softmax(router_logits)
//
// # Top-k selection (e.g., k=2)
// top_k_weights, top_k_indices = top_k(router_weights, k=2)
// top_k_weights = top_k_weights / sum(top_k_weights)  # Renormalize
//
// # Expert computation (only top-k)
// output = 0
// for i, weight in zip(top_k_indices, top_k_weights):
//     output += weight * Expert_i(x)
// ```
//
// ROUTING STRATEGIES:
// -------------------
//
// 1. Top-1 Routing (Switch Transformer):
//    • Select single expert with highest weight
//    • Most sparse: 1/N experts activated
//    • Fastest, but less capacity
//
//    Example (8 experts):
//      Routing: [0.1, 0.6, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02]
//      Selected: Expert 2 (weight=1.0)
//
// 2. Top-2 Routing:
//    • Select 2 experts with highest weights
//    • Balance: Capacity vs efficiency
//    • Used in many production models
//
//    Example (8 experts):
//      Routing: [0.1, 0.6, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02]
//      Selected: Expert 2 (0.6/0.7=0.857), Expert 1 (0.1/0.7=0.143)
//
// 3. Top-k Routing (k > 2):
//    • More experts for complex inputs
//    • Better quality, more compute
//    • Typical k=2-4
//
// 4. Soft Routing (Dense):
//    • Use all experts with their weights
//    • No sparsity, defeats purpose
//    • Only for very small number of experts
//
// EXPERT SPECIALIZATION:
// ----------------------
//
// Experts naturally specialize during training!
//
// Example discoveries (language models):
//   • Expert 1: Punctuation and grammar
//   • Expert 2: Named entities (people, places)
//   • Expert 3: Numbers and dates
//   • Expert 4: Technical/scientific terms
//   • Expert 5: Common words and phrases
//   • Expert 6: Rare/uncommon words
//
// Why?
//   • Router learns to send similar inputs to same expert
//   • Expert becomes specialized through repeated exposure
//   • Emergent behavior, not explicitly programmed!
//
// Visualization (Token → Expert):
//   "John" → Expert 2 (names)
//   "runs" → Expert 5 (common verbs)
//   "42" → Expert 3 (numbers)
//   "." → Expert 1 (punctuation)
//
// LOAD BALANCING:
// ---------------
//
// Problem: Without constraints, routing can collapse
//   • All tokens routed to Expert 1
//   • Experts 2-N never used (waste!)
//   • No benefit from having multiple experts
//
// Solutions:
//
// 1. Auxiliary Load Balance Loss:
//    L_balance = α × Σ f_i × P_i
//
//    f_i = Fraction of tokens routed to expert i
//    P_i = Fraction of routing probability to expert i
//
//    Minimizing encourages uniform distribution
//    Typical α = 0.01
//
// 2. Expert Capacity:
//    • Limit tokens per expert
//    • Capacity = (tokens_per_batch / num_experts) × capacity_factor
//    • If expert full, token dropped or sent to overflow
//    • Prevents overloading popular experts
//
// 3. Random Routing (with probability p):
//    • With probability p, route to random expert
//    • Ensures all experts get training signal
//    • Typical p = 0.1 during training
//
// 4. Differentiable Top-k:
//    • Smooth approximation to top-k
//    • Gradients flow to all experts
//    • Better optimization
//
// TRAINING CHALLENGES:
// --------------------
//
// 1. Expert Collapse:
//    Problem: Router sends everything to one expert
//    Solution: Load balance loss, capacity limits
//
// 2. Expert Imbalance:
//    Problem: Some experts process 90% tokens, others 1%
//    Solution: Auxiliary loss encourages balance
//
// 3. Communication Cost:
//    Problem: Experts on different GPUs/machines
//    Solution: All-to-All communication, expert parallelism
//
// 4. Gradient Noise:
//    Problem: Each expert sees different data distribution
//    Solution: Higher learning rate, more updates
//
// 5. Numerical Instability:
//    Problem: Routing probabilities can spike
//    Solution: Careful initialization, gradient clipping
//
// SCALING LAWS:
// -------------
//
// MoE enables unprecedented scale:
//
// Dense Model:
//   • GPT-3: 175B params, 175B FLOPs per token
//   • Training: $5-10M on cloud
//
// MoE Model (8 experts, top-2):
//   • Switch-XXL: 1.6T params, 200B FLOPs per token
//   • Training: ~$30M, but 8× more parameters!
//   • Better performance with similar compute
//
// Key Insight: Params ≠ Compute
//   • Dense: N params = N compute
//   • MoE: N params ≠ N compute (sparse activation)
//   • Can have massive models without massive compute
//
// FAMOUS MOE MODELS:
// ------------------
//
// 1. Switch Transformer (Google, 2021):
//    • 1.6 trillion parameters
//    • Top-1 routing (simplest)
//    • 7× speedup over T5-XXL (same quality)
//
// 2. GLaM (Google, 2021):
//    • 1.2 trillion parameters
//    • Top-2 routing
//    • Beats GPT-3 with 1/3 energy cost
//
// 3. GPT-4 (OpenAI, 2023):
//    • Rumored to use MoE (8 experts, ~1.8T params)
//    • Not officially confirmed
//    • Explains high quality with reasonable latency
//
// 4. Mixtral 8x7B (Mistral AI, 2023):
//    • Open-source MoE
//    • 47B total params, 13B active per token
//    • Beats GPT-3.5 on many benchmarks
//
// 5. DeepSeek-MoE (2024):
//    • 16B params, 2B active
//    • Fine-grained expert splitting
//    • SOTA efficiency
//
// IMPLEMENTATION VARIANTS:
// ------------------------
//
// 1. Expert Granularity:
//
//    Coarse-grained: Few large experts
//      • 8-64 experts
//      • Each expert = full FFN (e.g., 4096 → 16384 → 4096)
//      • Used in: Switch, GPT-4
//
//    Fine-grained: Many small experts
//      • 100-1000 experts
//      • Each expert = small network (256 → 512 → 256)
//      • Used in: BASE layers, DeepSeek-MoE
//
// 2. Hierarchical MoE:
//    • Router 1: Choose expert group
//    • Router 2: Choose expert within group
//    • Reduces routing complexity
//
// 3. Shared vs Separate Experts:
//
//    Separate: Each layer has own experts
//      • More capacity
//      • More parameters
//
//    Shared: All layers share same experts
//      • Fewer parameters
//      • Experts see more diverse data
//
// MOE vs DENSE MODELS:
// --------------------
//
// Dense Models:
//   ✅ Simpler architecture
//   ✅ Easier to train (no load balancing)
//   ✅ Better hardware utilization
//   ✅ Easier deployment (single model)
//   ❌ Linear scaling: N params = N compute
//   ❌ Expensive to scale beyond 100B params
//
// MoE Models:
//   ✅ Sublinear scaling: N params < N compute
//   ✅ Can reach trillion-scale params
//   ✅ Better sample efficiency (more capacity)
//   ❌ Training complexity (load balancing, routing)
//   ❌ Deployment complexity (larger model size)
//   ❌ Communication overhead (distributed experts)
//
// WHEN TO USE MOE:
// ----------------
//
// Use MoE when:
//   ✅ Need very large models (> 100B params)
//   ✅ Have multi-GPU/multi-node setup
//   ✅ Compute budget limited
//   ✅ Want state-of-the-art performance
//   ✅ Diverse data (benefits from specialization)
//
// Use Dense when:
//   ✅ Model size < 10B params
//   ✅ Single GPU deployment
//   ✅ Simpler training pipeline desired
//   ✅ Lower latency critical (no routing overhead)
//   ✅ Uniform data distribution
//
// HARDWARE CONSIDERATIONS:
// ------------------------
//
// Memory:
//   • Total params: All experts in GPU memory
//   • Active params: Subset used per token
//   • MoE needs more memory than dense of same compute
//
// Compute:
//   • Sparse activation → Lower FLOPs
//   • But: Routing overhead, load imbalance
//   • GPU utilization: 60-80% (vs 90%+ for dense)
//
// Communication:
//   • All-to-All between GPUs (expert parallelism)
//   • Bottleneck for small batches
//   • Requires high-bandwidth interconnect (NVLink, IB)
//
// PRACTICAL TIPS:
// ---------------
//
// 1. Start Small:
//    • 4-8 experts initially
//    • Verify load balancing works
//    • Scale up gradually
//
// 2. Monitor Expert Usage:
//    • Log tokens per expert
//    • Detect collapse early
//    • Adjust balance loss if needed
//
// 3. Capacity Factor:
//    • Start with 1.25-1.5×
//    • Higher = less dropping, more compute
//    • Lower = more dropping, less compute
//
// 4. Auxiliary Loss Weight:
//    • Start with α=0.01
//    • Too high: Hurts quality (forces bad routing)
//    • Too low: Experts collapse
//
// 5. Expert Initialization:
//    • Initialize experts differently
//    • Encourages diversity from start
//    • Random seeds per expert
//
// FUTURE DIRECTIONS:
// ------------------
//
// 1. Conditional Computation:
//    • Beyond FFN: Sparse attention, sparse layers
//    • Entire model sparse
//
// 2. Learned Routing:
//    • More sophisticated routers
//    • Hierarchical, content-based
//
// 3. Dynamic Experts:
//    • Create/remove experts during training
//    • Adapt to data distribution
//
// 4. Multi-modal MoE:
//    • Different experts for text, images, audio
//    • Routing based on modality
//
// 5. Edge Deployment:
//    • Small MoE for mobile
//    • 4-8 experts, on-device routing
//
// HISTORICAL CONTEXT:
// -------------------
//
// 1991: Jacobs et al. - Original MoE concept
// 2017: Shazeer et al. - Outrageously Large Neural Networks
// 2021: Switch Transformer - Simplified to top-1 routing
// 2021: GLaM - 1.2T params, beats GPT-3
// 2022-23: GPT-4 (rumored MoE architecture)
// 2023: Mixtral 8x7B - Open-source MoE
// 2024: DeepSeek-MoE - Fine-grained experts
//
// KEY TAKEAWAYS:
// --------------
//
// ✓ MoE enables sparse activation: Not all params for all inputs
// ✓ Routing mechanism decides which experts to use
// ✓ Enables trillion-parameter models with reasonable compute
// ✓ Experts naturally specialize during training
// ✓ Load balancing critical to prevent expert collapse
// ✓ Used in GPT-4, Switch Transformer, Mixtral, and other SOTA models
// ✓ Trade-off: Model size vs deployment complexity
// ✓ Future: Likely standard for large-scale models
//
// ============================================================================

use ndarray::Array1;
use rand::Rng;
use std::collections::HashMap;

/// Single expert (feed-forward network)
struct Expert {
    id: usize,
    hidden_size: usize,
    processed_count: usize,
}

impl Expert {
    fn new(id: usize, hidden_size: usize) -> Self {
        Self {
            id,
            hidden_size,
            processed_count: 0,
        }
    }

    /// Process input through expert
    fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        self.processed_count += 1;
        // Simulate expert computation (in reality: 2-layer FFN)
        // FFN: x → W1 → ReLU → W2
        input.mapv(|x| (x * 2.0).tanh()) // Simplified transformation
    }
}

/// Router/Gating network
struct Router {
    num_experts: usize,
    top_k: usize,
}

impl Router {
    fn new(num_experts: usize, top_k: usize) -> Self {
        Self { num_experts, top_k }
    }

    /// Compute routing weights and select top-k experts
    fn route(&self, _input: &Array1<f32>) -> (Vec<usize>, Vec<f32>) {
        // Simulate routing logits (in reality: learned W_gate @ input)
        let mut rng = rand::thread_rng();
        let logits: Vec<f32> = (0..self.num_experts)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        // Softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probabilities: Vec<f32> = exp_logits.iter().map(|e| e / sum_exp).collect();

        // Select top-k
        let mut indexed_probs: Vec<(usize, f32)> = probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_k_indices: Vec<usize> = indexed_probs[..self.top_k].iter().map(|(i, _)| *i).collect();
        let top_k_probs: Vec<f32> = indexed_probs[..self.top_k].iter().map(|(_, p)| *p).collect();

        // Renormalize top-k probabilities
        let sum_top_k: f32 = top_k_probs.iter().sum();
        let normalized_probs: Vec<f32> = top_k_probs.iter().map(|p| p / sum_top_k).collect();

        (top_k_indices, normalized_probs)
    }
}

/// Mixture of Experts Layer
struct MoELayer {
    experts: Vec<Expert>,
    router: Router,
}

impl MoELayer {
    fn new(num_experts: usize, hidden_size: usize, top_k: usize) -> Self {
        let experts = (0..num_experts)
            .map(|id| Expert::new(id, hidden_size))
            .collect();

        let router = Router::new(num_experts, top_k);

        println!("MoE Layer Configuration:");
        println!("  Number of experts: {}", num_experts);
        println!("  Hidden size: {}", hidden_size);
        println!("  Top-k routing: {}", top_k);
        println!("  Sparsity: {}/{} experts activated ({:.1}%)\n",
                 top_k, num_experts, (top_k as f32 / num_experts as f32) * 100.0);

        Self { experts, router }
    }

    /// Forward pass through MoE layer
    fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        // Route to top-k experts
        let (expert_indices, weights) = self.router.route(input);

        // Combine expert outputs
        let mut output = Array1::zeros(input.len());
        for (&expert_idx, &weight) in expert_indices.iter().zip(&weights) {
            let expert_output = self.experts[expert_idx].forward(input);
            output = output + expert_output * weight;
        }

        output
    }

    /// Get expert usage statistics
    fn get_expert_stats(&self) -> HashMap<usize, usize> {
        self.experts
            .iter()
            .map(|e| (e.id, e.processed_count))
            .collect()
    }
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("Mixture of Experts (MoE)");
    println!("{}", "=".repeat(70));
    println!();

    // Configuration
    let num_experts = 8;
    let hidden_size = 512;
    let top_k = 2; // Top-2 routing

    println!("ARCHITECTURE:");
    println!("-------------");
    let mut moe = MoELayer::new(num_experts, hidden_size, top_k);

    println!("PROCESSING TOKENS:");
    println!("------------------");
    let num_tokens = 100;

    for i in 0..num_tokens {
        // Simulate input token
        let mut rng = rand::thread_rng();
        let input: Array1<f32> = Array1::from_vec(
            (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect()
        );

        // Process through MoE
        let _output = moe.forward(&input);

        if i < 5 {
            println!("Token {}: Processed", i + 1);
        }
    }

    println!("\n... (processed {} tokens total)\n", num_tokens);

    println!("EXPERT USAGE STATISTICS:");
    println!("------------------------");
    let stats = moe.get_expert_stats();
    let total_calls: usize = stats.values().sum();

    println!("Total expert calls: {} (avg {:.1} per token)\n",
             total_calls, total_calls as f32 / num_tokens as f32);

    for i in 0..num_experts {
        let count = stats.get(&i).unwrap_or(&0);
        let percentage = (*count as f32 / total_calls as f32) * 100.0;
        let bar_length = (percentage / 2.0) as usize;
        println!("Expert {}: {:>3} calls ({:>5.1}%) {}",
                 i, count, percentage, "█".repeat(bar_length));
    }

    println!("\n{}", "=".repeat(70));
    println!("KEY CONCEPTS DEMONSTRATED:");
    println!("{}", "=".repeat(70));
    println!("✓ Router selects top-k experts for each input");
    println!("✓ Sparse activation: Only {}/{} experts active per token", top_k, num_experts);
    println!("✓ Weighted combination of expert outputs");
    println!("✓ Load balancing: All experts should get similar usage");
    println!();
    println!("SCALING BENEFITS:");
    println!("  • Dense 100B model: 100B FLOPs per token");
    println!("  • MoE 400B (8 experts, top-2): ~100B FLOPs per token");
    println!("  • Result: 4× more parameters, same compute!");
    println!();
    println!("FAMOUS MOE MODELS:");
    println!("  • Switch Transformer (Google): 1.6T parameters");
    println!("  • GLaM (Google): 1.2T parameters, beats GPT-3");
    println!("  • GPT-4 (OpenAI): Rumored 8 experts, ~1.8T total params");
    println!("  • Mixtral 8x7B (Mistral): 47B params, 13B active");
    println!();
    println!("APPLICATIONS:");
    println!("  • Large language models (GPT-4, PaLM-MoE)");
    println!("  • Efficient scaling to trillion-parameter models");
    println!("  • Multi-task learning (experts specialize per task)");
    println!();
}

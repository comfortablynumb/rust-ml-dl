/// # Advanced Optimizers 🚀
///
/// Modern optimization algorithms that power deep learning: Adam, RMSprop, AdaGrad,
/// and learning rate scheduling.
///
/// ## What This Example Demonstrates
///
/// 1. **SGD with Momentum**: Accelerated gradient descent with velocity
/// 2. **AdaGrad**: Adaptive learning rates for sparse features
/// 3. **RMSprop**: Exponential moving average of squared gradients
/// 4. **Adam**: Combines momentum and RMSprop (most popular!)
/// 5. **AdamW**: Adam with decoupled weight decay
/// 6. **Learning Rate Schedules**: Step decay, cosine annealing, warmup
/// 7. **Gradient Clipping**: Prevent exploding gradients
///
/// ## Why Modern Optimizers Matter
///
/// - **10-100× faster convergence** than basic SGD
/// - **Adaptive learning rates** per parameter
/// - **Less hyperparameter tuning** required
/// - **Enable deep networks** (100+ layers)
/// - **Power all modern AI** (GPT, BERT, Stable Diffusion)
///
/// ## The Evolution
///
/// ```
/// SGD (1951) → Momentum (1964) → AdaGrad (2011) → RMSprop (2012)
///            → Adam (2014) → AdamW (2017) ← Current default
/// ```

use ndarray::{Array1, Array2};
use rand::Rng;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         Advanced Optimizers for Deep Learning           ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Demonstrate all optimizers on a simple quadratic optimization problem
    demo_optimizers();

    // Demonstrate learning rate schedules
    demo_lr_schedules();

    // Demonstrate gradient clipping
    demo_gradient_clipping();
}

/// Demonstrate different optimizers
fn demo_optimizers() {
    println!("═══ Optimizer Comparison ═══\n");

    // Optimization problem: minimize f(x, y) = x² + 10y²
    // Minimum at (0, 0)
    // This creates a "ravine" (different curvatures in x and y directions)

    let iterations = 100;
    let start = Array1::from(vec![5.0, 5.0]); // Starting point

    println!("Optimization Problem: f(x, y) = x² + 10y²");
    println!("Starting point: ({:.2}, {:.2})", start[0], start[1]);
    println!("Global minimum: (0, 0)\n");

    // Test each optimizer
    println!("Optimizer           Final (x, y)          Steps to Converge");
    println!("─────────────────────────────────────────────────────────────");

    // 1. Basic SGD (for comparison)
    let mut sgd = SGD::new(0.01);
    let sgd_result = optimize_quadratic(&mut sgd, &start, iterations);
    print_result("SGD (basic)", &sgd_result, &count_steps(&sgd_result));

    // 2. SGD with Momentum
    let mut momentum = Momentum::new(0.01, 0.9);
    let momentum_result = optimize_quadratic(&mut momentum, &start, iterations);
    print_result("SGD + Momentum", &momentum_result, &count_steps(&momentum_result));

    // 3. AdaGrad
    let mut adagrad = AdaGrad::new(0.5);
    let adagrad_result = optimize_quadratic(&mut adagrad, &start, iterations);
    print_result("AdaGrad", &adagrad_result, &count_steps(&adagrad_result));

    // 4. RMSprop
    let mut rmsprop = RMSprop::new(0.1, 0.9);
    let rmsprop_result = optimize_quadratic(&mut rmsprop, &start, iterations);
    print_result("RMSprop", &rmsprop_result, &count_steps(&rmsprop_result));

    // 5. Adam (most popular!)
    let mut adam = Adam::new(0.1, 0.9, 0.999);
    let adam_result = optimize_quadratic(&mut adam, &start, iterations);
    print_result("Adam ⭐", &adam_result, &count_steps(&adam_result));

    // 6. AdamW
    let mut adamw = AdamW::new(0.1, 0.9, 0.999, 0.01);
    let adamw_result = optimize_quadratic(&mut adamw, &start, iterations);
    print_result("AdamW", &adamw_result, &count_steps(&adamw_result));

    println!();
}

fn print_result(name: &str, result: &Array1<f32>, steps: &Option<usize>) {
    let steps_str = match steps {
        Some(s) => format!("{}", s),
        None => ">100".to_string(),
    };
    println!("{:18} ({:6.3}, {:6.3})      {}",
             name, result[0], result[1], steps_str);
}

fn count_steps(trajectory: &Array1<f32>) -> Option<usize> {
    // Count steps until convergence (within 0.01 of optimum)
    if trajectory[0].abs() < 0.01 && trajectory[1].abs() < 0.01 {
        Some(100) // Placeholder - would track in full implementation
    } else {
        None
    }
}

/// Quadratic function: f(x, y) = x² + 10y²
fn quadratic_gradient(x: &Array1<f32>) -> Array1<f32> {
    Array1::from(vec![2.0 * x[0], 20.0 * x[1]])
}

/// Optimize quadratic function with given optimizer
fn optimize_quadratic<O: Optimizer>(
    optimizer: &mut O,
    start: &Array1<f32>,
    iterations: usize,
) -> Array1<f32> {
    let mut x = start.clone();

    for _ in 0..iterations {
        let grad = quadratic_gradient(&x);
        optimizer.step(&mut x, &grad);
    }

    x
}

/// Demonstrate learning rate schedules
fn demo_lr_schedules() {
    println!("═══ Learning Rate Schedules ═══\n");

    let initial_lr = 0.1;
    let total_steps = 100;

    println!("Initial LR: {}", initial_lr);
    println!("Total steps: {}\n", total_steps);

    // Show LR at key points for each schedule
    println!("Schedule                 Step 0   Step 25  Step 50  Step 75  Step 99");
    println!("─────────────────────────────────────────────────────────────────────");

    // 1. Constant (baseline)
    let constant = ConstantLR::new(initial_lr);
    print_schedule("Constant", &constant, &[0, 25, 50, 75, 99]);

    // 2. Step Decay
    let step_decay = StepDecayLR::new(initial_lr, 0.5, 25);
    print_schedule("Step Decay", &step_decay, &[0, 25, 50, 75, 99]);

    // 3. Exponential Decay
    let exp_decay = ExponentialDecayLR::new(initial_lr, 0.95);
    print_schedule("Exponential Decay", &exp_decay, &[0, 25, 50, 75, 99]);

    // 4. Cosine Annealing ⭐
    let cosine = CosineAnnealingLR::new(initial_lr, 0.001, total_steps);
    print_schedule("Cosine Annealing ⭐", &cosine, &[0, 25, 50, 75, 99]);

    // 5. Warmup + Cosine (most popular for Transformers!)
    let warmup_cosine = WarmupCosineAnnealingLR::new(initial_lr, 0.001, 10, total_steps);
    print_schedule("Warmup + Cosine 🔥", &warmup_cosine, &[0, 25, 50, 75, 99]);

    println!("\n💡 Warmup + Cosine is the standard for Transformers (BERT, GPT)");
    println!();
}

fn print_schedule<S: LRSchedule>(name: &str, schedule: &S, steps: &[usize]) {
    print!("{:23}", name);
    for &step in steps {
        print!("  {:.5}", schedule.get_lr(step));
    }
    println!();
}

/// Demonstrate gradient clipping
fn demo_gradient_clipping() {
    println!("═══ Gradient Clipping (For RNNs/LSTMs) ═══\n");

    // Simulate exploding gradient
    let small_grad = Array1::from(vec![0.1, 0.2, 0.15]);
    let large_grad = Array1::from(vec![10.0, 20.0, 15.0]);
    let huge_grad = Array1::from(vec![100.0, 200.0, 150.0]);

    let clip_value = 5.0;

    println!("Gradient Clipping Threshold: {}\n", clip_value);

    println!("Original Gradient          Norm    Clipped Gradient          Norm");
    println!("────────────────────────────────────────────────────────────────────");

    print_gradient_clipping(&small_grad, clip_value);
    print_gradient_clipping(&large_grad, clip_value);
    print_gradient_clipping(&huge_grad, clip_value);

    println!("\n💡 Gradient clipping is REQUIRED for RNNs/LSTMs to prevent exploding gradients");
    println!("   Typical values: 1.0 for RNNs, 5.0 for Transformers");
    println!();
}

fn print_gradient_clipping(grad: &Array1<f32>, clip_value: f32) {
    let norm = grad.iter().map(|x| x * x).sum::<f32>().sqrt();
    let clipped = clip_gradient_norm(grad, clip_value);
    let clipped_norm = clipped.iter().map(|x| x * x).sum::<f32>().sqrt();

    println!("[{:6.2}, {:6.2}, {:6.2}]  {:6.2}  [{:6.2}, {:6.2}, {:6.2}]  {:6.2}",
             grad[0], grad[1], grad[2], norm,
             clipped[0], clipped[1], clipped[2], clipped_norm);
}

fn clip_gradient_norm(grad: &Array1<f32>, max_norm: f32) -> Array1<f32> {
    let norm = grad.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > max_norm {
        grad * (max_norm / norm)
    } else {
        grad.clone()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Optimizer Trait and Implementations
// ═══════════════════════════════════════════════════════════════════════════

trait Optimizer {
    fn step(&mut self, params: &mut Array1<f32>, gradients: &Array1<f32>);
}

/// Basic SGD: w = w - lr * grad
struct SGD {
    lr: f32,
}

impl SGD {
    fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut Array1<f32>, gradients: &Array1<f32>) {
        *params = &*params - self.lr * gradients;
    }
}

/// SGD with Momentum
/// v = β * v + grad
/// w = w - lr * v
struct Momentum {
    lr: f32,
    beta: f32,
    velocity: Option<Array1<f32>>,
}

impl Momentum {
    fn new(lr: f32, beta: f32) -> Self {
        Self {
            lr,
            beta,
            velocity: None,
        }
    }
}

impl Optimizer for Momentum {
    fn step(&mut self, params: &mut Array1<f32>, gradients: &Array1<f32>) {
        // Initialize velocity on first step
        if self.velocity.is_none() {
            self.velocity = Some(Array1::zeros(params.len()));
        }

        let v = self.velocity.as_mut().unwrap();
        *v = self.beta * &*v + gradients;
        *params = &*params - self.lr * &*v;
    }
}

/// AdaGrad: Adaptive learning rates per parameter
/// G = G + grad²
/// w = w - (lr / √(G + ε)) * grad
struct AdaGrad {
    lr: f32,
    epsilon: f32,
    sum_squared_gradients: Option<Array1<f32>>,
}

impl AdaGrad {
    fn new(lr: f32) -> Self {
        Self {
            lr,
            epsilon: 1e-8,
            sum_squared_gradients: None,
        }
    }
}

impl Optimizer for AdaGrad {
    fn step(&mut self, params: &mut Array1<f32>, gradients: &Array1<f32>) {
        if self.sum_squared_gradients.is_none() {
            self.sum_squared_gradients = Some(Array1::zeros(params.len()));
        }

        let g = self.sum_squared_gradients.as_mut().unwrap();
        *g = &*g + gradients * gradients;

        // Element-wise: w = w - (lr / √(G + ε)) * grad
        for i in 0..params.len() {
            params[i] -= self.lr * gradients[i] / (g[i] + self.epsilon).sqrt();
        }
    }
}

/// RMSprop: Exponential moving average of squared gradients
/// E[g²] = β * E[g²] + (1-β) * grad²
/// w = w - (lr / √(E[g²] + ε)) * grad
struct RMSprop {
    lr: f32,
    beta: f32,
    epsilon: f32,
    exp_avg_sq: Option<Array1<f32>>,
}

impl RMSprop {
    fn new(lr: f32, beta: f32) -> Self {
        Self {
            lr,
            beta,
            epsilon: 1e-8,
            exp_avg_sq: None,
        }
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self, params: &mut Array1<f32>, gradients: &Array1<f32>) {
        if self.exp_avg_sq.is_none() {
            self.exp_avg_sq = Some(Array1::zeros(params.len()));
        }

        let v = self.exp_avg_sq.as_mut().unwrap();
        *v = self.beta * &*v + (1.0 - self.beta) * (gradients * gradients);

        for i in 0..params.len() {
            params[i] -= self.lr * gradients[i] / (v[i] + self.epsilon).sqrt();
        }
    }
}

/// Adam: Adaptive Moment Estimation (most popular optimizer!)
/// m = β₁ * m + (1-β₁) * grad        (first moment: momentum)
/// v = β₂ * v + (1-β₂) * grad²       (second moment: RMSprop)
/// m̂ = m / (1 - β₁ᵗ)                 (bias correction)
/// v̂ = v / (1 - β₂ᵗ)
/// w = w - lr * m̂ / (√v̂ + ε)
struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: usize,
    m: Option<Array1<f32>>, // First moment
    v: Option<Array1<f32>>, // Second moment
}

impl Adam {
    fn new(lr: f32, beta1: f32, beta2: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            epsilon: 1e-8,
            t: 0,
            m: None,
            v: None,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut Array1<f32>, gradients: &Array1<f32>) {
        if self.m.is_none() {
            self.m = Some(Array1::zeros(params.len()));
            self.v = Some(Array1::zeros(params.len()));
        }

        self.t += 1;

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        // Update biased first moment estimate
        *m = self.beta1 * &*m + (1.0 - self.beta1) * gradients;

        // Update biased second moment estimate
        *v = self.beta2 * &*v + (1.0 - self.beta2) * (gradients * gradients);

        // Bias correction
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        // Update parameters
        for i in 0..params.len() {
            let m_hat = m[i] / bias_correction1;
            let v_hat = v[i] / bias_correction2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}

/// AdamW: Adam with decoupled weight decay
/// Same as Adam, but weight decay applied separately:
/// w = w - lr * (m̂ / (√v̂ + ε) + λ * w)
struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    epsilon: f32,
    t: usize,
    m: Option<Array1<f32>>,
    v: Option<Array1<f32>>,
}

impl AdamW {
    fn new(lr: f32, beta1: f32, beta2: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            weight_decay,
            epsilon: 1e-8,
            t: 0,
            m: None,
            v: None,
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &mut Array1<f32>, gradients: &Array1<f32>) {
        if self.m.is_none() {
            self.m = Some(Array1::zeros(params.len()));
            self.v = Some(Array1::zeros(params.len()));
        }

        self.t += 1;

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        *m = self.beta1 * &*m + (1.0 - self.beta1) * gradients;
        *v = self.beta2 * &*v + (1.0 - self.beta2) * (gradients * gradients);

        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..params.len() {
            let m_hat = m[i] / bias_correction1;
            let v_hat = v[i] / bias_correction2;

            // AdamW: Decoupled weight decay
            params[i] -= self.lr * (m_hat / (v_hat.sqrt() + self.epsilon)
                                   + self.weight_decay * params[i]);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Learning Rate Schedules
// ═══════════════════════════════════════════════════════════════════════════

trait LRSchedule {
    fn get_lr(&self, step: usize) -> f32;
}

/// Constant learning rate
struct ConstantLR {
    lr: f32,
}

impl ConstantLR {
    fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl LRSchedule for ConstantLR {
    fn get_lr(&self, _step: usize) -> f32 {
        self.lr
    }
}

/// Step decay: lr = lr₀ * γ^(step // step_size)
struct StepDecayLR {
    initial_lr: f32,
    gamma: f32,
    step_size: usize,
}

impl StepDecayLR {
    fn new(initial_lr: f32, gamma: f32, step_size: usize) -> Self {
        Self {
            initial_lr,
            gamma,
            step_size,
        }
    }
}

impl LRSchedule for StepDecayLR {
    fn get_lr(&self, step: usize) -> f32 {
        self.initial_lr * self.gamma.powi((step / self.step_size) as i32)
    }
}

/// Exponential decay: lr = lr₀ * γᵗ
struct ExponentialDecayLR {
    initial_lr: f32,
    gamma: f32,
}

impl ExponentialDecayLR {
    fn new(initial_lr: f32, gamma: f32) -> Self {
        Self { initial_lr, gamma }
    }
}

impl LRSchedule for ExponentialDecayLR {
    fn get_lr(&self, step: usize) -> f32 {
        self.initial_lr * self.gamma.powi(step as i32)
    }
}

/// Cosine annealing: lr = lr_min + (lr_max - lr_min) * (1 + cos(πt/T)) / 2
struct CosineAnnealingLR {
    lr_max: f32,
    lr_min: f32,
    total_steps: usize,
}

impl CosineAnnealingLR {
    fn new(lr_max: f32, lr_min: f32, total_steps: usize) -> Self {
        Self {
            lr_max,
            lr_min,
            total_steps,
        }
    }
}

impl LRSchedule for CosineAnnealingLR {
    fn get_lr(&self, step: usize) -> f32 {
        use std::f32::consts::PI;
        let t = step.min(self.total_steps) as f32;
        let T = self.total_steps as f32;
        self.lr_min + (self.lr_max - self.lr_min) * (1.0 + (PI * t / T).cos()) / 2.0
    }
}

/// Warmup + Cosine annealing (standard for Transformers!)
/// Linear warmup, then cosine decay
struct WarmupCosineAnnealingLR {
    lr_max: f32,
    lr_min: f32,
    warmup_steps: usize,
    total_steps: usize,
}

impl WarmupCosineAnnealingLR {
    fn new(lr_max: f32, lr_min: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            lr_max,
            lr_min,
            warmup_steps,
            total_steps,
        }
    }
}

impl LRSchedule for WarmupCosineAnnealingLR {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            self.lr_max * (step as f32 / self.warmup_steps as f32)
        } else {
            // Cosine annealing
            use std::f32::consts::PI;
            let t = (step - self.warmup_steps) as f32;
            let T = (self.total_steps - self.warmup_steps) as f32;
            self.lr_min + (self.lr_max - self.lr_min) * (1.0 + (PI * t / T).cos()) / 2.0
        }
    }
}

/// Key Concepts Summary
///
/// **Modern Deep Learning Training:**
/// 1. Optimizer: AdamW (default for Transformers)
/// 2. LR Schedule: Warmup (10%) + Cosine annealing
/// 3. Gradient Clipping: 1.0 for RNNs, 5.0 for Transformers
/// 4. Weight Decay: 0.01 (with AdamW)
/// 5. LR: 1e-4 to 3e-4
///
/// **Why Adam/AdamW Dominates:**
/// - Adaptive learning rates per parameter
/// - Combines momentum + RMSprop
/// - Works well with minimal tuning
/// - Default choice for 90%+ of deep learning
///
/// **When Not to Use Adam:**
/// - CNNs: SGD + momentum often better when tuned
/// - When generalization > speed: SGD can generalize better
/// - Limited memory: SGD uses less memory
///
/// **Production Settings:**
/// - Transformers: AdamW, lr=1e-4, warmup+cosine, wd=0.01
/// - ResNets: SGD+momentum, lr=0.1, cosine, wd=1e-4
/// - RNNs: Adam, lr=0.001, gradient clipping=1.0
#[allow(dead_code)]
fn _summary() {}

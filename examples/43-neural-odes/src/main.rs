/// # Neural ODEs (Ordinary Differential Equations) 🌊
///
/// Continuous-depth neural networks using differential equations: Replace discrete
/// layers with continuous transformations. Memory-efficient, elegant, perfect for
/// irregular time series.
///
/// ## What This Example Demonstrates
///
/// 1. **ODE Solvers**: Euler, RK4 for solving differential equations
/// 2. **ResNet Connection**: How ResNet is discretized ODE
/// 3. **Irregular Time Series**: Handle non-uniform sampling
/// 4. **Memory Efficiency**: O(1) vs O(depth) backprop
///
/// ## Why Neural ODEs Matter
///
/// - **Best Paper NeurIPS 2018**: Major breakthrough
/// - **Memory efficiency**: O(1) backprop with adjoint method
/// - **Irregular time series**: Perfect for medical, sensor data
/// - **Elegant theory**: Unifies deep learning and differential equations
///
/// ## The Core Idea
///
/// ```
/// Traditional: h_{t+1} = h_t + f(h_t)  [Discrete layers]
/// Neural ODE:  dh/dt = f(h, t)         [Continuous transformation]
/// ```

use ndarray::Array1;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         Neural ODEs (Continuous Depth Networks)          ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Demonstrate ODE solving
    demo_ode_solvers();

    // Demonstrate ResNet as discretized ODE
    demo_resnet_connection();

    // Demonstrate irregular time series
    demo_irregular_timeseries();
}

/// Demonstrate different ODE solvers
fn demo_ode_solvers() {
    println!("═══ ODE Solvers: Euler vs RK4 ═══\n");

    // Simple ODE: dh/dt = -0.5 * h (exponential decay)
    // Analytical solution: h(t) = h(0) * e^(-0.5t)
    let dynamics = |h: f32, _t: f32| -0.5 * h;

    let h_0 = 1.0_f32;
    let t_final = 2.0_f32;
    let analytical_solution = h_0 * (-0.5 * t_final).exp();

    println!("ODE: dh/dt = -0.5h");
    println!("Initial condition: h(0) = {}", h_0);
    println!("Solve to: t = {}", t_final);
    println!("Analytical solution: h({}) = {:.6}\n", t_final, analytical_solution);

    // Euler's method with different step sizes
    println!("Euler's Method:");
    println!("─────────────────────────────────────────────────");
    println!("Step Size   Steps   Result      Error");
    println!("─────────────────────────────────────────────────");

    for &dt in &[0.5, 0.2, 0.1, 0.05] {
        let steps = (t_final / dt) as usize;
        let result = euler_method(&dynamics, h_0, t_final, dt);
        let error = (result - analytical_solution).abs();
        println!("{:8.2}    {:4}    {:.6}    {:.6}",
                 dt, steps, result, error);
    }

    // RK4 method
    println!("\nRunge-Kutta 4th Order (RK4):");
    println!("─────────────────────────────────────────────────");
    println!("Step Size   Steps   Result      Error");
    println!("─────────────────────────────────────────────────");

    for &dt in &[0.5, 0.2, 0.1, 0.05] {
        let steps = (t_final / dt) as usize;
        let result = rk4_method(&dynamics, h_0, t_final, dt);
        let error = (result - analytical_solution).abs();
        println!("{:8.2}    {:4}    {:.6}    {:.6}",
                 dt, steps, result, error);
    }

    println!("\n💡 Key Insight:");
    println!("   RK4 is much more accurate than Euler for same step size");
    println!("   RK4 with dt=0.5: error = 10^-6");
    println!("   Euler with dt=0.5: error = 10^-2\n");
}

/// Euler's method for solving ODEs
fn euler_method<F>(dynamics: &F, h_0: f32, t_final: f32, dt: f32) -> f32
where
    F: Fn(f32, f32) -> f32,
{
    let steps = (t_final / dt) as usize;
    let mut h = h_0;
    let mut t = 0.0;

    for _ in 0..steps {
        // Euler update: h_{t+1} = h_t + dt * f(h_t, t)
        h = h + dt * dynamics(h, t);
        t += dt;
    }

    h
}

/// Runge-Kutta 4th order method
fn rk4_method<F>(dynamics: &F, h_0: f32, t_final: f32, dt: f32) -> f32
where
    F: Fn(f32, f32) -> f32,
{
    let steps = (t_final / dt) as usize;
    let mut h = h_0;
    let mut t = 0.0;

    for _ in 0..steps {
        // RK4 update
        let k1 = dynamics(h, t);
        let k2 = dynamics(h + dt/2.0 * k1, t + dt/2.0);
        let k3 = dynamics(h + dt/2.0 * k2, t + dt/2.0);
        let k4 = dynamics(h + dt * k3, t + dt);

        h = h + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
        t += dt;
    }

    h
}

/// Demonstrate ResNet as discretized ODE
fn demo_resnet_connection() {
    println!("═══ ResNet as Discretized Neural ODE ═══\n");

    let input = Array1::from(vec![0.5, 0.3, 0.7]);
    println!("Input: {:?}\n", input.to_vec());

    // ResNet-style (discrete)
    println!("ResNet (Discrete Layers):");
    let mut h_discrete = input.clone();
    let num_layers = 4;

    for i in 0..num_layers {
        let residual = simple_residual_function(&h_discrete);
        h_discrete = &h_discrete + &residual;
        println!("  Layer {}: {:?}", i+1, h_discrete.to_vec());
    }

    println!("\nNeural ODE (Continuous):");
    // Neural ODE (continuous approximation)
    let t_final = num_layers as f32; // Each layer ~ unit time
    let dt = 0.25; // Small step size
    let mut h_continuous = input.clone();
    let mut t = 0.0;

    while t < t_final {
        let dh_dt = simple_residual_function(&h_continuous);
        h_continuous = &h_continuous + dt * &dh_dt;
        t += dt;

        if (t - t.floor()).abs() < 0.01 || t >= t_final {
            println!("  t={:.1}: {:?}", t, h_continuous.to_vec());
        }
    }

    println!("\n💡 Connection:");
    println!("   ResNet: h_{{i+1}} = h_i + f(h_i)");
    println!("   Euler ODE: h_{{t+dt}} = h_t + dt · f(h_t, t)");
    println!("   When dt=1: They're identical!");
    println!("   ResNet is just Euler discretization of an ODE!\n");
}

/// Simple residual function (simulates network layer)
fn simple_residual_function(h: &Array1<f32>) -> Array1<f32> {
    // Simple transformation: f(h) = tanh(0.5h)
    h.map(|&x| (0.5 * x).tanh())
}

/// Demonstrate irregular time series
fn demo_irregular_timeseries() {
    println!("═══ Irregular Time Series (Medical Data Example) ═══\n");

    // Simulate irregular measurements (e.g., hospital visits)
    let timestamps = vec![0.0, 0.5, 1.2, 3.5, 5.0, 8.2];
    let observations = vec![1.0, 0.9, 0.7, 0.4, 0.3, 0.1];

    println!("Patient measurements (irregular intervals):");
    println!("─────────────────────────────────────────");
    println!("Time    Observation   Interval");
    println!("─────────────────────────────────────────");

    for i in 0..timestamps.len() {
        let interval = if i > 0 {
            timestamps[i] - timestamps[i-1]
        } else {
            0.0
        };
        println!("{:.1}      {:.1}          {:.1}",
                 timestamps[i], observations[i], interval);
    }

    println!("\n💡 Challenge for Traditional RNNs:");
    println!("   - Requires fixed time steps");
    println!("   - Need to resample or pad data");
    println!("   - Loses temporal information\n");

    println!("✓ Neural ODE Solution:");
    println!("   - Evolve ODE to exact timestamps");
    println!("   - No resampling needed");
    println!("   - Preserves temporal dynamics\n");

    // Demonstrate ODE evolution
    println!("Neural ODE Evolution:");
    println!("─────────────────────────────────────────");

    // Simple decay dynamics
    let dynamics = |h: f32, _t: f32| -0.3 * h;

    let mut h = observations[0];
    println!("t={:.1}: h = {:.3} (observed)", timestamps[0], h);

    for i in 1..timestamps.len() {
        let dt_total = timestamps[i] - timestamps[i-1];
        let steps = 10; // More steps for accuracy
        let dt = dt_total / steps as f32;

        // Evolve ODE
        for _ in 0..steps {
            h = h + dt * dynamics(h, 0.0);
        }

        println!("t={:.1}: h = {:.3} (predicted: {:.3}, observed: {:.3})",
                 timestamps[i], h, h, observations[i]);
    }

    println!("\n📊 Applications:");
    println!("   - Medical records: Irregular hospital visits");
    println!("   - Sensor data: Variable sampling rates");
    println!("   - Financial data: Irregular trading times");
    println!("   - Climate data: Non-uniform measurements\n");
}

/// Key Concepts Summary
///
/// **Neural ODE Definition:**
/// ```
/// Standard ResNet: h_{t+1} = h_t + f(h_t, θ)  [Discrete]
/// Neural ODE:      dh/dt = f(h(t), t, θ)      [Continuous]
///
/// Solve ODE: h(T) = h(0) + ∫₀ᵀ f(h(t), t) dt
/// ```
///
/// **ODE Solvers:**
/// ```
/// Euler (simple, less accurate):
///   h_{t+1} = h_t + Δt · f(h_t, t)
///
/// RK4 (4th order, more accurate):
///   k1 = f(h_t, t)
///   k2 = f(h_t + Δt/2 · k1, t + Δt/2)
///   k3 = f(h_t + Δt/2 · k2, t + Δt/2)
///   k4 = f(h_t + Δt · k3, t + Δt)
///   h_{t+1} = h_t + Δt/6 · (k1 + 2k2 + 2k3 + k4)
///
/// Adaptive (DOPRI5):
///   Automatically adjust step size for accuracy/efficiency
/// ```
///
/// **Memory Efficiency (Adjoint Method):**
/// ```
/// Standard backprop: Store all intermediate states
///   Memory: O(depth) - scales with network depth
///
/// Adjoint method: Solve ODE backward in time
///   Memory: O(1) - constant memory!
///
/// Adjoint ODE:
///   da/dt = -a(t)^T · ∂f/∂h
///   ∂L/∂θ = -∫₀ᵀ a(t)^T · ∂f/∂θ dt
/// ```
///
/// **Benefits:**
/// 1. **Memory efficient**: O(1) vs O(depth)
/// 2. **Adaptive computation**: More steps for hard inputs
/// 3. **Continuous depth**: Can evaluate at any t
/// 4. **Irregular time series**: Natural fit
/// 5. **Parameter efficient**: Shared f across depth
///
/// **Applications:**
/// - **Irregular time series**: Medical records, sensor data
/// - **Normalizing flows**: Continuous transformations
/// - **Physical systems**: Learn dynamics from observations
/// - **Memory-constrained**: When depth is limited by memory
///
/// **When to Use:**
/// ✓ Irregular time series (non-uniform sampling)
/// ✓ Memory-constrained scenarios
/// ✓ Modeling physical dynamics
/// ✓ Continuous-time processes
///
/// × Standard image classification (ResNet faster)
/// × Real-time inference (can be slower)
/// × When regular sampling is fine
///
/// **Trade-offs:**
/// ```
/// Memory: O(1) ✓✓✓ (adjoint method)
/// Computation: Can be slower (many function evals)
/// Expressiveness: Very high ✓✓
/// Implementation: More complex
/// ```
///
/// **Modern Variants:**
/// - **Augmented Neural ODEs**: Add auxiliary dimensions (more expressive)
/// - **Latent ODEs**: For sequential data with missing values
/// - **Second-order ODEs**: d²h/dt² = f(h, dh/dt, t)
/// - **Hamiltonian Neural Networks**: Energy-conserving dynamics
///
/// **Impact:**
/// - Best Paper NeurIPS 2018
/// - Unified deep learning and differential equations
/// - Enabled memory-efficient deep networks
/// - Perfect for irregular time series
/// - Beautiful theoretical framework
#[allow(dead_code)]
fn _summary() {}

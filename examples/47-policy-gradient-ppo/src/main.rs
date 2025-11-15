use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Distribution, Normal, Uniform};
use rand::Rng;
use std::f64;

/// Simple CartPole-like environment for demonstration
/// State: [position, velocity, angle, angular_velocity]
/// Actions: 0 (left), 1 (right)
struct CartPoleEnv {
    state: Array1<f64>,
    steps: usize,
    max_steps: usize,
    rng: rand::rngs::ThreadRng,
}

impl CartPoleEnv {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        let state = Array1::from_vec(vec![
            rng.gen_range(-0.05..0.05),
            rng.gen_range(-0.05..0.05),
            rng.gen_range(-0.05..0.05),
            rng.gen_range(-0.05..0.05),
        ]);

        CartPoleEnv {
            state,
            steps: 0,
            max_steps: 500,
            rng,
        }
    }

    fn reset(&mut self) -> Array1<f64> {
        self.state = Array1::from_vec(vec![
            self.rng.gen_range(-0.05..0.05),
            self.rng.gen_range(-0.05..0.05),
            self.rng.gen_range(-0.05..0.05),
            self.rng.gen_range(-0.05..0.05),
        ]);
        self.steps = 0;
        self.state.clone()
    }

    fn step(&mut self, action: usize) -> (Array1<f64>, f64, bool) {
        let force = if action == 1 { 1.0 } else { -1.0 };

        let gravity = 9.8;
        let cart_mass = 1.0;
        let pole_mass = 0.1;
        let total_mass = cart_mass + pole_mass;
        let length = 0.5;
        let pole_mass_length = pole_mass * length;
        let dt = 0.02;

        let x = self.state[0];
        let x_dot = self.state[1];
        let theta = self.state[2];
        let theta_dot = self.state[3];

        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let temp = (force + pole_mass_length * theta_dot.powi(2) * sin_theta) / total_mass;
        let theta_acc = (gravity * sin_theta - cos_theta * temp)
            / (length * (4.0/3.0 - pole_mass * cos_theta.powi(2) / total_mass));
        let x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass;

        self.state[0] = x + dt * x_dot;
        self.state[1] = x_dot + dt * x_acc;
        self.state[2] = theta + dt * theta_dot;
        self.state[3] = theta_dot + dt * theta_acc;

        self.steps += 1;

        // Episode terminates if pole falls too far or cart goes out of bounds
        let done = self.state[0].abs() > 2.4
            || self.state[2].abs() > 0.2095  // ~12 degrees
            || self.steps >= self.max_steps;

        let reward = if !done { 1.0 } else { 0.0 };

        (self.state.clone(), reward, done)
    }

    fn state_dim() -> usize { 4 }
    fn action_dim() -> usize { 2 }
}

/// Policy Network (Actor) - outputs action probabilities
struct PolicyNetwork {
    w1: Array2<f64>,
    b1: Array1<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,
    w_out: Array2<f64>,
    b_out: Array1<f64>,
}

impl PolicyNetwork {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let scale = 0.1;

        PolicyNetwork {
            w1: Array2::random((input_dim, hidden_dim), Uniform::new(-scale, scale)),
            b1: Array1::zeros(hidden_dim),
            w2: Array2::random((hidden_dim, hidden_dim), Uniform::new(-scale, scale)),
            b2: Array1::zeros(hidden_dim),
            // Small initialization for output layer helps start near uniform policy
            w_out: Array2::random((hidden_dim, output_dim), Uniform::new(-0.01, 0.01)),
            b_out: Array1::zeros(output_dim),
        }
    }

    /// Forward pass returning action probabilities
    fn forward(&self, state: &Array1<f64>) -> Array1<f64> {
        // Layer 1
        let z1 = state.dot(&self.w1) + &self.b1;
        let h1 = z1.mapv(|x| x.tanh());

        // Layer 2
        let z2 = h1.dot(&self.w2) + &self.b2;
        let h2 = z2.mapv(|x| x.tanh());

        // Output layer with softmax
        let logits = h2.dot(&self.w_out) + &self.b_out;
        softmax(&logits)
    }

    /// Sample an action from the policy
    fn sample_action(&self, state: &Array1<f64>, rng: &mut rand::rngs::ThreadRng) -> usize {
        let probs = self.forward(state);
        sample_categorical(&probs, rng)
    }

    /// Get log probability of an action
    fn log_prob(&self, state: &Array1<f64>, action: usize) -> f64 {
        let probs = self.forward(state);
        (probs[action] + 1e-8).ln()
    }
}

/// Value Network (Critic) - estimates state values
struct ValueNetwork {
    w1: Array2<f64>,
    b1: Array1<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,
    w_out: Array2<f64>,
    b_out: Array1<f64>,
}

impl ValueNetwork {
    fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let scale = 0.1;

        ValueNetwork {
            w1: Array2::random((input_dim, hidden_dim), Uniform::new(-scale, scale)),
            b1: Array1::zeros(hidden_dim),
            w2: Array2::random((hidden_dim, hidden_dim), Uniform::new(-scale, scale)),
            b2: Array1::zeros(hidden_dim),
            w_out: Array2::random((hidden_dim, 1), Uniform::new(-scale, scale)),
            b_out: Array1::zeros(1),
        }
    }

    fn forward(&self, state: &Array1<f64>) -> f64 {
        // Layer 1
        let z1 = state.dot(&self.w1) + &self.b1;
        let h1 = z1.mapv(|x| x.tanh());

        // Layer 2
        let z2 = h1.dot(&self.w2) + &self.b2;
        let h2 = z2.mapv(|x| x.tanh());

        // Output
        let value = h2.dot(&self.w_out) + &self.b_out;
        value[0]
    }
}

/// Experience collected during rollout
#[derive(Clone)]
struct Experience {
    state: Array1<f64>,
    action: usize,
    reward: f64,
    next_state: Array1<f64>,
    done: bool,
    log_prob: f64,
    value: f64,
}

/// Trajectory buffer for storing experiences
struct TrajectoryBuffer {
    experiences: Vec<Experience>,
}

impl TrajectoryBuffer {
    fn new() -> Self {
        TrajectoryBuffer {
            experiences: Vec::new(),
        }
    }

    fn push(&mut self, exp: Experience) {
        self.experiences.push(exp);
    }

    fn clear(&mut self) {
        self.experiences.clear();
    }

    fn len(&self) -> usize {
        self.experiences.len()
    }

    /// Compute Generalized Advantage Estimation (GAE)
    fn compute_gae(&self, gamma: f64, lambda: f64) -> (Vec<f64>, Vec<f64>) {
        let n = self.experiences.len();
        let mut advantages = vec![0.0; n];
        let mut returns = vec![0.0; n];

        let mut gae = 0.0;

        // Compute GAE backwards through trajectory
        for t in (0..n).rev() {
            let exp = &self.experiences[t];

            // TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            let next_value = if exp.done { 0.0 } else {
                if t < n - 1 {
                    self.experiences[t + 1].value
                } else {
                    0.0
                }
            };

            let delta = exp.reward + gamma * next_value - exp.value;

            // GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
            gae = delta + gamma * lambda * gae * (1.0 - exp.done as i32 as f64);
            advantages[t] = gae;

            // Return is advantage + value
            returns[t] = advantages[t] + exp.value;
        }

        (advantages, returns)
    }
}

/// PPO Agent
struct PPOAgent {
    policy: PolicyNetwork,
    value: ValueNetwork,
    gamma: f64,
    lambda: f64,
    epsilon: f64,  // Clip range
    learning_rate: f64,
    value_coef: f64,
    entropy_coef: f64,
}

impl PPOAgent {
    fn new(state_dim: usize, action_dim: usize) -> Self {
        PPOAgent {
            policy: PolicyNetwork::new(state_dim, 64, action_dim),
            value: ValueNetwork::new(state_dim, 64),
            gamma: 0.99,
            lambda: 0.95,
            epsilon: 0.2,
            learning_rate: 3e-4,
            value_coef: 0.5,
            entropy_coef: 0.01,
        }
    }

    /// Collect trajectories using current policy
    fn collect_trajectories(
        &self,
        env: &mut CartPoleEnv,
        num_steps: usize,
        rng: &mut rand::rngs::ThreadRng,
    ) -> TrajectoryBuffer {
        let mut buffer = TrajectoryBuffer::new();
        let mut state = env.reset();
        let mut episode_reward = 0.0;

        for _ in 0..num_steps {
            let action = self.policy.sample_action(&state, rng);
            let log_prob = self.policy.log_prob(&state, action);
            let value = self.value.forward(&state);

            let (next_state, reward, done) = env.step(action);
            episode_reward += reward;

            buffer.push(Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
                log_prob,
                value,
            });

            if done {
                state = env.reset();
                episode_reward = 0.0;
            } else {
                state = next_state;
            }
        }

        buffer
    }

    /// PPO update with clipped objective
    fn update(&mut self, buffer: &TrajectoryBuffer, epochs: usize) -> (f64, f64, f64) {
        let (advantages, returns) = buffer.compute_gae(self.gamma, self.lambda);

        // Normalize advantages for stable training
        let adv_mean = advantages.iter().sum::<f64>() / advantages.len() as f64;
        let adv_std = (advantages.iter().map(|a| (a - adv_mean).powi(2)).sum::<f64>()
            / advantages.len() as f64).sqrt();
        let normalized_advantages: Vec<f64> = advantages.iter()
            .map(|a| (a - adv_mean) / (adv_std + 1e-8))
            .collect();

        let mut total_policy_loss = 0.0;
        let mut total_value_loss = 0.0;
        let mut total_entropy = 0.0;
        let mut update_count = 0;

        // Multiple epochs of updates
        for _ in 0..epochs {
            for (i, exp) in buffer.experiences.iter().enumerate() {
                let advantage = normalized_advantages[i];
                let return_value = returns[i];

                // Compute current policy log prob and value
                let current_log_prob = self.policy.log_prob(&exp.state, exp.action);
                let current_value = self.value.forward(&exp.state);

                // Probability ratio: r(θ) = π_new / π_old
                let ratio = (current_log_prob - exp.log_prob).exp();

                // PPO clipped objective
                let unclipped = ratio * advantage;
                let clipped = ratio.clamp(1.0 - self.epsilon, 1.0 + self.epsilon) * advantage;
                let policy_loss = -unclipped.min(clipped);

                // Value function loss
                let value_loss = (current_value - return_value).powi(2);

                // Entropy bonus for exploration
                let probs = self.policy.forward(&exp.state);
                let entropy = -probs.iter()
                    .map(|p| if *p > 1e-8 { p * p.ln() } else { 0.0 })
                    .sum::<f64>();

                // Combined loss
                let total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy;

                // Simple gradient descent update (for educational purposes)
                self.update_policy(&exp.state, exp.action, policy_loss, entropy);
                self.update_value(&exp.state, return_value);

                total_policy_loss += policy_loss;
                total_value_loss += value_loss;
                total_entropy += entropy;
                update_count += 1;
            }
        }

        (
            total_policy_loss / update_count as f64,
            total_value_loss / update_count as f64,
            total_entropy / update_count as f64,
        )
    }

    /// Update policy network (simplified gradient descent)
    fn update_policy(&mut self, state: &Array1<f64>, action: usize, loss: f64, entropy: f64) {
        let lr = self.learning_rate;
        let probs = self.policy.forward(state);

        // Compute gradients (simplified for educational purposes)
        // In production, use automatic differentiation
        let h = 1e-5;

        // Update output weights
        for i in 0..self.policy.w_out.shape()[0] {
            for j in 0..self.policy.w_out.shape()[1] {
                self.policy.w_out[[i, j]] -= lr * loss * 0.01 * (if j == action { 1.0 } else { -1.0 });
            }
        }
    }

    /// Update value network
    fn update_value(&mut self, state: &Array1<f64>, target: f64) {
        let lr = self.learning_rate;
        let prediction = self.value.forward(state);
        let error = prediction - target;

        // Gradient descent on MSE
        // Simplified update - in production use automatic differentiation
        for i in 0..self.value.w_out.shape()[0] {
            self.value.w_out[[i, 0]] -= lr * error * 0.01;
        }
    }
}

/// Vanilla Policy Gradient Agent (for comparison)
struct VanillaPGAgent {
    policy: PolicyNetwork,
    learning_rate: f64,
}

impl VanillaPGAgent {
    fn new(state_dim: usize, action_dim: usize) -> Self {
        VanillaPGAgent {
            policy: PolicyNetwork::new(state_dim, 64, action_dim),
            learning_rate: 1e-3,
        }
    }

    fn collect_episode(&self, env: &mut CartPoleEnv, rng: &mut rand::rngs::ThreadRng)
        -> (Vec<Array1<f64>>, Vec<usize>, Vec<f64>) {
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut rewards = Vec::new();

        let mut state = env.reset();

        loop {
            let action = self.policy.sample_action(&state, rng);
            states.push(state.clone());
            actions.push(action);

            let (next_state, reward, done) = env.step(action);
            rewards.push(reward);

            if done {
                break;
            }
            state = next_state;
        }

        (states, actions, rewards)
    }

    fn update(&mut self, states: &[Array1<f64>], actions: &[usize], returns: &[f64]) {
        let lr = self.learning_rate;

        // REINFORCE: ∇J = E[∇log π(a|s) * R]
        for (i, (state, &action)) in states.iter().zip(actions.iter()).enumerate() {
            let return_val = returns[i];

            // Simplified gradient update
            for j in 0..self.policy.w_out.shape()[0] {
                for k in 0..self.policy.w_out.shape()[1] {
                    self.policy.w_out[[j, k]] += lr * return_val * 0.01
                        * (if k == action { 1.0 } else { -1.0 });
                }
            }
        }
    }
}

/// Utility functions

fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Array1<f64> = x.mapv(|v| (v - max).exp());
    let sum = exp.sum();
    exp / sum
}

fn sample_categorical(probs: &Array1<f64>, rng: &mut rand::rngs::ThreadRng) -> usize {
    let sample: f64 = rng.gen();
    let mut cumsum = 0.0;

    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if sample < cumsum {
            return i;
        }
    }

    probs.len() - 1
}

fn compute_returns(rewards: &[f64], gamma: f64) -> Vec<f64> {
    let mut returns = vec![0.0; rewards.len()];
    let mut g = 0.0;

    for t in (0..rewards.len()).rev() {
        g = rewards[t] + gamma * g;
        returns[t] = g;
    }

    returns
}

fn main() {
    println!("=== Proximal Policy Optimization (PPO) ===\n");
    println!("This example demonstrates PPO, the algorithm that powers ChatGPT's RLHF training!");
    println!("We'll compare PPO with vanilla policy gradient to show its advantages.\n");

    let mut rng = rand::thread_rng();

    // Training parameters
    let num_iterations = 100;
    let steps_per_iteration = 2000;
    let ppo_epochs = 10;

    println!("Training Parameters:");
    println!("  Episodes: {}", num_iterations);
    println!("  Steps per iteration: {}", steps_per_iteration);
    println!("  PPO update epochs: {}", ppo_epochs);
    println!("  Clip range (ε): 0.2");
    println!("  GAE lambda (λ): 0.95\n");

    // Initialize agents
    let mut ppo_agent = PPOAgent::new(CartPoleEnv::state_dim(), CartPoleEnv::action_dim());
    let mut pg_agent = VanillaPGAgent::new(CartPoleEnv::state_dim(), CartPoleEnv::action_dim());

    let mut ppo_env = CartPoleEnv::new();
    let mut pg_env = CartPoleEnv::new();

    println!("Training PPO and Vanilla Policy Gradient...\n");
    println!("{:>4} | {:>12} | {:>12} | {:>10} | {:>10} | {:>10}",
             "Iter", "PPO Reward", "PG Reward", "PPO Loss", "Value Loss", "Entropy");
    println!("{}", "-".repeat(75));

    for iter in 0..num_iterations {
        // PPO Training
        let buffer = ppo_agent.collect_trajectories(&mut ppo_env, steps_per_iteration, &mut rng);
        let (policy_loss, value_loss, entropy) = ppo_agent.update(&buffer, ppo_epochs);

        // Compute average PPO reward
        let ppo_episode_rewards = compute_episode_rewards(&buffer);
        let ppo_avg_reward = if !ppo_episode_rewards.is_empty() {
            ppo_episode_rewards.iter().sum::<f64>() / ppo_episode_rewards.len() as f64
        } else {
            0.0
        };

        // Vanilla PG Training (train on multiple episodes)
        let mut pg_episode_rewards = Vec::new();
        let num_pg_episodes = 5;

        for _ in 0..num_pg_episodes {
            let (states, actions, rewards) = pg_agent.collect_episode(&mut pg_env, &mut rng);
            let returns = compute_returns(&rewards, 0.99);
            pg_agent.update(&states, &actions, &returns);

            pg_episode_rewards.push(rewards.iter().sum::<f64>());
        }

        let pg_avg_reward = pg_episode_rewards.iter().sum::<f64>() / pg_episode_rewards.len() as f64;

        // Print progress every 10 iterations
        if iter % 10 == 0 {
            println!("{:4} | {:12.2} | {:12.2} | {:10.4} | {:10.4} | {:10.4}",
                     iter, ppo_avg_reward, pg_avg_reward, policy_loss, value_loss, entropy);
        }
    }

    println!("\n=== Training Complete ===\n");

    // Final evaluation
    println!("Final Evaluation (100 episodes each):\n");

    let mut ppo_final_rewards = Vec::new();
    let mut pg_final_rewards = Vec::new();

    for _ in 0..100 {
        // PPO evaluation
        let mut state = ppo_env.reset();
        let mut episode_reward = 0.0;

        loop {
            let action = ppo_agent.policy.sample_action(&state, &mut rng);
            let (next_state, reward, done) = ppo_env.step(action);
            episode_reward += reward;

            if done {
                break;
            }
            state = next_state;
        }

        ppo_final_rewards.push(episode_reward);

        // PG evaluation
        let (_, _, rewards) = pg_agent.collect_episode(&mut pg_env, &mut rng);
        pg_final_rewards.push(rewards.iter().sum::<f64>());
    }

    let ppo_mean = ppo_final_rewards.iter().sum::<f64>() / ppo_final_rewards.len() as f64;
    let pg_mean = pg_final_rewards.iter().sum::<f64>() / pg_final_rewards.len() as f64;

    let ppo_std = (ppo_final_rewards.iter()
        .map(|r| (r - ppo_mean).powi(2))
        .sum::<f64>() / ppo_final_rewards.len() as f64).sqrt();
    let pg_std = (pg_final_rewards.iter()
        .map(|r| (r - pg_mean).powi(2))
        .sum::<f64>() / pg_final_rewards.len() as f64).sqrt();

    println!("PPO Results:");
    println!("  Mean Reward: {:.2} ± {:.2}", ppo_mean, ppo_std);
    println!("  Min/Max: {:.2} / {:.2}",
             ppo_final_rewards.iter().cloned().fold(f64::INFINITY, f64::min),
             ppo_final_rewards.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    println!("\nVanilla PG Results:");
    println!("  Mean Reward: {:.2} ± {:.2}", pg_mean, pg_std);
    println!("  Min/Max: {:.2} / {:.2}",
             pg_final_rewards.iter().cloned().fold(f64::INFINITY, f64::min),
             pg_final_rewards.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    println!("\nKey Observations:");
    println!("1. PPO typically shows:");
    println!("   - Higher average rewards");
    println!("   - Lower variance (more stable)");
    println!("   - More consistent performance");

    println!("\n2. PPO's advantages come from:");
    println!("   - Clipped objective prevents destructive updates");
    println!("   - GAE provides stable advantage estimates");
    println!("   - Multiple epochs improve sample efficiency");
    println!("   - Value function reduces variance");

    println!("\n3. This is why PPO is used for:");
    println!("   - ChatGPT RLHF training");
    println!("   - Robotic control (OpenAI Dactyl)");
    println!("   - Complex games (OpenAI Five)");
    println!("   - Any task requiring stable, reliable learning");

    println!("\n=== PPO: The Gold Standard in Policy Optimization ===");
}

/// Helper function to compute episode rewards from trajectory buffer
fn compute_episode_rewards(buffer: &TrajectoryBuffer) -> Vec<f64> {
    let mut episode_rewards = Vec::new();
    let mut current_reward = 0.0;

    for exp in &buffer.experiences {
        current_reward += exp.reward;

        if exp.done {
            episode_rewards.push(current_reward);
            current_reward = 0.0;
        }
    }

    episode_rewards
}

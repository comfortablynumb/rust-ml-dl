//! # Reinforcement Learning: Deep Q-Networks (DQN)
//!
//! Learn to make sequential decisions through trial and error
//! Paper: "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
//!
//! ## Reinforcement Learning Framework
//!
//! **Different from supervised learning:**
//! ```
//! Supervised: (input, label) pairs → learn mapping
//! RL: Agent interacts with environment → learn policy
//!
//! Components:
//! • Agent: Learner/decision maker
//! • Environment: World agent interacts with
//! • State (s): Current situation
//! • Action (a): What agent can do
//! • Reward (r): Feedback signal
//! • Policy (π): Strategy (state → action)
//! ```
//!
//! ## The RL Loop
//!
//! ```
//! 1. Observe state s_t
//! 2. Take action a_t
//! 3. Receive reward r_t
//! 4. Observe new state s_{t+1}
//! 5. Repeat
//!
//! Goal: Maximize cumulative reward
//! ```
//!
//! ## Q-Learning Basics
//!
//! **Q-Function: Expected future reward**
//! ```
//! Q(s, a) = Expected total reward starting from state s,
//!           taking action a, then following optimal policy
//!
//! Optimal policy: π*(s) = argmax_a Q(s, a)
//! "Choose action with highest Q-value"
//! ```
//!
//! ### Bellman Equation
//! ```
//! Q(s, a) = r + γ · max_a' Q(s', a')
//!
//! Where:
//! • r: Immediate reward
//! • γ: Discount factor (0-1)
//! • s': Next state
//! • max_a' Q(s', a'): Best future value
//!
//! Intuition: Current value = immediate + discounted future
//! ```
//!
//! ## Deep Q-Network (DQN)
//!
//! **Problem with traditional Q-learning:**
//! ```
//! Table-based: Store Q(s,a) for every (state, action)
//! Fails for large/continuous state spaces!
//!
//! Atari: 210×160×3 pixels = millions of states
//! Can't store table!
//! ```
//!
//! **Solution: Function approximation**
//! ```
//! Neural network Q(s, a; θ) approximates Q-function
//!
//! Input: State (image pixels)
//! Output: Q-value for each action
//!
//! Example Atari:
//! Input: 84×84×4 grayscale frames
//! Output: [Q(left), Q(right), Q(fire), Q(noop)]
//! ```
//!
//! ## DQN Architecture
//!
//! ```
//! Input: 84×84×4 (4 stacked frames)
//!   ↓
//! Conv1: 32 filters, 8×8, stride 4
//!   ↓
//! Conv2: 64 filters, 4×4, stride 2  
//!   ↓
//! Conv3: 64 filters, 3×3, stride 1
//!   ↓
//! FC1: 512 units
//!   ↓
//! FC2: n_actions outputs (Q-values)
//!
//! Output: Q(s, a) for each action
//! ```
//!
//! ## Training DQN
//!
//! ### 1. Experience Replay
//! ```
//! Problem: Sequential data is correlated
//! Solution: Store transitions, sample randomly
//!
//! Replay buffer: Store (s, a, r, s', done)
//! Training: Sample random mini-batch
//!
//! Benefits:
//! • Breaks correlation
//! • Reuse experiences
//! • More stable learning
//! ```
//!
//! ### 2. Target Network
//! ```
//! Problem: Q-value target uses same network (unstable!)
//! Q_target = r + γ · max_a' Q(s', a'; θ)
//!                              ↑
//!                         Changes every update!
//!
//! Solution: Separate target network
//! Q_target = r + γ · max_a' Q(s', a'; θ⁻)
//!                                     ↑
//!                            Frozen for N steps
//!
//! Update θ⁻ every C steps (copy from θ)
//! ```
//!
//! ### 3. Training Loop
//! ```
//! Initialize Q-network Q(s,a;θ)
//! Initialize target network Q(s,a;θ⁻) = Q(s,a;θ)
//! Initialize replay buffer D
//!
//! For episode in episodes:
//!   Initialize state s
//!   For t in timesteps:
//!     # Epsilon-greedy action selection
//!     With probability ε: random action
//!     Otherwise: a = argmax_a Q(s,a;θ)
//!     
//!     # Execute action
//!     Take action a, observe r, s'
//!     Store (s, a, r, s', done) in D
//!     
//!     # Sample and train
//!     Sample mini-batch from D
//!     Compute target: y = r + γ·max_a' Q(s',a';θ⁻)
//!     Compute loss: L = (y - Q(s,a;θ))²
//!     Update θ
//!     
//!     # Update target network
//!     Every C steps: θ⁻ ← θ
//! ```
//!
//! ## Exploration vs Exploitation
//!
//! ### Epsilon-Greedy
//! ```
//! With probability ε: Explore (random action)
//! With probability 1-ε: Exploit (best known action)
//!
//! Decay schedule:
//! ε_start = 1.0 (fully explore)
//! ε_end = 0.01 (mostly exploit)
//! Decay over 1M steps
//!
//! Balances learning new strategies vs using knowledge
//! ```
//!
//! ## Applications
//!
//! ### Game Playing
//! ```
//! Atari games: Human-level performance
//! Go: AlphaGo (DQN + Monte Carlo Tree Search)
//! StarCraft II: AlphaStar
//! Dota 2: OpenAI Five
//! ```
//!
//! ### Robotics
//! ```
//! Manipulation: Grasping, assembly
//! Navigation: Path planning, obstacle avoidance
//! Locomotion: Walking, running (ANYmal, Boston Dynamics)
//! ```
//!
//! ### Resource Management
//! ```
//! Data center cooling: Google (40% energy reduction)
//! Traffic light control: Reduce congestion
//! Portfolio optimization: Trading strategies
//! ```
//!
//! ### Recommendation Systems
//! ```
//! YouTube, Netflix: Content recommendations
//! Sequential decision making
//! Long-term user engagement
//! ```
//!
//! ## DQN Improvements
//!
//! **Double DQN:**
//! ```
//! Problem: DQN overestimates Q-values
//! Solution: Decouple action selection and evaluation
//!
//! Target: r + γ · Q(s', argmax_a Q(s',a;θ); θ⁻)
//!                      ↑                    ↑
//!                  Select with θ      Evaluate with θ⁻
//! ```
//!
//! **Dueling DQN:**
//! ```
//! Split Q-function:
//! Q(s,a) = V(s) + A(s,a)
//! Where:
//! • V(s): State value
//! • A(s,a): Advantage of action a
//!
//! Benefits: Better state value estimation
//! ```
//!
//! **Prioritized Experience Replay:**
//! ```
//! Sample important transitions more frequently
//! Priority = TD error (how surprising)
//! Learns faster from mistakes
//! ```
//!
//! **Rainbow DQN:**
//! ```
//! Combines 6 improvements:
//! 1. Double DQN
//! 2. Dueling DQN
//! 3. Prioritized replay
//! 4. Multi-step returns
//! 5. Distributional RL
//! 6. Noisy networks
//!
//! State-of-the-art on Atari
//! ```
//!
//! ## Beyond DQN: Modern RL
//!
//! **Policy Gradient Methods:**
//! - A3C, PPO, SAC
//! - Learn policy directly
//! - Better for continuous actions
//!
//! **Model-Based RL:**
//! - Learn environment model
//! - Plan with model
//! - More sample efficient
//!
//! **Offline RL:**
//! - Learn from fixed dataset
//! - No environment interaction
//! - Safer for real-world

fn main() {
    println!("=== Reinforcement Learning: Deep Q-Networks ===\n");
    println!("Learn through interaction: Agent + Environment + Rewards");
    println!("DQN: Neural network approximates Q-function");
    println!("Achievements: Human-level Atari, AlphaGo, robotics");
    println!("Applications: Games, robotics, resource management");
}

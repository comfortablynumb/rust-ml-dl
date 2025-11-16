# Reinforcement Learning (DQN) Example

Learn sequential decision making through trial and error using Deep Q-Networks.

## Overview

RL agents learn by interacting with an environment, receiving rewards, and maximizing cumulative return.

## Running

```bash
cargo run --package reinforcement-learning
```

## Key Concepts

```
Agent observes state → Takes action → Receives reward → Updates policy
```

### DQN Innovations

1. **Experience Replay**: Store and reuse past experiences
2. **Target Network**: Stabilize training
3. **Function Approximation**: Neural network for Q-function

## Applications

- Game playing (Atari, Go, StarCraft)
- Robotics (manipulation, navigation)
- Resource management (data centers, traffic)
- Recommendation systems

## Papers

- [Playing Atari with Deep RL](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013)
- [Human-level control](https://www.nature.com/articles/nature14236) (Mnih et al., 2015)
- [Rainbow DQN](https://arxiv.org/abs/1710.02298) (Hessel et al., 2017)

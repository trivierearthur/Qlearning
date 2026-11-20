# q-learning-from-scratch

Fun RL (Temporal Difference learning) agent using Q-value matrix updates (Bellman equation) to navigate out of randomized FrozenLake mazes.


## Overview

This project implements a **tabular Q-Learning agent from scratch** using pure NumPy—without high-level reinforcement learning frameworks. 

On every run, the script generates a brand-new, guaranteed-solvable 4x4 `FrozenLake-v1` maze, trains an agent over 10,000 episodes to discover the optimal path from Start (`S`) to Goal (`G`) while avoiding holes (`H`), and opens a visual window to watch the trained agent navigate the grid.

---

## How It Works

### 1. The Q-Table (State-Action Memory)
The agent maintains a 2D lookup table (`16 states × 4 actions`: `LEFT`, `DOWN`, `RIGHT`, `UP`) initialized to zeros. Each cell `Q(s, a)` represents the expected cumulative future reward of taking action `a` in state `s`.

### 2. Temporal Difference Learning & The Bellman Equation
Rather than waiting until the end of an episode to learn, the agent updates its Q-table at every step using the **Bellman optimality equation**:

```text
Q(s, a) ← Q(s, a) + α * [ R + γ * max(Q(s', a')) - Q(s, a) ]
```

* **Maximizing Future Rewards:** By combining the immediate reward (`R`) with the discounted maximum value of the next state (`γ * max(Q(s', a'))`), rewards propagate backward from the goal (`+1.0`) to the starting tile.
* **Temporal Difference (TD) Error:** The term `R + γ * max(Q(s', a')) - Q(s, a)` measures the difference between the agent's previous estimate and the newly observed outcome.

### 3. Epsilon-Greedy Exploration
To balance discovering new paths (**exploration**) with using known good paths (**exploitation**), the agent starts with `ε = 1.0` (100% random moves) and decays `ε` by `0.999` after every episode down to `0.01`.

---

## Hyperparameters

| Parameter | Value | Role |
| :--- | :--- | :--- |
| `episodes` | `10000` | Total training runs on the generated maze |
| `alpha` ($\alpha$) | `0.1` | Learning rate (how strongly new experiences override old Q-values) |
| `gamma` ($\gamma$) | `0.99` | Discount factor (prioritizes reaching the goal sooner) |
| `epsilon_decay` | `0.999` | Gradual shift from random exploration to greedy exploitation |
| `min_epsilon` | `0.01` | Minimum exploration floor |

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com
cd q-learning-from-scratch
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the agent
```bash
python main.py
```

### What You Will See
1. **Terminal Output:** The randomly generated 4x4 maze layout, followed by the trained `16x4` Q-value matrix showing learned action values.
2. **Visual Playback:** A Pygame window opens showing the trained agent stepping cleanly from `S` to `G` using `np.argmax(q_table[state])`.

---

## Key Takeaways & Next Steps

* **Generalization via Procedural Maps:** Using `generate_random_map(size=4, p=0.8)` ensures the algorithm converges on arbitrary layouts rather than overfitting to a single hardcoded grid.
* **Scaling Limitation:** Tabular Q-learning works well for small discrete grids, but memory grows rapidly with larger or continuous state spaces. 
* **Future Upgrade:** Replace the NumPy Q-table with a neural network (**Deep Q-Network / DQN** in PyTorch) to handle high-dimensional environments.

import time
import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# Generate a new solvable maze for each run.
maze = generate_random_map(size=4, p=0.8)
print("Random maze for this run:")
print("\n".join(maze))

# Use the same maze for training and the visual test.
env = gym.make("FrozenLake-v1", desc=maze, is_slippery=False)

# 2. Initialize the Q-table with zeros (states x actions)
state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))

# 3. Set hyperparameters
alpha = 0.1          # Learning rate (how much new info overrides old info)
gamma = 0.99         # Discount factor (importance of future rewards)
epsilon = 1.0        # Exploration rate (1.0 = 100% random actions at start)
epsilon_decay = 0.999
min_epsilon = 0.01
episodes = 10000

# 4. Training loop
for episode in range(episodes):
    state, _ = env.reset()
    done = False

    while not done:
        # Epsilon-greedy action selection (explore vs. exploit)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore: random action
        else:
            action = np.argmax(q_table[state])  # Exploit: best known action

        # Take action and observe the outcome
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Bellman Equation: Update the Q-value for the state-action pair
        best_next_q = np.max(q_table[next_state])
        td_target = reward + gamma * best_next_q
        q_table[state, action] += alpha * (td_target - q_table[state, action])

        # Move to the next state
        state = next_state

    # Decay exploration rate after each episode
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

env.close()

print("Trained Q-Table (columns: LEFT, DOWN, RIGHT, UP):")
print(np.array2string(q_table, precision=3, suppress_small=True))

# Watch the trained agent without printing each step to the terminal.
test_env = gym.make(
    "FrozenLake-v1", desc=maze, is_slippery=False, render_mode="human"
)
state, _ = test_env.reset()
done = False

while not done:
    action = np.argmax(q_table[state])
    state, reward, terminated, truncated, _ = test_env.step(action)
    done = terminated or truncated
    time.sleep(0.5)

if reward == 1.0:
    print("SUCCESS! The agent reached the goal.")
else:
    print("FAILED! The agent fell into a hole.")

test_env.close()
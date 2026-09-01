import numpy as np
import random

# ============================================================
# GRID-BASED PAC-MAN ENVIRONMENT
# ============================================================

class GridGame:

    def __init__(self):
        self.rows = 5
        self.cols = 5

        # Fixed positions
        self.start = (0, 0)
        self.food = (4, 4)

        # Ghost positions
        self.ghosts = [(2, 2), (3, 3)]

        self.agent_position = self.start

    def reset(self):
        self.agent_position = self.start
        return self.agent_position

    def step(self, action):

        row, col = self.agent_position

        # Actions:
        # 0 = Up
        # 1 = Down
        # 2 = Left
        # 3 = Right

        if action == 0:
            row -= 1
        elif action == 1:
            row += 1
        elif action == 2:
            col -= 1
        elif action == 3:
            col += 1

        # Prevent agent from leaving the grid
        row = max(0, min(self.rows - 1, row))
        col = max(0, min(self.cols - 1, col))

        new_position = (row, col)

        # ----------------------------------------------------
        # Reward system
        # ----------------------------------------------------

        reward = -1
        done = False

        # Food
        if new_position == self.food:
            reward = 20
            done = True

        # Ghost
        elif new_position in self.ghosts:
            reward = -20
            done = True

        # Normal movement
        else:
            reward = -1

        self.agent_position = new_position

        return new_position, reward, done


# ============================================================
# Q-LEARNING AGENT
# ============================================================

class QLearningAgent:

    def __init__(self, rows, cols):

        # Q-table:
        # rows × cols × 4 actions
        self.q_table = np.zeros((rows, cols, 4))

        # Learning parameters
        self.alpha = 0.1
        self.gamma = 0.9

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    # --------------------------------------------------------
    # Select action using epsilon-greedy strategy
    # --------------------------------------------------------

    def choose_action(self, state):

        row, col = state

        # Exploration
        if random.random() < self.epsilon:
            return random.randint(0, 3)

        # Exploitation
        return np.argmax(self.q_table[row, col])

    # --------------------------------------------------------
    # Update Q-value using Q-Learning equation
    # --------------------------------------------------------

    def update(self, state, action, reward, next_state):

        row, col = state
        next_row, next_col = next_state

        current_q = self.q_table[row, col, action]

        max_next_q = np.max(
            self.q_table[next_row, next_col]
        )

        new_q = current_q + self.alpha * (
            reward +
            self.gamma * max_next_q -
            current_q
        )

        self.q_table[row, col, action] = new_q

    # --------------------------------------------------------
    # Reduce exploration
    # --------------------------------------------------------

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )


# ============================================================
# TRAINING
# ============================================================

env = GridGame()

agent = QLearningAgent(
    env.rows,
    env.cols
)

episodes = 5000
max_steps = 100

rewards_history = []
successful_games = 0
ghost_hits = 0


print("============================================")
print(" Q-LEARNING GRID GAME")
print("============================================")

print("\nTraining started...")


for episode in range(episodes):

    state = env.reset()

    total_reward = 0

    for step in range(max_steps):

        # Choose action
        action = agent.choose_action(state)

        # Take action
        next_state, reward, done = env.step(action)

        # Update Q-table
        agent.update(
            state,
            action,
            reward,
            next_state
        )

        state = next_state

        total_reward += reward

        if done:

            if reward == 20:
                successful_games += 1

            elif reward == -20:
                ghost_hits += 1

            break

    # Decay exploration
    agent.decay_epsilon()

    rewards_history.append(total_reward)


# ============================================================
# TRAINING RESULTS
# ============================================================

print("\n============================================")
print(" TRAINING RESULTS")
print("============================================")

print("Number of Episodes :", episodes)

print(
    "Successful Games   :",
    successful_games
)

print(
    "Ghost Collisions   :",
    ghost_hits
)

print(
    "Final Epsilon      :",
    round(agent.epsilon, 4)
)

print(
    "Average Reward     :",
    round(np.mean(rewards_history[-100:]), 2)
)


# ============================================================
# DISPLAY Q-TABLE
# ============================================================

print("\n============================================")
print(" LEARNED Q-TABLE")
print("============================================")

for row in range(env.rows):

    for col in range(env.cols):

        values = agent.q_table[row, col]

        print(
            f"State ({row},{col}) : "
            f"Up={values[0]:6.2f}  "
            f"Down={values[1]:6.2f}  "
            f"Left={values[2]:6.2f}  "
            f"Right={values[3]:6.2f}"
        )


# ============================================================
# DISPLAY LEARNED POLICY
# ============================================================

print("\n============================================")
print(" LEARNED POLICY")
print("============================================")

symbols = {
    0: "↑",
    1: "↓",
    2: "←",
    3: "→"
}

for row in range(env.rows):

    policy_row = ""

    for col in range(env.cols):

        # Food
        if (row, col) == env.food:
            symbol = "F"

        # Ghost
        elif (row, col) in env.ghosts:
            symbol = "G"

        # Starting position
        elif (row, col) == env.start:
            action = np.argmax(
                agent.q_table[row, col]
            )
            symbol = symbols[action]

        else:
            action = np.argmax(
                agent.q_table[row, col]
            )
            symbol = symbols[action]

        policy_row += f"{symbol} "

    print(policy_row)


# ============================================================
# EVALUATION
# ============================================================

print("\n============================================")
print(" EVALUATION")
print("============================================")

# Turn off exploration
agent.epsilon = 0

evaluation_games = 10

wins = 0
losses = 0


for game in range(evaluation_games):

    state = env.reset()

    print(f"\nGame {game + 1}")

    path = [state]

    for step in range(max_steps):

        # Select best learned action
        action = agent.choose_action(state)

        next_state, reward, done = env.step(action)

        path.append(next_state)

        state = next_state

        if done:

            if reward == 20:
                wins += 1
                print("Result: FOOD COLLECTED")

            elif reward == -20:
                losses += 1
                print("Result: HIT BY GHOST")

            break

    print("Path:", path)


# ============================================================
# FINAL EVALUATION RESULTS
# ============================================================

print("\n============================================")
print(" FINAL EVALUATION RESULTS")
print("============================================")

print("Evaluation Games :", evaluation_games)
print("Wins             :", wins)
print("Losses           :", losses)

print(
    "Success Rate     :",
    round((wins / evaluation_games) * 100, 2),
    "%"
)

print("\nTraining and evaluation completed.")

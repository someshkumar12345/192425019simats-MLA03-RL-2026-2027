import numpy as np
import random

# ============================================================
# RL-BASED SMART ENERGY MANAGEMENT SYSTEM
# ============================================================

class SmartEnergyEnvironment:

    def __init__(self):
        # States:
        # 0 = Low energy demand
        # 1 = Medium energy demand
        # 2 = High energy demand
        #
        # Occupancy:
        # 0 = Unoccupied
        # 1 = Occupied
        #
        # Temperature:
        # 0 = Comfortable
        # 1 = Uncomfortable

        self.energy_states = 3
        self.occupancy_states = 2
        self.temperature_states = 2

        # Actions:
        # 0 = Turn OFF unnecessary loads
        # 1 = Normal operation
        # 2 = Energy saving mode

        self.actions = 3

        self.state = None

    def reset(self):
        self.state = (
            random.randint(0, 2),  # Energy demand
            random.randint(0, 1),  # Occupancy
            random.randint(0, 1)   # Temperature
        )
        return self.state

    def step(self, action):

        energy, occupancy, temperature = self.state

        # ----------------------------------------------------
        # Base energy consumption
        # ----------------------------------------------------

        energy_consumption = {
            0: 2,   # Low
            1: 5,   # Medium
            2: 8    # High
        }[energy]

        # ----------------------------------------------------
        # Action effects
        # ----------------------------------------------------

        if action == 0:
            # Turn OFF unnecessary loads
            energy_consumption -= 2

        elif action == 1:
            # Normal operation
            energy_consumption += 0

        elif action == 2:
            # Energy saving mode
            energy_consumption -= 3

        energy_consumption = max(0, energy_consumption)

        # ----------------------------------------------------
        # Safety constraint
        # ----------------------------------------------------

        # If the building is occupied and temperature is
        # uncomfortable, aggressive energy saving is unsafe.

        safety_violation = False

        if occupancy == 1 and temperature == 1 and action == 2:
            safety_violation = True

        # ----------------------------------------------------
        # Fairness constraint
        # ----------------------------------------------------

        # Occupied users should not be continuously denied
        # normal energy services.

        fairness_violation = False

        if occupancy == 1 and action == 0:
            fairness_violation = True

        # ----------------------------------------------------
        # Reward calculation
        # ----------------------------------------------------

        reward = -energy_consumption

        # Reward for energy saving
        if action == 2 and not safety_violation:
            reward += 3

        # Safety penalty
        if safety_violation:
            reward -= 15

        # Fairness penalty
        if fairness_violation:
            reward -= 5

        # ----------------------------------------------------
        # Generate next state
        # ----------------------------------------------------

        next_state = (
            random.randint(0, 2),
            random.randint(0, 1),
            random.randint(0, 1)
        )

        self.state = next_state

        done = False

        return next_state, reward, done, energy_consumption, \
               safety_violation, fairness_violation


# ============================================================
# Q-LEARNING AGENT
# ============================================================

class QLearningAgent:

    def __init__(self):

        self.alpha = 0.1       # Learning rate
        self.gamma = 0.9      # Discount factor
        self.epsilon = 1.0     # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        # Q-table:
        # Energy demand = 3
        # Occupancy = 2
        # Temperature = 2
        # Actions = 3

        self.q_table = np.zeros((3, 2, 2, 3))

    def choose_action(self, state):

        # Exploration
        if random.random() < self.epsilon:
            return random.randint(0, 2)

        # Exploitation
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):

        current_q = self.q_table[state][action]

        best_next_q = np.max(self.q_table[next_state])

        new_q = current_q + self.alpha * (
            reward + self.gamma * best_next_q - current_q
        )

        self.q_table[state][action] = new_q

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )


# ============================================================
# TRAINING
# ============================================================

env = SmartEnergyEnvironment()
agent = QLearningAgent()

episodes = 1000

total_rewards = []
total_energy = []
total_safety_violations = []
total_fairness_violations = []


for episode in range(episodes):

    state = env.reset()

    episode_reward = 0
    episode_energy = 0
    safety_count = 0
    fairness_count = 0

    for step in range(50):

        # Select action
        action = agent.choose_action(state)

        # Perform action
        next_state, reward, energy, \
        safety_violation, fairness_violation = env.step(action)

        # Update Q-table
        agent.update(
            state,
            action,
            reward,
            next_state
        )

        state = next_state

        episode_reward += reward
        episode_energy += energy

        if safety_violation:
            safety_count += 1

        if fairness_violation:
            fairness_count += 1

    # Reduce exploration
    agent.decay_epsilon()

    total_rewards.append(episode_reward)
    total_energy.append(episode_energy)
    total_safety_violations.append(safety_count)
    total_fairness_violations.append(fairness_count)


# ============================================================
# DISPLAY TRAINING RESULTS
# ============================================================

print("\n============================================")
print(" SMART ENERGY MANAGEMENT USING Q-LEARNING")
print("============================================")

print("\nTraining completed successfully.")

print("\nNumber of Episodes :", episodes)

print(
    "Average Reward     :",
    round(np.mean(total_rewards[-100:]), 2)
)

print(
    "Average Energy     :",
    round(np.mean(total_energy[-100:]), 2)
)

print(
    "Safety Violations  :",
    sum(total_safety_violations)
)

print(
    "Fairness Violations:",
    sum(total_fairness_violations)
)

print(
    "Final Epsilon      :",
    round(agent.epsilon, 4)
)


# ============================================================
# DISPLAY OPTIMAL POLICY
# ============================================================

print("\n============================================")
print(" OPTIMAL ENERGY MANAGEMENT POLICY")
print("============================================")

actions = [
    "Turn OFF unnecessary loads",
    "Normal operation",
    "Energy saving mode"
]

for energy in range(3):
    for occupancy in range(2):
        for temperature in range(2):

            state = (energy, occupancy, temperature)

            best_action = np.argmax(agent.q_table[state])

            print(
                "State:",
                state,
                " -> Action:",
                actions[best_action]
            )


# ============================================================
# TEST THE TRAINED AGENT
# ============================================================

print("\n============================================")
print(" TESTING TRAINED AGENT")
print("============================================")

agent.epsilon = 0

state = env.reset()

for step in range(10):

    action = agent.choose_action(state)

    next_state, reward, energy, \
    safety_violation, fairness_violation = env.step(action)

    print("\nStep:", step + 1)
    print("State:", state)
    print("Action:", actions[action])
    print("Energy Consumption:", energy)
    print("Reward:", reward)
    print("Safety Violation:", safety_violation)
    print("Fairness Violation:", fairness_violation)

    state = next_state

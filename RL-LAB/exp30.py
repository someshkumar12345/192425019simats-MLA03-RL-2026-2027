import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import matplotlib.pyplot as plt


# ============================================================
# 1. HIGHWAY ENVIRONMENT
# ============================================================

class HighwayEnvironment:

    def __init__(self):

        self.num_lanes = 3
        self.max_position = 100
        self.max_speed = 10

        # Actions
        # 0 = Keep Lane
        # 1 = Move Left
        # 2 = Move Right
        # 3 = Accelerate
        # 4 = Brake

        self.num_actions = 5

        self.reset()

    # --------------------------------------------------------
    # Reset environment
    # --------------------------------------------------------

    def reset(self):

        self.position = 0.0
        self.lane = 1
        self.speed = 5.0

        # Other vehicles
        self.vehicles = [
            {
                "lane": 1,
                "position": 25.0,
                "speed": 3.0
            },
            {
                "lane": 0,
                "position": 40.0,
                "speed": 6.0
            },
            {
                "lane": 2,
                "position": 35.0,
                "speed": 7.0
            },
            {
                "lane": 1,
                "position": 60.0,
                "speed": 4.0
            }
        ]

        self.done = False

        return self.get_state()

    # --------------------------------------------------------
    # Get state
    # --------------------------------------------------------

    def get_state(self):

        # Distance to nearest vehicle
        # in each lane.

        lane_distances = []

        for lane in range(
            self.num_lanes
        ):

            distances = []

            for vehicle in self.vehicles:

                if (
                    vehicle["lane"] == lane
                    and
                    vehicle["position"] >
                    self.position
                ):

                    distances.append(
                        vehicle["position"]
                        - self.position
                    )

            if len(distances) == 0:

                distance = 50.0

            else:

                distance = min(
                    distances
                )

            distance = min(
                distance,
                50.0
            )

            lane_distances.append(
                distance / 50.0
            )

        # State vector
        state = np.array([
            self.position /
            self.max_position,

            self.speed /
            self.max_speed,

            self.lane /
            (self.num_lanes - 1),

            lane_distances[0],
            lane_distances[1],
            lane_distances[2]

        ], dtype=np.float32)

        return state

    # --------------------------------------------------------
    # Perform action
    # --------------------------------------------------------

    def step(self, action):

        reward = 0.0
        collision = False

        old_position = self.position

        # ----------------------------------------------------
        # Lane control
        # ----------------------------------------------------

        if action == 0:

            # Keep lane
            pass

        elif action == 1:

            # Move left
            if self.lane > 0:

                self.lane -= 1

            else:

                reward -= 5

        elif action == 2:

            # Move right
            if self.lane < self.num_lanes - 1:

                self.lane += 1

            else:

                reward -= 5

        elif action == 3:

            # Accelerate
            self.speed += 1

        elif action == 4:

            # Brake
            self.speed -= 1

        # Keep speed within limits
        self.speed = np.clip(
            self.speed,
            0,
            self.max_speed
        )

        # ----------------------------------------------------
        # Move autonomous vehicle
        # ----------------------------------------------------

        self.position += self.speed

        # ----------------------------------------------------
        # Move other vehicles
        # ----------------------------------------------------

        for vehicle in self.vehicles:

            vehicle["position"] += (
                vehicle["speed"]
            )

        # ----------------------------------------------------
        # Collision detection
        # ----------------------------------------------------

        for vehicle in self.vehicles:

            distance = abs(
                vehicle["position"]
                - self.position
            )

            if (
                vehicle["lane"] == self.lane
                and distance < 3
            ):

                collision = True

                reward -= 100

                self.done = True

                break

        # ----------------------------------------------------
        # Reward for forward progress
        # ----------------------------------------------------

        progress = (
            self.position -
            old_position
        )

        reward += (
            progress * 0.5
        )

        # ----------------------------------------------------
        # Reward for reaching destination
        # ----------------------------------------------------

        if (
            self.position >=
            self.max_position
        ):

            reward += 100

            self.done = True

        # ----------------------------------------------------
        # Penalize very low speed
        # ----------------------------------------------------

        if self.speed < 2:

            reward -= 2

        # ----------------------------------------------------
        # Penalize unnecessary lane changes
        # ----------------------------------------------------

        if action in [1, 2]:

            reward -= 0.5

        return (
            self.get_state(),
            reward,
            self.done,
            {
                "collision": collision,
                "position": self.position,
                "lane": self.lane,
                "speed": self.speed
            }
        )


# ============================================================
# 2. REPLAY BUFFER
# ============================================================

class ReplayBuffer:

    def __init__(self, capacity=20000):

        self.buffer = deque(
            maxlen=capacity
        )

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    def sample(self, batch_size):

        batch = random.sample(
            self.buffer,
            batch_size
        )

        states = np.array([
            item[0]
            for item in batch
        ])

        actions = np.array([
            item[1]
            for item in batch
        ])

        rewards = np.array([
            item[2]
            for item in batch
        ])

        next_states = np.array([
            item[3]
            for item in batch
        ])

        dones = np.array([
            item[4]
            for item in batch
        ])

        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    def __len__(self):

        return len(self.buffer)


# ============================================================
# 3. DQN AGENT
# ============================================================

class DQNAgent:

    def __init__(
        self,
        state_size,
        action_size
    ):

        self.state_size = state_size
        self.action_size = action_size

        # ----------------------------------------------------
        # Hyperparameters
        # ----------------------------------------------------

        self.gamma = 0.95

        self.epsilon = 1.0

        self.epsilon_min = 0.05

        self.epsilon_decay = 0.995

        self.learning_rate = 0.001

        self.batch_size = 64

        # ----------------------------------------------------
        # Create neural networks
        # ----------------------------------------------------

        self.model = self.build_model()

        self.target_model = self.build_model()

        self.update_target_network()

        # ----------------------------------------------------
        # Experience replay
        # ----------------------------------------------------

        self.memory = ReplayBuffer(
            capacity=20000
        )

    # --------------------------------------------------------
    # Build DQN
    # --------------------------------------------------------

    def build_model(self):

        model = Sequential([

            Dense(
                64,
                activation="relu",
                input_shape=(
                    self.state_size,
                )
            ),

            Dense(
                64,
                activation="relu"
            ),

            Dense(
                self.action_size,
                activation="linear"
            )
        ])

        model.compile(
            optimizer=Adam(
                learning_rate=
                self.learning_rate
            ),
            loss="mse"
        )

        return model

    # --------------------------------------------------------
    # Update target network
    # --------------------------------------------------------

    def update_target_network(self):

        self.target_model.set_weights(
            self.model.get_weights()
        )

    # --------------------------------------------------------
    # Epsilon-greedy action selection
    # --------------------------------------------------------

    def choose_action(self, state):

        if (
            np.random.rand()
            <= self.epsilon
        ):

            return random.randrange(
                self.action_size
            )

        state = np.expand_dims(
            state,
            axis=0
        )

        q_values = self.model.predict(
            state,
            verbose=0
        )

        return np.argmax(
            q_values[0]
        )

    # --------------------------------------------------------
    # Store experience
    # --------------------------------------------------------

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.add(
            state,
            action,
            reward,
            next_state,
            done
        )

    # --------------------------------------------------------
    # Train DQN
    # --------------------------------------------------------

    def replay(self):

        if len(self.memory) < self.batch_size:

            return

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.memory.sample(
            self.batch_size
        )

        # Current Q-values
        current_q = self.model.predict(
            states,
            verbose=0
        )

        # Next Q-values from target network
        next_q = self.target_model.predict(
            next_states,
            verbose=0
        )

        # Bellman target
        for i in range(
            self.batch_size
        ):

            if dones[i]:

                target = rewards[i]

            else:

                target = (
                    rewards[i]
                    +
                    self.gamma
                    * np.max(
                        next_q[i]
                    )
                )

            current_q[i, actions[i]] = (
                target
            )

        # Train neural network
        self.model.fit(
            states,
            current_q,
            epochs=1,
            verbose=0
        )

        # Reduce exploration
        if self.epsilon > self.epsilon_min:

            self.epsilon *= (
                self.epsilon_decay
            )

            self.epsilon = max(
                self.epsilon,
                self.epsilon_min
            )


# ============================================================
# 4. TRAINING
# ============================================================

env = HighwayEnvironment()

agent = DQNAgent(
    state_size=6,
    action_size=5
)

episodes = 300
max_steps = 50

target_update_frequency = 10

reward_history = []

successful_trips = 0
collisions = 0


print("=" * 65)
print(" DQN AUTONOMOUS HIGHWAY DRIVING")
print("=" * 65)

print("\nTraining started...\n")


for episode in range(
    episodes
):

    state = env.reset()

    total_reward = 0

    for step in range(
        max_steps
    ):

        # Choose action
        action = agent.choose_action(
            state
        )

        # Execute action
        (
            next_state,
            reward,
            done,
            info
        ) = env.step(
            action
        )

        # Store experience
        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        # Train DQN
        agent.replay()

        state = next_state

        total_reward += reward

        if done:

            if info["collision"]:

                collisions += 1

            else:

                successful_trips += 1

            break

    # Update target network periodically
    if (
        (episode + 1)
        % target_update_frequency
        == 0
    ):

        agent.update_target_network()

    reward_history.append(
        total_reward
    )

    # Display progress
    if (
        (episode + 1)
        % 25 == 0
    ):

        average_reward = np.mean(
            reward_history[-25:]
        )

        print(
            f"Episode {episode + 1:3d} | "
            f"Average Reward: "
            f"{average_reward:8.2f} | "
            f"Epsilon: "
            f"{agent.epsilon:.3f}"
        )


# ============================================================
# 5. TRAINING RESULTS
# ============================================================

print("\n" + "=" * 65)
print(" TRAINING RESULTS")
print("=" * 65)

print(
    "Total Episodes     :",
    episodes
)

print(
    "Successful Trips   :",
    successful_trips
)

print(
    "Collisions         :",
    collisions
)

print(
    "Final Epsilon      :",
    round(
        agent.epsilon,
        4
    )
)

print(
    "Average Reward     :",
    round(
        np.mean(
            reward_history[-25:]
        ),
        2
    )
)


# ============================================================
# 6. EVALUATION
# ============================================================

print("\n" + "=" * 65)
print(" EVALUATION OF TRAINED DQN")
print("=" * 65)

# Disable exploration
agent.epsilon = 0

evaluation_episodes = 10

evaluation_success = 0
evaluation_collisions = 0

evaluation_rewards = []


action_names = [
    "Keep Lane",
    "Move Left",
    "Move Right",
    "Accelerate",
    "Brake"
]


for episode in range(
    evaluation_episodes
):

    state = env.reset()

    total_reward = 0

    path = []

    for step in range(
        max_steps
    ):

        # Select best learned action
        action = agent.choose_action(
            state
        )

        (
            next_state,
            reward,
            done,
            info
        ) = env.step(
            action
        )

        path.append(
            (
                info["position"],
                info["lane"],
                info["speed"],
                action_names[action]
            )
        )

        total_reward += reward

        state = next_state

        if done:

            if info["collision"]:

                evaluation_collisions += 1

            else:

                evaluation_success += 1

            break

    evaluation_rewards.append(
        total_reward
    )

    print(
        f"\nEvaluation Episode "
        f"{episode + 1}"
    )

    print(
        "Final Position:",
        round(
            info["position"],
            2
        )
    )

    print(
        "Final Lane:",
        info["lane"]
    )

    print(
        "Final Speed:",
        info["speed"]
    )

    print(
        "Total Reward:",
        round(
            total_reward,
            2
        )
    )

    if info["collision"]:

        print(
            "Result: COLLISION"
        )

    else:

        print(
            "Result: DESTINATION REACHED"
        )


# ============================================================
# 7. FINAL PERFORMANCE
# ============================================================

print("\n" + "=" * 65)
print(" FINAL PERFORMANCE")
print("=" * 65)

print(
    "Evaluation Episodes :",
    evaluation_episodes
)

print(
    "Successful Trips    :",
    evaluation_success
)

print(
    "Collisions          :",
    evaluation_collisions
)

print(
    "Success Rate        :",
    round(
        evaluation_success
        / evaluation_episodes
        * 100,
        2
    ),
    "%"
)

print(
    "Average Evaluation Reward:",
    round(
        np.mean(
            evaluation_rewards
        ),
        2
    )
)


# ============================================================
# 8. TRAINING REWARD GRAPH
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    reward_history
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Total Reward"
)

plt.title(
    "DQN Training Performance"
)

plt.grid(True)

plt.show()


# ============================================================
# 9. SMOOTHED REWARD GRAPH
# ============================================================

window = 20

if len(reward_history) >= window:

    smoothed_rewards = np.convolve(
        reward_history,
        np.ones(window) / window,
        mode="valid"
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        smoothed_rewards
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Average Reward"
    )

    plt.title(
        "Smoothed DQN Learning Curve"
    )

    plt.grid(True)

    plt.show()

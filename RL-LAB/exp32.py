import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense
)
from tensorflow.keras.optimizers import Adam
from collections import deque
import matplotlib.pyplot as plt


# ============================================================
# 1. GRIDWORLD ENVIRONMENT
# ============================================================

class GridWorld:

    def __init__(self):

        self.rows = 8
        self.cols = 8

        self.start = (0, 0)
        self.goal = (7, 7)

        # Obstacles
        self.obstacles = {
            (1, 1), (1, 2),
            (2, 2), (2, 5),
            (3, 1), (3, 3),
            (3, 4),
            (4, 4), (4, 6),
            (5, 1), (5, 2),
            (5, 6),
            (6, 3), (6, 4)
        }

        # Actions:
        # 0 = Up
        # 1 = Down
        # 2 = Left
        # 3 = Right

        self.action_size = 4

        self.max_steps = 100

        self.reset()

    # --------------------------------------------------------
    # Reset environment
    # --------------------------------------------------------

    def reset(self):

        self.position = self.start

        self.steps = 0

        return self.get_state()

    # --------------------------------------------------------
    # State representation
    # --------------------------------------------------------

    def get_state(self):

        state = np.zeros(
            self.rows * self.cols,
            dtype=np.float32
        )

        index = (
            self.position[0] * self.cols
            + self.position[1]
        )

        state[index] = 1.0

        return state

    # --------------------------------------------------------
    # Check valid position
    # --------------------------------------------------------

    def is_valid(self, position):

        row, col = position

        if row < 0 or row >= self.rows:
            return False

        if col < 0 or col >= self.cols:
            return False

        if position in self.obstacles:
            return False

        return True

    # --------------------------------------------------------
    # Execute action
    # --------------------------------------------------------

    def step(self, action):

        row, col = self.position

        old_position = self.position

        # Up
        if action == 0:
            new_position = (
                row - 1,
                col
            )

        # Down
        elif action == 1:
            new_position = (
                row + 1,
                col
            )

        # Left
        elif action == 2:
            new_position = (
                row,
                col - 1
            )

        # Right
        else:
            new_position = (
                row,
                col + 1
            )

        reward = -1

        # ----------------------------------------------------
        # Invalid movement
        # ----------------------------------------------------

        if not self.is_valid(new_position):

            new_position = old_position

            reward = -5

        else:

            self.position = new_position

        self.steps += 1

        done = False

        # ----------------------------------------------------
        # Goal
        # ----------------------------------------------------

        if self.position == self.goal:

            reward = 100

            done = True

        # ----------------------------------------------------
        # Maximum steps
        # ----------------------------------------------------

        elif self.steps >= self.max_steps:

            done = True

        return (
            self.get_state(),
            reward,
            done
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
# 3. STANDARD DQN
# ============================================================

class StandardDQN:

    def __init__(
        self,
        state_size,
        action_size
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.95

        self.epsilon = 1.0

        self.epsilon_min = 0.05

        self.epsilon_decay = 0.995

        self.learning_rate = 0.001

        self.batch_size = 64

        self.model = self.build_model()

        self.target_model = self.build_model()

        self.update_target_network()

        self.memory = ReplayBuffer()

    # --------------------------------------------------------
    # Standard DQN architecture
    # --------------------------------------------------------

    def build_model(self):

        model = tf.keras.Sequential([

            Dense(
                128,
                activation="relu",
                input_shape=(
                    self.state_size,
                )
            ),

            Dense(
                128,
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
    # Target network update
    # --------------------------------------------------------

    def update_target_network(self):

        self.target_model.set_weights(
            self.model.get_weights()
        )

    # --------------------------------------------------------
    # Action selection
    # --------------------------------------------------------

    def choose_action(self, state):

        if (
            np.random.rand()
            < self.epsilon
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

        current_q = self.model.predict(
            states,
            verbose=0
        )

        next_q = self.target_model.predict(
            next_states,
            verbose=0
        )

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

            current_q[
                i,
                actions[i]
            ] = target

        self.model.fit(
            states,
            current_q,
            epochs=1,
            verbose=0
        )

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon *
            self.epsilon_decay
        )


# ============================================================
# 4. DUELING DQN
# ============================================================

class DuelingDQN:

    def __init__(
        self,
        state_size,
        action_size
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.95

        self.epsilon = 1.0

        self.epsilon_min = 0.05

        self.epsilon_decay = 0.995

        self.learning_rate = 0.001

        self.batch_size = 64

        self.model = self.build_model()

        self.target_model = self.build_model()

        self.update_target_network()

        self.memory = ReplayBuffer()

    # --------------------------------------------------------
    # DUELING DQN ARCHITECTURE
    # --------------------------------------------------------

    def build_model(self):

        inputs = Input(
            shape=(self.state_size,)
        )

        # Shared feature layers
        x = Dense(
            128,
            activation="relu"
        )(inputs)

        x = Dense(
            128,
            activation="relu"
        )(x)

        # ----------------------------------------------------
        # VALUE STREAM
        # ----------------------------------------------------

        value = Dense(
            64,
            activation="relu"
        )(x)

        value = Dense(
            1,
            activation="linear"
        )(value)

        # ----------------------------------------------------
        # ADVANTAGE STREAM
        # ----------------------------------------------------

        advantage = Dense(
            64,
            activation="relu"
        )(x)

        advantage = Dense(
            self.action_size,
            activation="linear"
        )(advantage)

        # ----------------------------------------------------
        # DUELING AGGREGATION
        # ----------------------------------------------------

        def combine_streams(inputs):

            value, advantage = inputs

            return (
                value
                +
                (
                    advantage
                    -
                    tf.reduce_mean(
                        advantage,
                        axis=1,
                        keepdims=True
                    )
                )
            )

        q_values = tf.keras.layers.Lambda(
            combine_streams
        )(
            [value, advantage]
        )

        model = Model(
            inputs=inputs,
            outputs=q_values
        )

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
    # Action selection
    # --------------------------------------------------------

    def choose_action(self, state):

        if (
            np.random.rand()
            < self.epsilon
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
    # Train Dueling DQN
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

        current_q = self.model.predict(
            states,
            verbose=0
        )

        next_q = self.target_model.predict(
            next_states,
            verbose=0
        )

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

            current_q[
                i,
                actions[i]
            ] = target

        self.model.fit(
            states,
            current_q,
            epochs=1,
            verbose=0
        )

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon *
            self.epsilon_decay
        )


# ============================================================
# 5. TRAINING FUNCTION
# ============================================================

def train_agent(
    agent,
    environment,
    episodes=300
):

    reward_history = []

    success_history = []

    target_update_frequency = 10

    for episode in range(
        episodes
    ):

        state = environment.reset()

        total_reward = 0

        success = False

        for step in range(
            environment.max_steps
        ):

            action = agent.choose_action(
                state
            )

            (
                next_state,
                reward,
                done
            ) = environment.step(
                action
            )

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            agent.replay()

            state = next_state

            total_reward += reward

            if done:

                if environment.position == environment.goal:

                    success = True

                break

        if (
            (episode + 1)
            % target_update_frequency
            == 0
        ):

            agent.update_target_network()

        reward_history.append(
            total_reward
        )

        success_history.append(
            int(success)
        )

        if (
            (episode + 1)
            % 50
            == 0
        ):

            avg_reward = np.mean(
                reward_history[-50:]
            )

            success_rate = np.mean(
                success_history[-50:]
            ) * 100

            print(
                f"Episode {episode + 1:3d} | "
                f"Average Reward: "
                f"{avg_reward:7.2f} | "
                f"Success Rate: "
                f"{success_rate:6.2f}% | "
                f"Epsilon: "
                f"{agent.epsilon:.3f}"
            )

    return (
        reward_history,
        success_history
    )


# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate_agent(
    agent,
    environment,
    episodes=100
):

    # Disable exploration
    agent.epsilon = 0

    successes = 0

    total_rewards = []

    path_lengths = []

    for episode in range(
        episodes
    ):

        state = environment.reset()

        total_reward = 0

        path_length = 0

        for step in range(
            environment.max_steps
        ):

            action = agent.choose_action(
                state
            )

            (
                next_state,
                reward,
                done
            ) = environment.step(
                action
            )

            state = next_state

            total_reward += reward

            path_length += 1

            if done:

                break

        total_rewards.append(
            total_reward
        )

        path_lengths.append(
            path_length
        )

        if environment.position == environment.goal:

            successes += 1

    success_rate = (
        successes /
        episodes
        * 100
    )

    average_reward = np.mean(
        total_rewards
    )

    average_path_length = np.mean(
        path_lengths
    )

    return (
        success_rate,
        average_reward,
        average_path_length
    )


# ============================================================
# 7. TRAIN STANDARD DQN
# ============================================================

print("=" * 70)
print(" TRAINING STANDARD DQN")
print("=" * 70)

env_dqn = GridWorld()

dqn_agent = StandardDQN(
    state_size=64,
    action_size=4
)

dqn_rewards, dqn_success = train_agent(
    dqn_agent,
    env_dqn,
    episodes=300
)


# ============================================================
# 8. TRAIN DUELING DQN
# ============================================================

print("\n" + "=" * 70)
print(" TRAINING DUELING DQN")
print("=" * 70)

env_dueling = GridWorld()

dueling_agent = DuelingDQN(
    state_size=64,
    action_size=4
)

dueling_rewards, dueling_success = train_agent(
    dueling_agent,
    env_dueling,
    episodes=300
)


# ============================================================
# 9. EVALUATE STANDARD DQN
# ============================================================

print("\n" + "=" * 70)
print(" EVALUATING STANDARD DQN")
print("=" * 70)

dqn_results = evaluate_agent(
    dqn_agent,
    GridWorld(),
    episodes=100
)


print(
    "Success Rate:",
    f"{dqn_results[0]:.2f}%"
)

print(
    "Average Reward:",
    f"{dqn_results[1]:.2f}"
)

print(
    "Average Steps:",
    f"{dqn_results[2]:.2f}"
)


# ============================================================
# 10. EVALUATE DUELING DQN
# ============================================================

print("\n" + "=" * 70)
print(" EVALUATING DUELING DQN")
print("=" * 70)

dueling_results = evaluate_agent(
    dueling_agent,
    GridWorld(),
    episodes=100
)


print(
    "Success Rate:",
    f"{dueling_results[0]:.2f}%"
)

print(
    "Average Reward:",
    f"{dueling_results[1]:.2f}"
)

print(
    "Average Steps:",
    f"{dueling_results[2]:.2f}"
)


# ============================================================
# 11. FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print(" DQN VS DUELING DQN")
print("=" * 70)

print(
    f"{'Metric':<25}"
    f"{'Standard DQN':<20}"
    f"{'Dueling DQN':<20}"
)

print("-" * 65)

print(
    f"{'Success Rate':<25}"
    f"{dqn_results[0]:<20.2f}"
    f"{dueling_results[0]:<20.2f}"
)

print(
    f"{'Average Reward':<25}"
    f"{dqn_results[1]:<20.2f}"
    f"{dueling_results[1]:<20.2f}"
)

print(
    f"{'Average Steps':<25}"
    f"{dqn_results[2]:<20.2f}"
    f"{dueling_results[2]:<20.2f}"
)


# ============================================================
# 12. TRAINING REWARD COMPARISON
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    dqn_rewards,
    label="Standard DQN"
)

plt.plot(
    dueling_rewards,
    label="Dueling DQN"
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Total Reward"
)

plt.title(
    "DQN vs Dueling DQN Training Performance"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 13. SUCCESS RATE COMPARISON
# ============================================================

window = 20

dqn_success_rate = (
    np.convolve(
        dqn_success,
        np.ones(window) / window,
        mode="valid"
    ) * 100
)

dueling_success_rate = (
    np.convolve(
        dueling_success,
        np.ones(window) / window,
        mode="valid"
    ) * 100
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    dqn_success_rate,
    label="Standard DQN"
)

plt.plot(
    dueling_success_rate,
    label="Dueling DQN"
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Success Rate (%)"
)

plt.title(
    "Navigation Success Rate Comparison"
)

plt.legend()

plt.grid(True)

plt.show()

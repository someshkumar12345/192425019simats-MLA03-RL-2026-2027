import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


# ============================================================
# 1. SMART HOME ENVIRONMENT
# ============================================================

class SmartHomeEnvironment:

    def __init__(self):

        # Desired comfortable temperature
        self.target_temperature = 22.0

        # Temperature limits
        self.min_temperature = 15.0
        self.max_temperature = 30.0

        # Actions
        #
        # 0 -> Strong Cooling
        # 1 -> Mild Cooling
        # 2 -> Maintain Temperature
        # 3 -> Mild Heating
        # 4 -> Strong Heating

        self.action_size = 5

        self.max_steps = 50

        self.reset()

    # --------------------------------------------------------
    # RESET ENVIRONMENT
    # --------------------------------------------------------

    def reset(self):

        # Initial indoor temperature
        self.temperature = np.random.uniform(
            18,
            28
        )

        # Outdoor temperature changes
        self.outdoor_temperature = np.random.uniform(
            15,
            35
        )

        self.steps = 0

        return self.get_state()

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    def get_state(self):

        temperature_difference = (
            self.temperature -
            self.target_temperature
        )

        state = np.array([
            self.temperature / 40.0,

            self.outdoor_temperature / 40.0,

            temperature_difference / 15.0,

            self.steps / self.max_steps

        ], dtype=np.float32)

        return state

    # --------------------------------------------------------
    # ENVIRONMENT STEP
    # --------------------------------------------------------

    def step(self, action):

        self.steps += 1

        # ----------------------------------------------------
        # Heating/Cooling effect
        # ----------------------------------------------------

        temperature_change = {

            0: -1.5,   # Strong cooling
            1: -0.7,   # Mild cooling
            2:  0.0,   # Maintain
            3:  0.7,   # Mild heating
            4:  1.5    # Strong heating
        }

        control_effect = (
            temperature_change[action]
        )

        # ----------------------------------------------------
        # Outdoor temperature influence
        # ----------------------------------------------------

        outside_effect = (
            self.outdoor_temperature
            - self.temperature
        ) * 0.05

        # Small random disturbance
        weather_noise = np.random.normal(
            0,
            0.1
        )

        # Update indoor temperature
        self.temperature += (
            control_effect
            +
            outside_effect
            +
            weather_noise
        )

        # ----------------------------------------------------
        # Change outdoor temperature
        # ----------------------------------------------------

        self.outdoor_temperature += np.random.normal(
            0,
            0.3
        )

        self.outdoor_temperature = np.clip(
            self.outdoor_temperature,
            10,
            40
        )

        # ----------------------------------------------------
        # COMFORT PENALTY
        # ----------------------------------------------------

        temperature_error = abs(
            self.temperature
            - self.target_temperature
        )

        comfort_penalty = (
            temperature_error * 2
        )

        # ----------------------------------------------------
        # ENERGY COST
        # ----------------------------------------------------

        energy_cost = {

            0: 3.0,   # Strong cooling
            1: 1.5,   # Mild cooling
            2: 0.1,   # Maintain
            3: 1.5,   # Mild heating
            4: 3.0    # Strong heating
        }

        energy_penalty = (
            energy_cost[action]
        )

        # ----------------------------------------------------
        # TOTAL REWARD
        # ----------------------------------------------------

        reward = -(
            comfort_penalty
            +
            energy_penalty
        )

        # Extra reward for ideal comfort
        if temperature_error <= 0.5:

            reward += 5

        # ----------------------------------------------------
        # TERMINATION
        # ----------------------------------------------------

        done = False

        if self.steps >= self.max_steps:

            done = True

        return (
            self.get_state(),
            reward,
            done
        )


# ============================================================
# 2. REINFORCE POLICY NETWORK
# ============================================================

class REINFORCEAgent:

    def __init__(
        self,
        state_size,
        action_size
    ):

        self.state_size = state_size

        self.action_size = action_size

        self.gamma = 0.99

        self.learning_rate = 0.001

        # ----------------------------------------------------
        # Policy network
        # ----------------------------------------------------

        self.policy = Sequential([

            Dense(
                64,
                activation="relu",
                input_shape=(
                    state_size,
                )
            ),

            Dense(
                64,
                activation="relu"
            ),

            Dense(
                action_size,
                activation="softmax"
            )
        ])

        self.optimizer = Adam(
            learning_rate=
            self.learning_rate
        )

    # --------------------------------------------------------
    # SELECT ACTION
    # --------------------------------------------------------

    def choose_action(
        self,
        state
    ):

        state = np.expand_dims(
            state,
            axis=0
        )

        probabilities = (
            self.policy(
                state,
                training=False
            ).numpy()[0]
        )

        # Sample action according
        # to learned probability
        action = np.random.choice(
            self.action_size,
            p=probabilities
        )

        return (
            action,
            probabilities
        )

    # --------------------------------------------------------
    # REINFORCE UPDATE
    # --------------------------------------------------------

    def train_episode(
        self,
        states,
        actions,
        rewards
    ):

        # ====================================================
        # COMPUTE DISCOUNTED RETURNS
        # ====================================================

        returns = []

        discounted_return = 0

        for reward in reversed(
            rewards
        ):

            discounted_return = (
                reward
                +
                self.gamma
                * discounted_return
            )

            returns.insert(
                0,
                discounted_return
            )

        returns = np.array(
            returns,
            dtype=np.float32
        )

        # Normalize returns
        if len(returns) > 1:

            returns = (
                returns
                -
                np.mean(returns)
            ) / (
                np.std(returns)
                + 1e-8
            )

        # ====================================================
        # POLICY GRADIENT UPDATE
        # ====================================================

        states_tensor = tf.convert_to_tensor(
            np.array(states),
            dtype=tf.float32
        )

        actions_tensor = tf.convert_to_tensor(
            np.array(actions),
            dtype=tf.int32
        )

        returns_tensor = tf.convert_to_tensor(
            returns,
            dtype=tf.float32
        )

        with tf.GradientTape() as tape:

            probabilities = self.policy(
                states_tensor,
                training=True
            )

            # Create action indices
            indices = tf.stack(
                [
                    tf.range(
                        tf.shape(
                            actions_tensor
                        )[0]
                    ),
                    actions_tensor
                ],
                axis=1
            )

            selected_probabilities = (
                tf.gather_nd(
                    probabilities,
                    indices
                )
            )

            log_probabilities = tf.math.log(
                selected_probabilities
                + 1e-8
            )

            # REINFORCE loss
            loss = -tf.reduce_mean(
                log_probabilities
                * returns_tensor
            )

        gradients = tape.gradient(
            loss,
            self.policy.trainable_variables
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.policy.trainable_variables
            )
        )

        return loss.numpy()


# ============================================================
# 3. TRAINING
# ============================================================

environment = SmartHomeEnvironment()

agent = REINFORCEAgent(
    state_size=4,
    action_size=5
)

episodes = 500

reward_history = []

temperature_history = []

energy_history = []


print("=" * 65)
print(" SMART HOME TEMPERATURE CONTROL")
print(" REINFORCE ALGORITHM")
print("=" * 65)

print("\nTraining started...\n")


for episode in range(
    episodes
):

    state = environment.reset()

    states = []
    actions = []
    rewards = []

    episode_temperatures = []

    episode_energy = 0

    for step in range(
        environment.max_steps
    ):

        # ----------------------------------------------------
        # Select action
        # ----------------------------------------------------

        action, probabilities = (
            agent.choose_action(
                state
            )
        )

        # ----------------------------------------------------
        # Execute action
        # ----------------------------------------------------

        (
            next_state,
            reward,
            done
        ) = environment.step(
            action
        )

        # Store episode information
        states.append(
            state
        )

        actions.append(
            action
        )

        rewards.append(
            reward
        )

        episode_temperatures.append(
            environment.temperature
        )

        # Energy consumption
        energy_values = [
            3.0,
            1.5,
            0.1,
            1.5,
            3.0
        ]

        episode_energy += (
            energy_values[action]
        )

        state = next_state

        if done:

            break

    # --------------------------------------------------------
    # REINFORCE update after complete episode
    # --------------------------------------------------------

    loss = agent.train_episode(
        states,
        actions,
        rewards
    )

    total_reward = sum(
        rewards
    )

    reward_history.append(
        total_reward
    )

    temperature_history.append(
        np.mean(
            episode_temperatures
        )
    )

    energy_history.append(
        episode_energy
    )

    # --------------------------------------------------------
    # Display training progress
    # --------------------------------------------------------

    if (
        (episode + 1)
        % 50 == 0
    ):

        avg_reward = np.mean(
            reward_history[-50:]
        )

        avg_temperature = np.mean(
            temperature_history[-50:]
        )

        avg_energy = np.mean(
            energy_history[-50:]
        )

        print(
            f"Episode {episode + 1:3d} | "
            f"Average Reward: "
            f"{avg_reward:8.2f} | "
            f"Average Temperature: "
            f"{avg_temperature:5.2f}°C | "
            f"Energy: "
            f"{avg_energy:6.2f}"
        )


# ============================================================
# 4. TRAINING RESULTS
# ============================================================

print("\n" + "=" * 65)
print(" TRAINING COMPLETED")
print("=" * 65)

print(
    "Total Episodes:",
    episodes
)

print(
    "Final Average Reward:",
    round(
        np.mean(
            reward_history[-50:]
        ),
        2
    )
)

print(
    "Final Average Temperature:",
    round(
        np.mean(
            temperature_history[-50:]
        ),
        2
    ),
    "°C"
)

print(
    "Final Average Energy:",
    round(
        np.mean(
            energy_history[-50:]
        ),
        2
    )
)


# ============================================================
# 5. EVALUATION
# ============================================================

print("\n" + "=" * 65)
print(" EVALUATING TRAINED SMART HOME AGENT")
print("=" * 65)

evaluation_episodes = 20

evaluation_rewards = []

evaluation_temperatures = []

evaluation_energy = []


for episode in range(
    evaluation_episodes
):

    state = environment.reset()

    total_reward = 0

    temperatures = []

    energy_used = 0

    for step in range(
        environment.max_steps
    ):

        # ----------------------------------------------------
        # Get policy probabilities
        # ----------------------------------------------------

        state_input = np.expand_dims(
            state,
            axis=0
        )

        probabilities = (
            agent.policy(
                state_input,
                training=False
            ).numpy()[0]
        )

        # Select BEST action
        action = np.argmax(
            probabilities
        )

        (
            next_state,
            reward,
            done
        ) = environment.step(
            action
        )

        total_reward += reward

        temperatures.append(
            environment.temperature
        )

        energy_used += [
            3.0,
            1.5,
            0.1,
            1.5,
            3.0
        ][action]

        state = next_state

        if done:

            break

    evaluation_rewards.append(
        total_reward
    )

    evaluation_temperatures.append(
        np.mean(temperatures)
    )

    evaluation_energy.append(
        energy_used
    )

    print(
        f"Episode {episode + 1:2d} | "
        f"Reward: "
        f"{total_reward:8.2f} | "
        f"Avg Temp: "
        f"{np.mean(temperatures):5.2f}°C | "
        f"Energy: "
        f"{energy_used:6.2f}"
    )


# ============================================================
# 6. FINAL EVALUATION
# ============================================================

print("\n" + "=" * 65)
print(" FINAL EVALUATION")
print("=" * 65)

print(
    "Evaluation Episodes:",
    evaluation_episodes
)

print(
    "Average Reward:",
    round(
        np.mean(
            evaluation_rewards
        ),
        2
    )
)

print(
    "Average Temperature:",
    round(
        np.mean(
            evaluation_temperatures
        ),
        2
    ),
    "°C"
)

print(
    "Average Energy Consumption:",
    round(
        np.mean(
            evaluation_energy
        ),
        2
    )
)


# ============================================================
# 7. DISPLAY LEARNED POLICY
# ============================================================

print("\n" + "=" * 65)
print(" LEARNED TEMPERATURE CONTROL POLICY")
print("=" * 65)

test_temperatures = [
    16,
    18,
    20,
    21,
    22,
    23,
    25,
    27,
    30
]

for temperature in test_temperatures:

    state = np.array([
        temperature / 40.0,

        25.0 / 40.0,

        (
            temperature
            - environment.target_temperature
        ) / 15.0,

        0.5

    ], dtype=np.float32)

    state_input = np.expand_dims(
        state,
        axis=0
    )

    probabilities = (
        agent.policy(
            state_input,
            training=False
        ).numpy()[0]
    )

    best_action = np.argmax(
        probabilities
    )

    action_names = [
        "Strong Cooling",
        "Mild Cooling",
        "Maintain",
        "Mild Heating",
        "Strong Heating"
    ]

    print(
        f"Temperature: "
        f"{temperature:2d}°C -> "
        f"{action_names[best_action]}"
    )


# ============================================================
# 8. PLOT TRAINING REWARD
# ============================================================

plt.figure(
    figsize=(10, 6)
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
    "REINFORCE Training Reward"
)

plt.grid(True)

plt.show()


# ============================================================
# 9. PLOT TEMPERATURE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    temperature_history,
    label="Average Indoor Temperature"
)

plt.axhline(
    y=22,
    linestyle="--",
    label="Target Temperature (22°C)"
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Temperature (°C)"
)

plt.title(
    "Indoor Temperature During Training"
)

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 10. PLOT ENERGY CONSUMPTION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    energy_history
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Energy Consumption"
)

plt.title(
    "Energy Consumption During Training"
)

plt.grid(True)

plt.show()

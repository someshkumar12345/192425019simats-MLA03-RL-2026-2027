import numpy as np
import random
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

from collections import deque

import matplotlib.pyplot as plt


# ============================================================
# 1. SIMPLIFIED RTS ENVIRONMENT
# ============================================================

class RTSEnvironment:

    def __init__(self):

        self.max_steps = 100

        self.action_size = 3

        # Actions:
        # 0 -> Resource gathering
        # 1 -> Build units
        # 2 -> Attack

        self.reset()

    # --------------------------------------------------------
    # RESET GAME
    # --------------------------------------------------------

    def reset(self):

        self.resources = 50.0

        self.units = 2.0

        self.enemy_units = 20.0

        self.enemy_base = 100.0

        self.steps = 0

        return self.get_state()

    # --------------------------------------------------------
    # STATE REPRESENTATION
    # --------------------------------------------------------

    def get_state(self):

        state = np.array([
            self.resources / 200.0,
            self.units / 50.0,
            self.enemy_units / 50.0,
            self.enemy_base / 100.0,
            self.steps / self.max_steps
        ], dtype=np.float32)

        return state

    # --------------------------------------------------------
    # GAME STEP
    # --------------------------------------------------------

    def step(self, action):

        self.steps += 1

        reward = 0.0

        # Keep actions within [0,1]
        action = np.clip(
            action,
            0.0,
            1.0
        )

        gather_action = action[0]

        build_action = action[1]

        attack_action = action[2]

        # ====================================================
        # RESOURCE GATHERING
        # ====================================================

        gathered = (
            gather_action * 10
        )

        self.resources += gathered

        reward += gathered * 0.5

        # ====================================================
        # BUILD UNITS
        # ====================================================

        # Each unit costs 10 resources

        units_to_build = (
            build_action * 2
        )

        build_cost = (
            units_to_build * 10
        )

        if self.resources >= build_cost:

            self.resources -= build_cost

            self.units += units_to_build

            reward += units_to_build * 2

        else:

            # Penalize attempting to spend
            # unavailable resources.

            reward -= 2

        # ====================================================
        # ATTACK ENEMY
        # ====================================================

        if self.units > 0:

            attack_power = (
                attack_action
                * self.units
                * 0.8
            )

            # Enemy defense reduces attack damage

            damage = max(
                0,
                attack_power
                - self.enemy_units * 0.1
            )

            self.enemy_base -= damage

            reward += damage * 2

            # Enemy units are also reduced

            enemy_damage = (
                attack_action
                * self.units
                * 0.2
            )

            self.enemy_units = max(
                0,
                self.enemy_units
                - enemy_damage
            )

        # ====================================================
        # OPPONENT COUNTER ATTACK
        # ====================================================

        if self.enemy_units > 0:

            enemy_attack = (
                self.enemy_units * 0.05
            )

            self.units -= enemy_attack

            self.units = max(
                0,
                self.units
            )

            reward -= enemy_attack

        # ====================================================
        # RESOURCE LIMIT
        # ====================================================

        self.resources = min(
            self.resources,
            200
        )

        # ====================================================
        # WIN CONDITION
        # ====================================================

        done = False

        if self.enemy_base <= 0:

            reward += 200

            done = True

        # ====================================================
        # LOSE CONDITION
        # ====================================================

        elif self.units <= 0:

            reward -= 100

            done = True

        # ====================================================
        # TIME LIMIT
        # ====================================================

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

    def __init__(
        self,
        capacity=50000
    ):

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

    def sample(
        self,
        batch_size
    ):

        batch = random.sample(
            self.buffer,
            batch_size
        )

        states = np.array([
            x[0]
            for x in batch
        ])

        actions = np.array([
            x[1]
            for x in batch
        ])

        rewards = np.array([
            x[2]
            for x in batch
        ])

        next_states = np.array([
            x[3]
            for x in batch
        ])

        dones = np.array([
            x[4]
            for x in batch
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
# 3. ACTOR NETWORK
# ============================================================

def build_actor(
    state_size,
    action_size
):

    inputs = Input(
        shape=(state_size,)
    )

    x = Dense(
        128,
        activation="relu"
    )(inputs)

    x = Dense(
        128,
        activation="relu"
    )(x)

    # Sigmoid keeps continuous actions
    # between 0 and 1.

    outputs = Dense(
        action_size,
        activation="sigmoid"
    )(x)

    return Model(
        inputs,
        outputs
    )


# ============================================================
# 4. CRITIC NETWORK
# ============================================================

def build_critic(
    state_size,
    action_size
):

    state_input = Input(
        shape=(state_size,)
    )

    action_input = Input(
        shape=(action_size,)
    )

    # Process state
    state_features = Dense(
        128,
        activation="relu"
    )(state_input)

    # Combine state and action
    combined = tf.keras.layers.Concatenate()([
        state_features,
        action_input
    ])

    x = Dense(
        128,
        activation="relu"
    )(combined)

    x = Dense(
        64,
        activation="relu"
    )(x)

    output = Dense(
        1,
        activation="linear"
    )(x)

    return Model(
        [state_input, action_input],
        output
    )


# ============================================================
# 5. DDPG AGENT
# ============================================================

class DDPGAgent:

    def __init__(
        self,
        state_size,
        action_size
    ):

        self.state_size = state_size

        self.action_size = action_size

        self.gamma = 0.99

        self.tau = 0.005

        self.batch_size = 64

        self.actor_lr = 0.001

        self.critic_lr = 0.002

        # ----------------------------------------------------
        # Actor
        # ----------------------------------------------------

        self.actor = build_actor(
            state_size,
            action_size
        )

        self.target_actor = build_actor(
            state_size,
            action_size
        )

        # ----------------------------------------------------
        # Critic
        # ----------------------------------------------------

        self.critic = build_critic(
            state_size,
            action_size
        )

        self.target_critic = build_critic(
            state_size,
            action_size
        )

        # ----------------------------------------------------
        # Optimizers
        # ----------------------------------------------------

        self.actor_optimizer = Adam(
            learning_rate=self.actor_lr
        )

        self.critic_optimizer = Adam(
            learning_rate=self.critic_lr
        )

        # ----------------------------------------------------
        # Initialize target networks
        # ----------------------------------------------------

        self.target_actor.set_weights(
            self.actor.get_weights()
        )

        self.target_critic.set_weights(
            self.critic.get_weights()
        )

        # ----------------------------------------------------
        # Replay buffer
        # ----------------------------------------------------

        self.memory = ReplayBuffer()

        # ----------------------------------------------------
        # Exploration noise
        # ----------------------------------------------------

        self.noise_std = 0.2

    # ========================================================
    # SELECT ACTION
    # ========================================================

    def choose_action(
        self,
        state,
        training=True
    ):

        state = np.expand_dims(
            state,
            axis=0
        )

        action = self.actor(
            state,
            training=False
        ).numpy()[0]

        # Add Gaussian exploration noise

        if training:

            noise = np.random.normal(
                0,
                self.noise_std,
                size=self.action_size
            )

            action += noise

        return np.clip(
            action,
            0,
            1
        )

    # ========================================================
    # STORE EXPERIENCE
    # ========================================================

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

    # ========================================================
    # TRAIN NETWORKS
    # ========================================================

    def train(self):

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

        states = tf.convert_to_tensor(
            states,
            dtype=tf.float32
        )

        actions = tf.convert_to_tensor(
            actions,
            dtype=tf.float32
        )

        rewards = tf.convert_to_tensor(
            rewards,
            dtype=tf.float32
        )

        next_states = tf.convert_to_tensor(
            next_states,
            dtype=tf.float32
        )

        dones = tf.convert_to_tensor(
            dones,
            dtype=tf.float32
        )

        # ====================================================
        # CRITIC UPDATE
        # ====================================================

        with tf.GradientTape() as tape:

            next_actions = self.target_actor(
                next_states,
                training=True
            )

            target_q = self.target_critic(
                [
                    next_states,
                    next_actions
                ],
                training=True
            )

            target = (
                rewards
                +
                self.gamma
                * (1 - dones)
                * tf.squeeze(
                    target_q,
                    axis=1
                )
            )

            current_q = self.critic(
                [
                    states,
                    actions
                ],
                training=True
            )

            critic_loss = tf.reduce_mean(
                tf.square(
                    target
                    -
                    tf.squeeze(
                        current_q,
                        axis=1
                    )
                )
            )

        critic_gradients = tape.gradient(
            critic_loss,
            self.critic.trainable_variables
        )

        self.critic_optimizer.apply_gradients(
            zip(
                critic_gradients,
                self.critic.trainable_variables
            )
        )

        # ====================================================
        # ACTOR UPDATE
        # ====================================================

        with tf.GradientTape() as tape:

            predicted_actions = self.actor(
                states,
                training=True
            )

            actor_q = self.critic(
                [
                    states,
                    predicted_actions
                ],
                training=True
            )

            actor_loss = -tf.reduce_mean(
                actor_q
            )

        actor_gradients = tape.gradient(
            actor_loss,
            self.actor.trainable_variables
        )

        self.actor_optimizer.apply_gradients(
            zip(
                actor_gradients,
                self.actor.trainable_variables
            )
        )

        # ====================================================
        # SOFT TARGET UPDATE
        # ====================================================

        self.soft_update(
            self.actor,
            self.target_actor
        )

        self.soft_update(
            self.critic,
            self.target_critic
        )

    # ========================================================
    # SOFT UPDATE
    # ========================================================

    def soft_update(
        self,
        source,
        target
    ):

        for source_weight, target_weight in zip(
            source.weights,
            target.weights
        ):

            target_weight.assign(
                self.tau * source_weight
                +
                (1 - self.tau)
                * target_weight
            )


# ============================================================
# 6. TRAINING
# ============================================================

environment = RTSEnvironment()

agent = DDPGAgent(
    state_size=5,
    action_size=3
)

episodes = 300

reward_history = []

print("=" * 65)
print(" DDPG RTS GAME AGENT")
print("=" * 65)

print("\nTraining started...\n")


for episode in range(
    episodes
):

    state = environment.reset()

    total_reward = 0

    for step in range(
        environment.max_steps
    ):

        # Select continuous action
        action = agent.choose_action(
            state,
            training=True
        )

        # Environment transition
        (
            next_state,
            reward,
            done
        ) = environment.step(
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

        # Train DDPG
        agent.train()

        state = next_state

        total_reward += reward

        if done:

            break

    reward_history.append(
        total_reward
    )

    # Gradually reduce exploration noise
    agent.noise_std = max(
        0.05,
        agent.noise_std * 0.995
    )

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
            f"Noise: "
            f"{agent.noise_std:.3f}"
        )


# ============================================================
# 7. TRAINING RESULTS
# ============================================================

print("\n" + "=" * 65)
print(" TRAINING COMPLETED")
print("=" * 65)

print(
    "Episodes:",
    episodes
)

print(
    "Final Average Reward:",
    round(
        np.mean(
            reward_history[-25:]
        ),
        2
    )
)


# ============================================================
# 8. EVALUATION
# ============================================================

print("\n" + "=" * 65)
print(" EVALUATING TRAINED AGENT")
print("=" * 65)

evaluation_episodes = 20

wins = 0
losses = 0

evaluation_rewards = []


for episode in range(
    evaluation_episodes
):

    state = environment.reset()

    total_reward = 0

    for step in range(
        environment.max_steps
    ):

        # No exploration during evaluation
        action = agent.choose_action(
            state,
            training=False
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

        if done:

            break

    evaluation_rewards.append(
        total_reward
    )

    if environment.enemy_base <= 0:

        wins += 1

        result = "WIN"

    else:

        losses += 1

        result = "LOSS"

    print(
        f"Game {episode + 1:2d} | "
        f"Reward: {total_reward:8.2f} | "
        f"Resources: {environment.resources:6.2f} | "
        f"Units: {environment.units:6.2f} | "
        f"Enemy Base: {environment.enemy_base:6.2f} | "
        f"{result}"
    )


# ============================================================
# 9. FINAL PERFORMANCE
# ============================================================

print("\n" + "=" * 65)
print(" FINAL PERFORMANCE")
print("=" * 65)

print(
    "Evaluation Games:",
    evaluation_episodes
)

print(
    "Wins:",
    wins
)

print(
    "Losses:",
    losses
)

print(
    "Win Rate:",
    round(
        wins /
        evaluation_episodes *
        100,
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
# 10. DEMONSTRATE LEARNED STRATEGY
# ============================================================

print("\n" + "=" * 65)
print(" SAMPLE GAME - LEARNED DDPG POLICY")
print("=" * 65)

state = environment.reset()

for step in range(20):

    action = agent.choose_action(
        state,
        training=False
    )

    (
        next_state,
        reward,
        done
    ) = environment.step(
        action
    )

    print(
        f"Step {step + 1:2d} | "
        f"Gather: {action[0]:.2f} | "
        f"Build: {action[1]:.2f} | "
        f"Attack: {action[2]:.2f} | "
        f"Resources: {environment.resources:6.2f} | "
        f"Units: {environment.units:6.2f} | "
        f"Enemy Base: {environment.enemy_base:6.2f}"
    )

    state = next_state

    if done:

        break


# ============================================================
# 11. PLOT TRAINING REWARD
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
    "DDPG RTS Agent Training Performance"
)

plt.grid(True)

plt.show()


# ============================================================
# 12. SMOOTHED REWARD
# ============================================================

window = 20

if len(reward_history) >= window:

    smoothed = np.convolve(
        reward_history,
        np.ones(window) / window,
        mode="valid"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        smoothed
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Average Reward"
    )

    plt.title(
        "Smoothed DDPG Learning Curve"
    )

    plt.grid(True)

    plt.show()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

STATE_SIZE = 6
ACTION_SIZE = 3
MAX_STEPS = 100


class LaneEnv:

    def __init__(self):
        self.reset()

    def reset(self):
        self.position = 0.0
        self.velocity = 1.0
        self.heading = 0.0
        self.previous_steering = 0.0
        self.steps = 0
        return self.state()

    def state(self):
        return np.array([
            self.position,
            self.velocity,
            self.heading,
            self.heading * 0.5,
            abs(self.position),
            self.previous_steering
        ], dtype=np.float32)

    def step(self, action):

        self.steps += 1

        if action == 0:
            steering = -0.1
        elif action == 1:
            steering = 0.0
        else:
            steering = 0.1

        self.previous_steering = steering

        self.heading += steering

        self.heading *= 0.95

        self.position += (
            self.velocity *
            np.sin(self.heading) *
            0.1
        )

        self.position *= 0.98

        center_reward = 5 - abs(self.position) * 5

        stability_reward = 3 - abs(self.heading) * 3

        movement_reward = 2

        steering_penalty = abs(steering) * 2

        reward = (
            center_reward
            + stability_reward
            + movement_reward
            - steering_penalty
        )

        done = False

        if abs(self.position) > 1.0:
            reward -= 50
            done = True

        if self.steps >= MAX_STEPS:
            done = True

        return self.state(), reward, done


class PolicyNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(STATE_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_SIZE)
        )

    def forward(self, state):

        return torch.softmax(
            self.network(state),
            dim=-1
        )


class ActorCritic(nn.Module):

    def __init__(self):

        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(STATE_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.actor = nn.Linear(
            64,
            ACTION_SIZE
        )

        self.critic = nn.Linear(
            64,
            1
        )

    def forward(self, state):

        x = self.shared(state)

        policy = torch.softmax(
            self.actor(x),
            dim=-1
        )

        value = self.critic(x)

        return policy, value


def train_reinforce(episodes=500):

    env = LaneEnv()

    model = PolicyNetwork()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    gamma = 0.99

    history = []

    for episode in range(episodes):

        state = env.reset()

        log_probs = []
        rewards = []

        done = False

        while not done:

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            )

            probabilities = model(
                state_tensor
            )

            distribution = Categorical(
                probabilities
            )

            action = distribution.sample()

            next_state, reward, done = env.step(
                action.item()
            )

            log_probs.append(
                distribution.log_prob(action)
            )

            rewards.append(reward)

            state = next_state

        returns = []

        G = 0

        for reward in reversed(rewards):

            G = reward + gamma * G

            returns.insert(0, G)

        returns = torch.tensor(
            returns,
            dtype=torch.float32
        )

        returns = (
            returns - returns.mean()
        ) / (
            returns.std() + 1e-8
        )

        loss = 0

        for log_prob, G in zip(
            log_probs,
            returns
        ):

            loss += -log_prob * G

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        history.append(sum(rewards))

    return model, history


def train_actor_critic(episodes=500):

    env = LaneEnv()

    model = ActorCritic()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    gamma = 0.99

    history = []

    for episode in range(episodes):

        state = env.reset()

        log_probs = []
        values = []
        rewards = []

        done = False

        while not done:

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            )

            policy, value = model(
                state_tensor
            )

            distribution = Categorical(
                policy
            )

            action = distribution.sample()

            next_state, reward, done = env.step(
                action.item()
            )

            log_probs.append(
                distribution.log_prob(action)
            )

            values.append(value.squeeze())

            rewards.append(reward)

            state = next_state

        returns = []

        G = 0

        for reward in reversed(rewards):

            G = reward + gamma * G

            returns.insert(0, G)

        returns = torch.tensor(
            returns,
            dtype=torch.float32
        )

        values = torch.stack(values)

        advantages = (
            returns - values.detach()
        )

        actor_loss = 0

        for log_prob, advantage in zip(
            log_probs,
            advantages
        ):

            actor_loss += (
                -log_prob * advantage
            )

        critic_loss = (
            returns - values
        ).pow(2).mean()

        loss = (
            actor_loss
            + 0.5 * critic_loss
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        history.append(sum(rewards))

    return model, history


def train_ppo(episodes=500):

    env = LaneEnv()

    model = ActorCritic()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0003
    )

    gamma = 0.99
    clip = 0.2

    history = []

    for episode in range(episodes):

        state = env.reset()

        states = []
        actions = []
        rewards = []
        old_log_probs = []

        done = False

        while not done:

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            )

            policy, _ = model(
                state_tensor
            )

            distribution = Categorical(
                policy
            )

            action = distribution.sample()

            next_state, reward, done = env.step(
                action.item()
            )

            states.append(state)

            actions.append(action)

            rewards.append(reward)

            old_log_probs.append(
                distribution.log_prob(action).detach()
            )

            state = next_state

        returns = []

        G = 0

        for reward in reversed(rewards):

            G = reward + gamma * G

            returns.insert(0, G)

        states = torch.tensor(
            np.array(states),
            dtype=torch.float32
        )

        actions = torch.stack(actions)

        returns = torch.tensor(
            returns,
            dtype=torch.float32
        )

        old_log_probs = torch.stack(
            old_log_probs
        )

        for _ in range(5):

            policies, values = model(
                states
            )

            distribution = Categorical(
                policies
            )

            new_log_probs = distribution.log_prob(
                actions
            )

            advantages = (
                returns - values.squeeze()
            ).detach()

            ratio = torch.exp(
                new_log_probs -
                old_log_probs
            )

            clipped_ratio = torch.clamp(
                ratio,
                1 - clip,
                1 + clip
            )

            actor_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()

            critic_loss = (
                returns -
                values.squeeze()
            ).pow(2).mean()

            loss = (
                actor_loss +
                0.5 * critic_loss
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        history.append(sum(rewards))

    return model, history


print("Training REINFORCE")

reinforce_model, reinforce_history = \
    train_reinforce()

print("REINFORCE Training Completed")


print("\nTraining Actor-Critic")

ac_model, ac_history = \
    train_actor_critic()

print("Actor-Critic Training Completed")


print("\nTraining PPO")

ppo_model, ppo_history = \
    train_ppo()

print("PPO Training Completed")


print("\nFinal Results")

print(
    "REINFORCE:",
    round(np.mean(reinforce_history[-50:]), 2)
)

print(
    "Actor-Critic:",
    round(np.mean(ac_history[-50:]), 2)
)

print(
    "PPO:",
    round(np.mean(ppo_history[-50:]), 2)
)

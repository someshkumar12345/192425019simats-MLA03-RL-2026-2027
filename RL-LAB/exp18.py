import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

GRID_SIZE = 5
STATE_SIZE = 4
ACTION_SIZE = 4
MAX_STEPS = 30


class RobotTask:

    def __init__(self, target):
        self.target = target
        self.reset()

    def reset(self):
        self.position = [0, 0]
        self.steps = 0
        return self.state()

    def state(self):
        return np.array([
            self.position[0] / 4,
            self.position[1] / 4,
            self.target[0] / 4,
            self.target[1] / 4
        ], dtype=np.float32)

    def step(self, action):

        self.steps += 1

        x, y = self.position

        if action == 0:
            y += 1

        elif action == 1:
            y -= 1

        elif action == 2:
            x -= 1

        elif action == 3:
            x += 1

        x = max(0, min(4, x))
        y = max(0, min(4, y))

        self.position = [x, y]

        distance = (
            abs(x - self.target[0])
            + abs(y - self.target[1])
        )

        if self.position == list(self.target):

            return self.state(), 100, True

        reward = -1

        if distance <= 2:
            reward += 5

        elif distance >= 5:
            reward -= 2

        done = self.steps >= MAX_STEPS

        return self.state(), reward, done


class MetaPolicy(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(STATE_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_SIZE)
        )

    def forward(self, state):

        return torch.softmax(
            self.network(state),
            dim=-1
        )


def collect_episode(
    model,
    task
):

    state = task.reset()

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

        distribution = torch.distributions.Categorical(
            probabilities
        )

        action = distribution.sample()

        next_state, reward, done = task.step(
            action.item()
        )

        log_probs.append(
            distribution.log_prob(action)
        )

        rewards.append(reward)

        state = next_state

    return log_probs, rewards


def calculate_loss(
    model,
    task
):

    log_probs, rewards = collect_episode(
        model,
        task
    )

    returns = []

    G = 0

    for reward in reversed(rewards):

        G = reward + 0.99 * G

        returns.insert(0, G)

    returns = torch.tensor(
        returns,
        dtype=torch.float32
    )

    if len(returns) > 1:

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

    return loss


def meta_train(
    model,
    tasks,
    iterations=500
):

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for iteration in range(iterations):

        meta_loss = 0

        for target in tasks:

            task = RobotTask(target)

            loss = calculate_loss(
                model,
                task
            )

            meta_loss += loss

        meta_loss /= len(tasks)

        optimizer.zero_grad()

        meta_loss.backward()

        optimizer.step()

        if (iteration + 1) % 50 == 0:

            print(
                "Meta Iteration:",
                iteration + 1,
                "Loss:",
                round(
                    meta_loss.item(),
                    3
                )
            )


def adapt(
    model,
    new_task,
    episodes=5
):

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.005
    )

    for episode in range(episodes):

        loss = calculate_loss(
            model,
            new_task
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        print(
            "Adaptation Episode:",
            episode + 1,
            "Loss:",
            round(
                loss.item(),
                3
            )
        )


def evaluate(
    model,
    task
):

    state = task.reset()

    path = [tuple(task.position)]

    total_reward = 0

    for _ in range(MAX_STEPS):

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32
        )

        with torch.no_grad():

            probabilities = model(
                state_tensor
            )

        action = torch.argmax(
            probabilities
        ).item()

        state, reward, done = task.step(
            action
        )

        total_reward += reward

        path.append(
            tuple(task.position)
        )

        if done:
            break

    success = (
        task.position ==
        list(task.target)
    )

    return (
        total_reward,
        path,
        success
    )


training_tasks = [
    (4, 4),
    (4, 3),
    (3, 4),
    (3, 3)
]

model = MetaPolicy()

print("Meta-RL Training")

meta_train(
    model,
    training_tasks,
    iterations=500
)

print("\nMeta Training Completed")

new_task = RobotTask(
    (1, 4)
)

print("\nBefore Adaptation")

reward, path, success = evaluate(
    model,
    new_task
)

print(
    "Reward:",
    reward
)

print(
    "Success:",
    success
)

print("\nAdapting to New Task")

adapt(
    model,
    new_task,
    episodes=5
)

print("\nAfter Adaptation")

reward, path, success = evaluate(
    model,
    new_task
)

print(
    "Reward:",
    reward
)

print(
    "Path:",
    path
)

print(
    "Success:",
    success
)

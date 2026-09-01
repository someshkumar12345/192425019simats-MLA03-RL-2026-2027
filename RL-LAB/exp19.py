import numpy as np
import random

GRID_SIZE = 5
NUM_ROBOTS = 3
NUM_TASKS = 3

START_POSITIONS = [
    (0, 0),
    (2, 2),
    (4, 0)
]

TASK_POSITIONS = [
    (4, 4),
    (0, 4),
    (3, 2)
]

OBSTACLES = {
    (0, 3),
    (1, 1),
    (3, 3)
}

ACTIONS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
    4: (0, 0)
}


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Warehouse:

    def __init__(self):
        self.reset()

    def reset(self):

        self.positions = START_POSITIONS.copy()

        self.tasks = TASK_POSITIONS.copy()

        self.completed = [False] * NUM_TASKS

        self.assignments = [-1] * NUM_ROBOTS

        return self.get_states()

    def allocate_tasks(self):

        available = [
            i for i in range(NUM_TASKS)
            if not self.completed[i]
        ]

        for robot in range(NUM_ROBOTS):

            if self.assignments[robot] != -1:
                task = self.assignments[robot]

                if not self.completed[task]:
                    continue

            if not available:
                self.assignments[robot] = -1
                continue

            best_task = min(
                available,
                key=lambda t:
                distance(
                    self.positions[robot],
                    self.tasks[t]
                )
            )

            self.assignments[robot] = best_task

            available.remove(best_task)

    def get_states(self):

        states = []

        for robot in range(NUM_ROBOTS):

            x, y = self.positions[robot]

            task = self.assignments[robot]

            if task == -1:

                states.append(
                    (x, y, -1, -1)
                )

            else:

                tx, ty = self.tasks[task]

                states.append(
                    (x, y, tx, ty)
                )

        return states

    def step(self, actions):

        old_positions = self.positions.copy()

        rewards = [-1] * NUM_ROBOTS

        new_positions = []

        for robot in range(NUM_ROBOTS):

            x, y = self.positions[robot]

            dx, dy = ACTIONS[actions[robot]]

            nx = x + dx
            ny = y + dy

            if (
                nx < 0 or
                nx >= GRID_SIZE or
                ny < 0 or
                ny >= GRID_SIZE or
                (nx, ny) in OBSTACLES
            ):

                nx, ny = x, y

                rewards[robot] -= 10

            new_positions.append(
                (nx, ny)
            )

        for i in range(NUM_ROBOTS):

            for j in range(i + 1, NUM_ROBOTS):

                if new_positions[i] == new_positions[j]:

                    rewards[i] -= 50
                    rewards[j] -= 50

                    new_positions[i] = old_positions[i]
                    new_positions[j] = old_positions[j]

        self.positions = new_positions

        for robot in range(NUM_ROBOTS):

            task = self.assignments[robot]

            if task == -1:
                continue

            target = self.tasks[task]

            old_distance = distance(
                old_positions[robot],
                target
            )

            new_distance = distance(
                self.positions[robot],
                target
            )

            if new_distance < old_distance:
                rewards[robot] += 5

            elif new_distance > old_distance:
                rewards[robot] -= 2

            if self.positions[robot] == target:

                if not self.completed[task]:

                    self.completed[task] = True

                    rewards[robot] += 100

        done = all(self.completed)

        return self.get_states(), rewards, done


class MultiAgentQLearning:

    def __init__(self):

        self.q_tables = [
            np.zeros(
                (
                    GRID_SIZE,
                    GRID_SIZE,
                    GRID_SIZE,
                    GRID_SIZE,
                    5
                )
            )
            for _ in range(NUM_ROBOTS)
        ]

        self.alpha = 0.1
        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def choose_action(self, robot, state):

        x, y, tx, ty = state

        if random.random() < self.epsilon:

            return random.randint(0, 4)

        if tx == -1:

            return 4

        return np.argmax(
            self.q_tables[robot][x, y, tx, ty]
        )

    def update(
        self,
        robot,
        state,
        action,
        reward,
        next_state
    ):

        x, y, tx, ty = state

        nx, ny, ntx, nty = next_state

        current = self.q_tables[
            robot
        ][x, y, tx, ty, action]

        if ntx == -1:

            target = reward

        else:

            future = np.max(
                self.q_tables[
                    robot
                ][nx, ny, ntx, nty]
            )

            target = (
                reward +
                self.gamma * future
            )

        self.q_tables[
            robot
        ][x, y, tx, ty, action] += (
            self.alpha *
            (target - current)
        )


env = Warehouse()

agent = MultiAgentQLearning()

episodes = 1000

reward_history = []

success_history = []

for episode in range(episodes):

    states = env.reset()

    env.allocate_tasks()

    total_reward = 0

    for step in range(100):

        states = env.get_states()

        actions = []

        for robot in range(NUM_ROBOTS):

            action = agent.choose_action(
                robot,
                states[robot]
            )

            actions.append(action)

        next_states, rewards, done = env.step(
            actions
        )

        env.allocate_tasks()

        for robot in range(NUM_ROBOTS):

            agent.update(
                robot,
                states[robot],
                actions[robot],
                rewards[robot],
                next_states[robot]
            )

        total_reward += sum(rewards)

        if done:
            break

    agent.epsilon = max(
        agent.epsilon_min,
        agent.epsilon *
        agent.epsilon_decay
    )

    reward_history.append(
        total_reward
    )

    success_history.append(
        int(done)
    )

    if (episode + 1) % 100 == 0:

        success_rate = (
            np.mean(
                success_history[-100:]
            ) * 100
        )

        print(
            "Episode:",
            episode + 1,
            "Reward:",
            round(total_reward, 2),
            "Success Rate:",
            round(success_rate, 2),
            "%",
            "Epsilon:",
            round(agent.epsilon, 3)
        )


print("\nTraining Completed")

print(
    "Average Reward:",
    round(
        np.mean(reward_history[-100:]),
        2
    )
)

print(
    "Final Success Rate:",
    round(
        np.mean(success_history[-100:]) * 100,
        2
    ),
    "%"
)


states = env.reset()

env.allocate_tasks()

print("\nFinal Task Allocation")

for robot in range(NUM_ROBOTS):

    task = env.assignments[robot]

    print(
        "Robot",
        robot + 1,
        "-> Task",
        task + 1
        if task != -1
        else "None"
    )

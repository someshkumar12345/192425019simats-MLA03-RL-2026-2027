import numpy as np
import random

GRID_SIZE = 5

ACTIONS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

OBSTACLES = {
    (0, 2),
    (1, 1),
    (2, 4),
    (3, 3)
}

MAX_STEPS = 50


class RescuePOMDP:

    def __init__(self):
        self.reset()

    def reset(self):

        self.robot = (4, 0)

        possible_positions = [
            (r, c)
            for r in range(GRID_SIZE)
            for c in range(GRID_SIZE)
            if (r, c) not in OBSTACLES
            and (r, c) != self.robot
        ]

        self.victim = random.choice(
            possible_positions
        )

        self.belief = np.ones(
            (GRID_SIZE, GRID_SIZE),
            dtype=np.float32
        )

        for r, c in OBSTACLES:
            self.belief[r, c] = 0

        self.belief[self.robot] = 0

        self.normalize_belief()

        self.steps = 0
        self.found = False

        return self.get_state()

    def normalize_belief(self):

        total = np.sum(self.belief)

        if total > 0:
            self.belief /= total

    def get_state(self):

        max_position = np.unravel_index(
            np.argmax(self.belief),
            self.belief.shape
        )

        return (
            self.robot[0],
            self.robot[1],
            max_position[0],
            max_position[1]
        )

    def get_observation(self):

        distance = (
            abs(
                self.robot[0]
                - self.victim[0]
            )
            +
            abs(
                self.robot[1]
                - self.victim[1]
            )
        )

        if self.robot == self.victim:

            return 2

        if distance <= 1:

            if random.random() < 0.8:
                return 1

        if random.random() < 0.1:
            return 1

        return 0

    def update_belief(self, observation):

        for r in range(GRID_SIZE):

            for c in range(GRID_SIZE):

                if (r, c) in OBSTACLES:
                    self.belief[r, c] = 0
                    continue

                distance = (
                    abs(self.robot[0] - r)
                    +
                    abs(self.robot[1] - c)
                )

                if observation == 1:

                    if distance <= 1:
                        self.belief[r, c] *= 4

                    else:
                        self.belief[r, c] *= 0.5

                elif observation == 0:

                    if distance <= 1:
                        self.belief[r, c] *= 0.2

                    else:
                        self.belief[r, c] *= 1.0

        self.belief[self.robot] = 0

        self.normalize_belief()

    def step(self, action):

        self.steps += 1

        reward = -1

        if action < 4:

            dr, dc = ACTIONS[action]

            nr = self.robot[0] + dr
            nc = self.robot[1] + dc

            if (
                nr < 0
                or nr >= GRID_SIZE
                or nc < 0
                or nc >= GRID_SIZE
                or (nr, nc) in OBSTACLES
            ):

                reward = -20

            else:

                old_distance = (
                    abs(
                        self.robot[0]
                        - self.victim[0]
                    )
                    +
                    abs(
                        self.robot[1]
                        - self.victim[1]
                    )
                )

                self.robot = (nr, nc)

                new_distance = (
                    abs(
                        self.robot[0]
                        - self.victim[0]
                    )
                    +
                    abs(
                        self.robot[1]
                        - self.victim[1]
                    )
                )

                if new_distance < old_distance:
                    reward += 5

                elif new_distance > old_distance:
                    reward -= 2

                if self.robot == self.victim:

                    reward += 100
                    self.found = True

        else:

            observation = self.get_observation()

            if observation == 2:

                reward = 100
                self.found = True

            elif observation == 1:

                reward = 20

            else:

                reward = -3

            self.update_belief(
                observation
            )

        done = (
            self.found
            or self.steps >= MAX_STEPS
        )

        return (
            self.get_state(),
            reward,
            done
        )


class POMDPAgent:

    def __init__(self):

        self.q_table = {}

        self.alpha = 0.1
        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def get_q_values(self, state):

        if state not in self.q_table:

            self.q_table[state] = np.zeros(5)

        return self.q_table[state]

    def choose_action(self, state):

        if random.random() < self.epsilon:

            return random.randint(0, 4)

        return np.argmax(
            self.get_q_values(state)
        )

    def update(
        self,
        state,
        action,
        reward,
        next_state
    ):

        current = self.get_q_values(
            state
        )[action]

        next_value = np.max(
            self.get_q_values(
                next_state
            )
        )

        target = (
            reward
            + self.gamma * next_value
        )

        self.q_table[state][action] += (
            self.alpha
            * (target - current)
        )


env = RescuePOMDP()

agent = POMDPAgent()

episodes = 1000

reward_history = []
success_history = []

for episode in range(episodes):

    state = env.reset()

    total_reward = 0
    done = False

    while not done:

        action = agent.choose_action(
            state
        )

        next_state, reward, done = \
            env.step(action)

        agent.update(
            state,
            action,
            reward,
            next_state
        )

        state = next_state

        total_reward += reward

    agent.epsilon = max(
        agent.epsilon_min,
        agent.epsilon *
        agent.epsilon_decay
    )

    reward_history.append(
        total_reward
    )

    success_history.append(
        int(env.found)
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
        np.mean(
            success_history[-100:]
        ) * 100,
        2
    ),
    "%"
)


print("\nTesting Trained Robot")

state = env.reset()

path = [env.robot]

done = False

while not done:

    action = np.argmax(
        agent.get_q_values(state)
    )

    next_state, reward, done = \
        env.step(action)

    path.append(env.robot)

    state = next_state

print(
    "\nRobot Path:"
)

print(path)

print(
    "\nActual Victim Location:",
    env.victim
)

print(
    "Victim Found:",
    env.found
)

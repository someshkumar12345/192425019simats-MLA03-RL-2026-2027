import numpy as np
import random

GRID_SIZE = 5

START = (0, 0)
OBJECT = (2, 2)
TARGET = (4, 3)
CHARGE = (4, 4)

OBSTACLES = {
    (0, 3),
    (1, 1),
    (1, 3),
    (3, 1),
    (3, 3)
}

ACTIONS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}


class HouseholdRobot:

    def __init__(self):
        self.reset()

    def reset(self):

        self.position = START
        self.object_picked = False
        self.object_placed = False

        return self.state()

    def state(self):

        return (
            self.position[0],
            self.position[1],
            int(self.object_picked),
            int(self.object_placed)
        )

    def move(self, action):

        dr, dc = ACTIONS[action]

        nr = self.position[0] + dr
        nc = self.position[1] + dc

        if nr < 0 or nr >= GRID_SIZE:
            return -10

        if nc < 0 or nc >= GRID_SIZE:
            return -10

        if (nr, nc) in OBSTACLES:
            return -20

        self.position = (nr, nc)

        return -1

    def pick(self):

        if (
            self.position == OBJECT
            and not self.object_picked
        ):

            self.object_picked = True

            return 20

        return -10

    def place(self):

        if (
            self.position == TARGET
            and self.object_picked
        ):

            self.object_picked = False
            self.object_placed = True

            return 50

        return -10


def distance(a, b):

    return (
        abs(a[0] - b[0])
        + abs(a[1] - b[1])
    )


class MAXQ:

    def __init__(self):

        self.q = {}

        self.alpha = 0.1
        self.gamma = 0.95

    def get_q(self, state, action):

        key = (state, action)

        if key not in self.q:
            self.q[key] = 0.0

        return self.q[key]

    def update(
        self,
        state,
        action,
        reward,
        next_state
    ):

        current = self.get_q(
            state,
            action
        )

        next_values = [
            self.get_q(
                next_state,
                a
            )
            for a in range(4)
        ]

        target = reward + self.gamma * max(
            next_values
        )

        self.q[(state, action)] += (
            self.alpha *
            (target - current)
        )

    def navigate(
        self,
        robot,
        destination
    ):

        total_reward = 0

        for _ in range(50):

            state = robot.state()

            best_action = None
            best_distance = float("inf")

            for action, move in ACTIONS.items():

                nr = (
                    robot.position[0]
                    + move[0]
                )

                nc = (
                    robot.position[1]
                    + move[1]
                )

                if (
                    nr < 0
                    or nr >= GRID_SIZE
                    or nc < 0
                    or nc >= GRID_SIZE
                ):
                    continue

                if (nr, nc) in OBSTACLES:
                    continue

                d = distance(
                    (nr, nc),
                    destination
                )

                if d < best_distance:
                    best_distance = d
                    best_action = action

            if best_action is None:
                break

            reward = robot.move(
                best_action
            )

            total_reward += reward

            next_state = robot.state()

            self.update(
                state,
                best_action,
                reward,
                next_state
            )

            if robot.position == destination:
                break

        return total_reward

    def get_object(self, robot):

        reward = self.navigate(
            robot,
            OBJECT
        )

        reward += robot.pick()

        return reward

    def deliver_object(self, robot):

        reward = self.navigate(
            robot,
            TARGET
        )

        reward += robot.place()

        return reward

    def charge(self, robot):

        return self.navigate(
            robot,
            CHARGE
        )

    def execute(self, robot):

        total_reward = 0

        total_reward += self.get_object(
            robot
        )

        total_reward += self.deliver_object(
            robot
        )

        total_reward += self.charge(
            robot
        )

        return total_reward


class HAM:

    def __init__(self):

        self.robot = HouseholdRobot()

    def navigate(self, destination):

        path = []

        for _ in range(50):

            current = self.robot.position

            if current == destination:
                break

            best_action = None
            best_distance = float("inf")

            for action, move in ACTIONS.items():

                nr = current[0] + move[0]
                nc = current[1] + move[1]

                if nr < 0 or nr >= GRID_SIZE:
                    continue

                if nc < 0 or nc >= GRID_SIZE:
                    continue

                if (nr, nc) in OBSTACLES:
                    continue

                d = distance(
                    (nr, nc),
                    destination
                )

                if d < best_distance:
                    best_distance = d
                    best_action = action

            if best_action is None:
                break

            self.robot.move(best_action)

            path.append(
                self.robot.position
            )

        return path

    def execute(self):

        total_reward = 0

        object_path = self.navigate(
            OBJECT
        )

        total_reward -= len(object_path)

        total_reward += self.robot.pick()

        target_path = self.navigate(
            TARGET
        )

        total_reward -= len(target_path)

        total_reward += self.robot.place()

        charge_path = self.navigate(
            CHARGE
        )

        total_reward -= len(charge_path)

        return (
            total_reward,
            object_path,
            target_path,
            charge_path
        )


print("Training MAXQ")

maxq = MAXQ()

maxq_rewards = []

for episode in range(500):

    robot = HouseholdRobot()

    reward = maxq.execute(
        robot
    )

    maxq_rewards.append(reward)

    if (episode + 1) % 100 == 0:

        print(
            "MAXQ Episode:",
            episode + 1,
            "Reward:",
            reward
        )


print("\nTraining HAM")

ham_rewards = []

for episode in range(500):

    ham = HAM()

    reward, object_path, target_path, charge_path = \
        ham.execute()

    ham_rewards.append(reward)

    if (episode + 1) % 100 == 0:

        print(
            "HAM Episode:",
            episode + 1,
            "Reward:",
            reward
        )


print("\nTraining Completed")

print(
    "\nMAXQ Average Reward:",
    round(
        np.mean(maxq_rewards[-50:]),
        2
    )
)

print(
    "HAM Average Reward:",
    round(
        np.mean(ham_rewards[-50:]),
        2
    )
)

print("\nHAM Object Path:")
print(object_path)

print("\nHAM Target Path:")
print(target_path)

print("\nHAM Charging Path:")
print(charge_path)

if ham.object_placed:
    print("\nObject Successfully Delivered")

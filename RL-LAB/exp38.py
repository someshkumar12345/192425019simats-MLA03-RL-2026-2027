import numpy as np
import random
import matplotlib.pyplot as plt


# ============================================================
# POMDP ROBOT NAVIGATION ENVIRONMENT
# ============================================================

class RobotPOMDP:

    def __init__(self):

        # ----------------------------------------------------
        # Grid size
        # ----------------------------------------------------

        self.rows = 6
        self.cols = 6

        # Start and goal
        self.start = (0, 0)
        self.goal = (5, 5)

        # Obstacles
        self.obstacles = {
            (1, 1),
            (1, 2),
            (2, 4),
            (3, 1),
            (3, 2),
            (4, 3)
        }

        # ----------------------------------------------------
        # Actions
        # ----------------------------------------------------

        # 0 = Up
        # 1 = Down
        # 2 = Left
        # 3 = Right

        self.actions = [
            "UP",
            "DOWN",
            "LEFT",
            "RIGHT"
        ]

        self.num_actions = 4

        # ----------------------------------------------------
        # Possible observations
        # ----------------------------------------------------

        # 0 = No nearby obstacle
        # 1 = Obstacle nearby
        # 2 = Goal detected

        self.num_observations = 3

        # Sensor accuracy
        self.sensor_accuracy = 0.85

        # Maximum steps
        self.max_steps = 50

        self.reset()

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.position = self.start

        self.steps = 0

        self.done = False

        return self.position

    # ========================================================
    # VALID POSITION
    # ========================================================

    def is_valid(self, position):

        row, col = position

        if row < 0 or row >= self.rows:
            return False

        if col < 0 or col >= self.cols:
            return False

        if position in self.obstacles:
            return False

        return True

    # ========================================================
    # TRUE TRANSITION
    # ========================================================

    def transition(self, position, action):

        row, col = position

        if action == 0:
            next_position = (
                row - 1,
                col
            )

        elif action == 1:
            next_position = (
                row + 1,
                col
            )

        elif action == 2:
            next_position = (
                row,
                col - 1
            )

        else:
            next_position = (
                row,
                col + 1
            )

        # Invalid movement
        if not self.is_valid(
            next_position
        ):

            return position

        return next_position

    # ========================================================
    # OBSERVATION MODEL
    # ========================================================

    def get_observation(self):

        # Goal has a special observation

        if self.position == self.goal:

            return 2

        # Check nearby cells

        row, col = self.position

        nearby_obstacle = False

        for action in range(
            self.num_actions
        ):

            next_position = self.transition(
                self.position,
                action
            )

            if next_position == self.position:

                # Movement blocked
                nearby_obstacle = True

                break

        # Sensor uncertainty
        random_value = np.random.random()

        if nearby_obstacle:

            if random_value < self.sensor_accuracy:

                return 1

            else:

                return 0

        else:

            if random_value < self.sensor_accuracy:

                return 0

            else:

                return 1

    # ========================================================
    # ENVIRONMENT STEP
    # ========================================================

    def step(self, action):

        old_position = self.position

        self.position = self.transition(
            self.position,
            action
        )

        self.steps += 1

        # Default movement penalty
        reward = -1

        # Collision / blocked movement
        if self.position == old_position:

            reward = -5

        # Goal
        if self.position == self.goal:

            reward = 100

            self.done = True

        # Maximum steps
        elif self.steps >= self.max_steps:

            self.done = True

        observation = self.get_observation()

        return (
            observation,
            reward,
            self.done
        )


# ============================================================
# POMDP BELIEF STATE
# ============================================================

class BeliefState:

    def __init__(
        self,
        environment
    ):

        self.env = environment

        self.num_states = (
            environment.rows
            *
            environment.cols
        )

        # ----------------------------------------------------
        # Valid states
        # ----------------------------------------------------

        self.states = []

        for row in range(
            environment.rows
        ):

            for col in range(
                environment.cols
            ):

                position = (
                    row,
                    col
                )

                if environment.is_valid(
                    position
                ):

                    self.states.append(
                        position
                    )

        self.num_states = len(
            self.states
        )

        # Initially assume the robot
        # is at the known start position.

        self.belief = np.zeros(
            self.num_states
        )

        start_index = self.states.index(
            environment.start
        )

        self.belief[
            start_index
        ] = 1.0

    # ========================================================
    # GET STATE INDEX
    # ========================================================

    def state_index(
        self,
        position
    ):

        return self.states.index(
            position
        )

    # ========================================================
    # TRANSITION MODEL
    # ========================================================

    def predict(self, action):

        new_belief = np.zeros(
            self.num_states
        )

        for i, state in enumerate(
            self.states
        ):

            probability = (
                self.belief[i]
            )

            if probability == 0:

                continue

            next_state = (
                self.env.transition(
                    state,
                    action
                )
            )

            next_index = (
                self.state_index(
                    next_state
                )
            )

            new_belief[
                next_index
            ] += probability

        self.belief = new_belief

    # ========================================================
    # OBSERVATION UPDATE
    # ========================================================

    def update(
        self,
        observation
    ):

        updated_belief = np.zeros(
            self.num_states
        )

        for i, state in enumerate(
            self.states
        ):

            probability = (
                self.belief[i]
            )

            if probability == 0:

                continue

            # Calculate probability of
            # observation given state.

            expected_observation = (
                self.get_expected_observation(
                    state
                )
            )

            if observation == expected_observation:

                likelihood = (
                    self.env.sensor_accuracy
                )

            else:

                likelihood = (
                    1
                    -
                    self.env.sensor_accuracy
                )

            updated_belief[i] = (
                probability
                *
                likelihood
            )

        # Normalize belief
        total = np.sum(
            updated_belief
        )

        if total > 0:

            updated_belief /= total

        else:

            updated_belief = (
                np.ones(
                    self.num_states
                )
                /
                self.num_states
            )

        self.belief = (
            updated_belief
        )

    # ========================================================
    # EXPECTED OBSERVATION
    # ========================================================

    def get_expected_observation(
        self,
        state
    ):

        if state == self.env.goal:

            return 2

        row, col = state

        nearby_obstacle = False

        for action in range(
            self.env.num_actions
        ):

            next_position = (
                self.env.transition(
                    state,
                    action
                )
            )

            if next_position == state:

                nearby_obstacle = True

                break

        if nearby_obstacle:

            return 1

        return 0

    # ========================================================
    # MOST LIKELY POSITION
    # ========================================================

    def get_estimated_position(self):

        index = np.argmax(
            self.belief
        )

        return self.states[index]

    # ========================================================
    # BELIEF UNCERTAINTY
    # ========================================================

    def entropy(self):

        nonzero = (
            self.belief[
                self.belief > 0
            ]
        )

        return -np.sum(
            nonzero
            *
            np.log(
                nonzero
            )
        )


# ============================================================
# POMDP NAVIGATION POLICY
# ============================================================

class POMDPPolicy:

    def __init__(
        self,
        environment
    ):

        self.env = environment

    # ========================================================
    # MANHATTAN DISTANCE
    # ========================================================

    def distance(
        self,
        position
    ):

        return (
            abs(
                position[0]
                -
                self.env.goal[0]
            )
            +
            abs(
                position[1]
                -
                self.env.goal[1]
            )
        )

    # ========================================================
    # ACTION SELECTION
    # ========================================================

    def choose_action(
        self,
        belief_state
    ):

        estimated_position = (
            belief_state
            .get_estimated_position()
        )

        row, col = (
            estimated_position
        )

        goal_row, goal_col = (
            self.env.goal
        )

        candidate_actions = []

        # ----------------------------------------------------
        # Determine useful directions
        # ----------------------------------------------------

        if goal_row < row:

            candidate_actions.append(0)

        if goal_row > row:

            candidate_actions.append(1)

        if goal_col < col:

            candidate_actions.append(2)

        if goal_col > col:

            candidate_actions.append(3)

        # If no direct action exists,
        # consider all actions.

        if not candidate_actions:

            candidate_actions = list(
                range(
                    self.env.num_actions
                )
            )

        # ----------------------------------------------------
        # Evaluate candidate actions
        # ----------------------------------------------------

        best_action = None

        best_score = float(
            "-inf"
        )

        for action in range(
            self.env.num_actions
        ):

            # Calculate expected next state
            # using the current belief.

            score = 0

            for i, state in enumerate(
                belief_state.states
            ):

                probability = (
                    belief_state.belief[i]
                )

                if probability == 0:

                    continue

                next_state = (
                    self.env.transition(
                        state,
                        action
                    )
                )

                # Distance to goal
                distance_score = (
                    -self.distance(
                        next_state
                    )
                )

                # Penalize blocked movements
                if next_state == state:

                    distance_score -= 5

                score += (
                    probability
                    *
                    distance_score
                )

            # Small preference for candidate
            # directions toward goal.

            if action in candidate_actions:

                score += 2

            # Encourage information gathering
            # when uncertainty is high.

            if (
                belief_state.entropy()
                > 1.5
            ):

                # Prefer actions that move
                # into new areas.

                score += 0.2

            if score > best_score:

                best_score = score

                best_action = action

        return best_action


# ============================================================
# RUN SINGLE POMDP EPISODE
# ============================================================

def run_episode(
    environment,
    policy,
    verbose=False
):

    environment.reset()

    belief = BeliefState(
        environment
    )

    total_reward = 0

    actual_path = []

    estimated_path = []

    observations = []

    belief_uncertainty = []

    for step in range(
        environment.max_steps
    ):

        # ----------------------------------------------------
        # Estimate current location
        # ----------------------------------------------------

        estimated_position = (
            belief.get_estimated_position()
        )

        # ----------------------------------------------------
        # Select action
        # ----------------------------------------------------

        action = policy.choose_action(
            belief
        )

        # ----------------------------------------------------
        # Environment transition
        # ----------------------------------------------------

        (
            observation,
            reward,
            done
        ) = environment.step(
            action
        )

        # ----------------------------------------------------
        # Belief prediction
        # ----------------------------------------------------

        belief.predict(
            action
        )

        # ----------------------------------------------------
        # Belief correction
        # ----------------------------------------------------

        belief.update(
            observation
        )

        # Store information
        actual_path.append(
            environment.position
        )

        estimated_path.append(
            estimated_position
        )

        observations.append(
            observation
        )

        belief_uncertainty.append(
            belief.entropy()
        )

        total_reward += reward

        if verbose:

            print(
                f"Step {step + 1:2d} | "
                f"Actual: {environment.position} | "
                f"Estimated: {estimated_position} | "
                f"Action: {environment.actions[action]:5s} | "
                f"Observation: {observation} | "
                f"Reward: {reward:4d}"
            )

        if done:

            break

    success = (
        environment.position
        ==
        environment.goal
    )

    return {
        "reward": total_reward,
        "steps": step + 1,
        "success": success,
        "actual_path": actual_path,
        "estimated_path": estimated_path,
        "observations": observations,
        "entropy": belief_uncertainty
    }


# ============================================================
# RUN MULTIPLE SCENARIOS
# ============================================================

def evaluate_scenarios():

    scenarios = [

        {
            "name":
            "High Sensor Accuracy",
            "accuracy":
            0.95
        },

        {
            "name":
            "Normal Sensor Accuracy",
            "accuracy":
            0.85
        },

        {
            "name":
            "Low Sensor Accuracy",
            "accuracy":
            0.65
        }
    ]

    results = []

    print("=" * 75)
    print(" POMDP AUTONOMOUS ROBOT NAVIGATION")
    print("=" * 75)

    for scenario in scenarios:

        environment = (
            RobotPOMDP()
        )

        environment.sensor_accuracy = (
            scenario["accuracy"]
        )

        policy = POMDPPolicy(
            environment
        )

        rewards = []

        steps = []

        successes = 0

        for episode in range(
            100
        ):

            result = run_episode(
                environment,
                policy
            )

            rewards.append(
                result["reward"]
            )

            steps.append(
                result["steps"]
            )

            if result["success"]:

                successes += 1

        success_rate = (
            successes /
            100
            * 100
        )

        results.append({

            "scenario":
                scenario["name"],

            "accuracy":
                scenario["accuracy"],

            "success_rate":
                success_rate,

            "average_reward":
                np.mean(rewards),

            "average_steps":
                np.mean(steps)
        })

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print(" SCENARIO COMPARISON")
    print("=" * 75)

    print(
        f"{'Scenario':<25}"
        f"{'Accuracy':<12}"
        f"{'Success %':<15}"
        f"{'Avg Reward':<15}"
        f"{'Avg Steps':<12}"
    )

    print("-" * 75)

    for result in results:

        print(
            f"{result['scenario']:<25}"
            f"{result['accuracy']:<12.2f}"
            f"{result['success_rate']:<15.2f}"
            f"{result['average_reward']:<15.2f}"
            f"{result['average_steps']:<12.2f}"
        )

    return results


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Evaluate different sensor scenarios
    # --------------------------------------------------------

    results = evaluate_scenarios()

    # --------------------------------------------------------
    # Demonstrate one episode
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print(" SAMPLE POMDP NAVIGATION")
    print("=" * 75)

    environment = RobotPOMDP()

    policy = POMDPPolicy(
        environment
    )

    result = run_episode(
        environment,
        policy,
        verbose=True
    )

    print("\n" + "=" * 75)
    print(" SAMPLE EPISODE RESULT")
    print("=" * 75)

    print(
        "Destination:",
        environment.goal
    )

    print(
        "Final Position:",
        environment.position
    )

    print(
        "Steps:",
        result["steps"]
    )

    print(
        "Total Reward:",
        result["reward"]
    )

    print(
        "Success:",
        result["success"]
    )

    # --------------------------------------------------------
    # Plot actual vs estimated position
    # --------------------------------------------------------

    actual_rows = [
        position[0]
        for position
        in result["actual_path"]
    ]

    actual_cols = [
        position[1]
        for position
        in result["actual_path"]
    ]

    estimated_rows = [
        position[0]
        for position
        in result["estimated_path"]
    ]

    estimated_cols = [
        position[1]
        for position
        in result["estimated_path"]
    ]

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        actual_cols,
        actual_rows,
        marker="o",
        label="Actual Path"
    )

    plt.plot(
        estimated_cols,
        estimated_rows,
        marker="x",
        linestyle="--",
        label="Estimated Path"
    )

    # Plot obstacles
    for obstacle in environment.obstacles:

        plt.scatter(
            obstacle[1],
            obstacle[0],
            marker="s",
            s=100
        )

    # Start
    plt.scatter(
        environment.start[1],
        environment.start[0],
        marker="o",
        s=150,
        label="Start"
    )

    # Goal
    plt.scatter(
        environment.goal[1],
        environment.goal[0],
        marker="*",
        s=200,
        label="Goal"
    )

    plt.gca().invert_yaxis()

    plt.xticks(
        range(environment.cols)
    )

    plt.yticks(
        range(environment.rows)
    )

    plt.xlabel(
        "Column"
    )

    plt.ylabel(
        "Row"
    )

    plt.title(
        "POMDP Robot Navigation: "
        "Actual vs Estimated Position"
    )

    plt.grid(True)

    plt.legend()

    plt.show()

    # --------------------------------------------------------
    # Plot belief uncertainty
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        result["entropy"],
        marker="o"
    )

    plt.xlabel(
        "Time Step"
    )

    plt.ylabel(
        "Belief Entropy"
    )

    plt.title(
        "Robot Localization Uncertainty"
    )

    plt.grid(True)

    plt.show()

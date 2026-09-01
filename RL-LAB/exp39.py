import numpy as np
import random
import matplotlib.pyplot as plt


# ============================================================
# HEALTHCARE MANAGEMENT ENVIRONMENT
# ============================================================

class HealthcareEnvironment:

    def __init__(self):

        # ----------------------------------------------------
        # Patient conditions
        # ----------------------------------------------------

        # 0 = Stable
        # 1 = Moderate
        # 2 = Critical

        self.num_conditions = 3

        self.condition_names = {
            0: "Stable",
            1: "Moderate",
            2: "Critical"
        }

        # ----------------------------------------------------
        # Resource levels
        # ----------------------------------------------------

        # 0 = Low
        # 1 = Medium
        # 2 = High

        self.num_resources = 3

        self.resource_names = {
            0: "Low",
            1: "Medium",
            2: "High"
        }

        # ----------------------------------------------------
        # Waiting-time levels
        # ----------------------------------------------------

        # 0 = Short
        # 1 = Medium
        # 2 = Long

        self.num_waiting_levels = 3

        self.waiting_names = {
            0: "Short",
            1: "Medium",
            2: "Long"
        }

        # ----------------------------------------------------
        # Actions
        # ----------------------------------------------------

        # 0 = Routine treatment
        # 1 = Priority treatment
        # 2 = Allocate additional resources
        # 3 = Send to ICU

        self.actions = [
            "Routine Treatment",
            "Priority Treatment",
            "Allocate Resources",
            "ICU Treatment"
        ]

        self.num_actions = len(
            self.actions
        )

        # Maximum number of time steps
        self.max_steps = 50

        self.reset()

    # ========================================================
    # RESET ENVIRONMENT
    # ========================================================

    def reset(self):

        # Random initial patient condition
        self.condition = random.choice(
            [0, 1, 2]
        )

        # Initial hospital resources
        self.resources = 1

        # Initial waiting time
        self.waiting = 0

        # Patient outcome
        # 0 = not completed
        # 1 = recovered
        # -1 = deteriorated

        self.outcome = 0

        self.steps = 0

        self.done = False

        return self.get_state()

    # ========================================================
    # STATE
    # ========================================================

    def get_state(self):

        return (
            self.condition,
            self.resources,
            min(
                self.waiting,
                2
            )
        )

    # ========================================================
    # STATE INDEX
    # ========================================================

    def state_to_index(self, state):

        condition, resources, waiting = state

        return (
            condition * 9
            +
            resources * 3
            +
            waiting
        )

    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):

        self.steps += 1

        reward = 0

        # ----------------------------------------------------
        # RESOURCE REGENERATION
        # ----------------------------------------------------

        # Resources occasionally become available.

        if random.random() < 0.30:

            self.resources = min(
                2,
                self.resources + 1
            )

        # ====================================================
        # ACTION 0: ROUTINE TREATMENT
        # ====================================================

        if action == 0:

            if self.resources >= 1:

                self.resources -= 1

                # Treatment success probability
                success_probability = {
                    0: 0.95,
                    1: 0.70,
                    2: 0.40
                }

                if (
                    random.random()
                    <
                    success_probability[
                        self.condition
                    ]
                ):

                    self.outcome = 1

                    reward += 40

                    self.done = True

                else:

                    # Patient condition can worsen
                    if random.random() < 0.30:

                        self.condition = min(
                            2,
                            self.condition + 1
                        )

                    reward -= 5

            else:

                # Not enough resources
                reward -= 10

                self.waiting += 1

        # ====================================================
        # ACTION 1: PRIORITY TREATMENT
        # ====================================================

        elif action == 1:

            if self.resources >= 1:

                self.resources -= 1

                # Priority treatment is more effective
                success_probability = {
                    0: 0.98,
                    1: 0.85,
                    2: 0.65
                }

                if (
                    random.random()
                    <
                    success_probability[
                        self.condition
                    ]
                ):

                    self.outcome = 1

                    reward += 50

                    self.done = True

                else:

                    self.condition = min(
                        2,
                        self.condition + 1
                    )

                    reward -= 5

            else:

                reward -= 12

                self.waiting += 1

        # ====================================================
        # ACTION 2: ALLOCATE RESOURCES
        # ====================================================

        elif action == 2:

            # Allocate additional hospital resources

            if self.resources < 2:

                self.resources += 1

                reward -= 3

            else:

                reward -= 6

        # ====================================================
        # ACTION 3: ICU TREATMENT
        # ====================================================

        elif action == 3:

            if self.resources >= 2:

                self.resources -= 2

                success_probability = {
                    0: 0.99,
                    1: 0.95,
                    2: 0.85
                }

                if (
                    random.random()
                    <
                    success_probability[
                        self.condition
                    ]
                ):

                    self.outcome = 1

                    reward += 60

                    self.done = True

                else:

                    self.condition = min(
                        2,
                        self.condition + 1
                    )

                    reward -= 10

            else:

                reward -= 15

                self.waiting += 1

        # ====================================================
        # WAITING-TIME PENALTY
        # ====================================================

        if not self.done:

            self.waiting += 1

            reward -= (
                self.waiting * 2
            )

        # ====================================================
        # PATIENT DETERIORATION
        # ====================================================

        if (
            self.condition == 2
            and
            self.waiting >= 2
            and
            not self.done
        ):

            if random.random() < 0.25:

                self.outcome = -1

                reward -= 50

                self.done = True

        # ====================================================
        # RESOURCE COST
        # ====================================================

        if action in [1, 3]:

            reward -= 3

        # ====================================================
        # TIME LIMIT
        # ====================================================

        if self.steps >= self.max_steps:

            if not self.done:

                self.outcome = -1

                reward -= 30

                self.done = True

        return (
            self.get_state(),
            reward,
            self.done,
            {
                "outcome": self.outcome,
                "condition": self.condition,
                "resources": self.resources,
                "waiting": self.waiting
            }
        )


# ============================================================
# Q-LEARNING AGENT
# ============================================================

class QLearningAgent:

    def __init__(
        self,
        state_size,
        action_size
    ):

        self.state_size = state_size

        self.action_size = action_size

        # Learning rate
        self.alpha = 0.1

        # Discount factor
        self.gamma = 0.95

        # Exploration probability
        self.epsilon = 1.0

        self.epsilon_min = 0.05

        self.epsilon_decay = 0.995

        # Q-table
        self.q_table = np.zeros(
            (
                state_size,
                action_size
            )
        )

    # ========================================================
    # ACTION SELECTION
    # ========================================================

    def choose_action(
        self,
        state
    ):

        # Exploration
        if (
            random.random()
            <
            self.epsilon
        ):

            return random.randrange(
                self.action_size
            )

        # Exploitation
        return np.argmax(
            self.q_table[state]
        )

    # ========================================================
    # Q-LEARNING UPDATE
    # ========================================================

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        current_q = (
            self.q_table[
                state,
                action
            ]
        )

        if done:

            target = reward

        else:

            target = (
                reward
                +
                self.gamma
                *
                np.max(
                    self.q_table[
                        next_state
                    ]
                )
            )

        self.q_table[
            state,
            action
        ] = (
            current_q
            +
            self.alpha
            *
            (
                target
                -
                current_q
            )
        )

    # ========================================================
    # EPSILON DECAY
    # ========================================================

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon
            *
            self.epsilon_decay
        )


# ============================================================
# CREATE ENVIRONMENT AND AGENT
# ============================================================

environment = HealthcareEnvironment()

state_size = (
    environment.num_conditions
    *
    environment.num_resources
    *
    environment.num_waiting_levels
)

agent = QLearningAgent(
    state_size=state_size,
    action_size=environment.num_actions
)


# ============================================================
# TRAINING
# ============================================================

episodes = 5000

reward_history = []

success_history = []

waiting_history = []

resource_history = []

print("=" * 75)
print(" REINFORCEMENT LEARNING FOR HEALTHCARE MANAGEMENT")
print("=" * 75)

print("\nTraining started...\n")


for episode in range(
    episodes
):

    state_tuple = environment.reset()

    state = environment.state_to_index(
        state_tuple
    )

    total_reward = 0

    total_waiting = 0

    total_resources = 0

    success = 0

    for step in range(
        environment.max_steps
    ):

        # ----------------------------------------------------
        # Select action
        # ----------------------------------------------------

        action = agent.choose_action(
            state
        )

        # ----------------------------------------------------
        # Environment transition
        # ----------------------------------------------------

        (
            next_state_tuple,
            reward,
            done,
            info
        ) = environment.step(
            action
        )

        next_state = (
            environment.state_to_index(
                next_state_tuple
            )
        )

        # ----------------------------------------------------
        # Update Q-table
        # ----------------------------------------------------

        agent.update(
            state,
            action,
            reward,
            next_state,
            done
        )

        state = next_state

        total_reward += reward

        total_waiting += (
            info["waiting"]
        )

        total_resources += (
            info["resources"]
        )

        if (
            info["outcome"]
            == 1
        ):

            success = 1

        if done:

            break

    # --------------------------------------------------------
    # Decay exploration
    # --------------------------------------------------------

    agent.decay_epsilon()

    reward_history.append(
        total_reward
    )

    success_history.append(
        success
    )

    waiting_history.append(
        total_waiting
    )

    resource_history.append(
        total_resources
    )

    # --------------------------------------------------------
    # Training progress
    # --------------------------------------------------------

    if (
        (episode + 1)
        % 500
        == 0
    ):

        avg_reward = np.mean(
            reward_history[-500:]
        )

        success_rate = (
            np.mean(
                success_history[-500:]
            )
            *
            100
        )

        avg_waiting = np.mean(
            waiting_history[-500:]
        )

        print(
            f"Episode {episode + 1:4d} | "
            f"Avg Reward: "
            f"{avg_reward:8.2f} | "
            f"Success Rate: "
            f"{success_rate:6.2f}% | "
            f"Avg Waiting: "
            f"{avg_waiting:6.2f} | "
            f"Epsilon: "
            f"{agent.epsilon:.3f}"
        )


# ============================================================
# TRAINING RESULTS
# ============================================================

print("\n" + "=" * 75)
print(" TRAINING RESULTS")
print("=" * 75)

print(
    "Training Episodes:",
    episodes
)

print(
    "Learned States:",
    np.sum(
        np.any(
            agent.q_table != 0,
            axis=1
        )
    )
)

print(
    "Final Epsilon:",
    round(
        agent.epsilon,
        4
    )
)

print(
    "Final Average Reward:",
    round(
        np.mean(
            reward_history[-500:]
        ),
        2
    )
)

print(
    "Final Success Rate:",
    round(
        np.mean(
            success_history[-500:]
        )
        * 100,
        2
    ),
    "%"
)


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 75)
print(" EVALUATION OF LEARNED POLICY")
print("=" * 75)

# Disable exploration
agent.epsilon = 0

evaluation_episodes = 500

evaluation_rewards = []

evaluation_waiting = []

evaluation_resources = []

evaluation_success = 0

evaluation_failures = 0


for episode in range(
    evaluation_episodes
):

    state_tuple = environment.reset()

    state = environment.state_to_index(
        state_tuple
    )

    total_reward = 0

    total_waiting = 0

    total_resources = 0

    for step in range(
        environment.max_steps
    ):

        # Choose best learned action
        action = agent.choose_action(
            state
        )

        (
            next_state_tuple,
            reward,
            done,
            info
        ) = environment.step(
            action
        )

        state = environment.state_to_index(
            next_state_tuple
        )

        total_reward += reward

        total_waiting += (
            info["waiting"]
        )

        total_resources += (
            info["resources"]
        )

        if done:

            break

    evaluation_rewards.append(
        total_reward
    )

    evaluation_waiting.append(
        total_waiting
    )

    evaluation_resources.append(
        total_resources
    )

    if info["outcome"] == 1:

        evaluation_success += 1

    else:

        evaluation_failures += 1


# ============================================================
# EVALUATION RESULTS
# ============================================================

success_rate = (
    evaluation_success
    /
    evaluation_episodes
    *
    100
)


print(
    "Evaluation Episodes:",
    evaluation_episodes
)

print(
    "Successful Patients:",
    evaluation_success
)

print(
    "Unsuccessful Cases:",
    evaluation_failures
)

print(
    "Patient Outcome Success Rate:",
    round(
        success_rate,
        2
    ),
    "%"
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
    "Average Waiting:",
    round(
        np.mean(
            evaluation_waiting
        ),
        2
    )
)

print(
    "Average Resource Utilization:",
    round(
        np.mean(
            evaluation_resources
        ),
        2
    )
)


# ============================================================
# DISPLAY LEARNED POLICY
# ============================================================

print("\n" + "=" * 75)
print(" LEARNED HEALTHCARE POLICY")
print("=" * 75)


for condition in range(3):

    for resources in range(3):

        for waiting in range(3):

            state = environment.state_to_index(
                (
                    condition,
                    resources,
                    waiting
                )
            )

            best_action = np.argmax(
                agent.q_table[state]
            )

            print(
                f"Condition: "
                f"{environment.condition_names[condition]:<9} | "
                f"Resources: "
                f"{environment.resource_names[resources]:<6} | "
                f"Waiting: "
                f"{environment.waiting_names[waiting]:<7} | "
                f"Action: "
                f"{environment.actions[best_action]}"
            )


# ============================================================
# SAMPLE PATIENT SIMULATION
# ============================================================

print("\n" + "=" * 75)
print(" SAMPLE PATIENT FLOW")
print("=" * 75)

state_tuple = environment.reset()

state = environment.state_to_index(
    state_tuple
)

print(
    "\nInitial State:",
    state_tuple
)

for step in range(
    environment.max_steps
):

    action = agent.choose_action(
        state
    )

    print(
        f"\nStep {step + 1}"
    )

    print(
        "Patient Condition:",
        environment.condition_names[
            environment.condition
        ]
    )

    print(
        "Available Resources:",
        environment.resources
    )

    print(
        "Waiting Time:",
        environment.waiting
    )

    print(
        "Selected Action:",
        environment.actions[action]
    )

    (
        next_state_tuple,
        reward,
        done,
        info
    ) = environment.step(
        action
    )

    print(
        "Reward:",
        reward
    )

    state = environment.state_to_index(
        next_state_tuple
    )

    if done:

        print(
            "\nFinal Outcome:",
            "Recovered"
            if info["outcome"] == 1
            else "Unsuccessful"
        )

        break


# ============================================================
# PLOT TRAINING REWARD
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
    "Healthcare RL Training Reward"
)

plt.grid(True)

plt.show()


# ============================================================
# PLOT SUCCESS RATE
# ============================================================

window = 100

success_curve = (
    np.convolve(
        success_history,
        np.ones(window) / window,
        mode="valid"
    )
    *
    100
)

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    success_curve
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Success Rate (%)"
)

plt.title(
    "Patient Outcome Success Rate"
)

plt.grid(True)

plt.show()


# ============================================================
# PLOT WAITING TIME
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    waiting_history
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Total Waiting Time"
)

plt.title(
    "Patient Waiting Time During Training"
)

plt.grid(True)

plt.show()

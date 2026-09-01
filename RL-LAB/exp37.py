import random
import numpy as np
from collections import defaultdict


# ============================================================
# MULTI-AGENT MAXQ COOPERATIVE ENVIRONMENT
# ============================================================

class CooperativeEnvironment:

    def __init__(self):

        self.max_steps = 50

        self.reset()

    # --------------------------------------------------------
    # RESET ENVIRONMENT
    # --------------------------------------------------------

    def reset(self):

        # Shared resources
        self.resources = 0

        # Construction status
        self.built = False

        # Delivery status
        self.delivered = False

        # Agent locations
        self.agent_positions = {
            "Gatherer": 0,
            "Builder": 1,
            "Deliverer": 2
        }

        # Step counter
        self.steps = 0

        return self.get_state()

    # --------------------------------------------------------
    # GET GLOBAL STATE
    # --------------------------------------------------------

    def get_state(self):

        return (
            self.resources,
            int(self.built),
            int(self.delivered)
        )

    # --------------------------------------------------------
    # EXECUTE AGENT ACTION
    # --------------------------------------------------------

    def step(
        self,
        agent_name,
        action
    ):

        reward = -1

        self.steps += 1

        # ====================================================
        # GATHERER
        # ====================================================

        if agent_name == "Gatherer":

            if action == "gather":

                self.resources += 1

                reward = 5

            elif action == "wait":

                reward = -1

        # ====================================================
        # BUILDER
        # ====================================================

        elif agent_name == "Builder":

            if action == "build":

                if self.resources >= 2:

                    self.resources -= 2

                    self.built = True

                    reward = 10

                else:

                    reward = -5

            elif action == "wait":

                reward = -1

        # ====================================================
        # DELIVERER
        # ====================================================

        elif agent_name == "Deliverer":

            if action == "deliver":

                if self.built:

                    self.delivered = True

                    reward = 20

                else:

                    reward = -5

            elif action == "wait":

                reward = -1

        # ====================================================
        # GLOBAL SUCCESS
        # ====================================================

        done = False

        if self.delivered:

            reward += 50

            done = True

        elif self.steps >= self.max_steps:

            done = True

        return (
            self.get_state(),
            reward,
            done
        )


# ============================================================
# MAXQ SUBTASK AGENT
# ============================================================

class MAXQAgent:

    def __init__(
        self,
        name,
        actions,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995
    ):

        self.name = name

        self.actions = actions

        self.alpha = alpha

        self.gamma = gamma

        self.epsilon = epsilon

        self.epsilon_min = epsilon_min

        self.epsilon_decay = epsilon_decay

        # MAXQ value table
        self.Q = defaultdict(
            lambda: defaultdict(float)
        )

    # --------------------------------------------------------
    # STATE ENCODING
    # --------------------------------------------------------

    def state_key(self, state):

        return tuple(state)

    # --------------------------------------------------------
    # EPSILON-GREEDY POLICY
    # --------------------------------------------------------

    def choose_action(
        self,
        state,
        training=True
    ):

        state_key = self.state_key(
            state
        )

        # Exploration
        if (
            training
            and
            random.random()
            < self.epsilon
        ):

            return random.choice(
                self.actions
            )

        # Exploitation
        q_values = [
            self.Q[
                state_key
            ][action]
            for action in self.actions
        ]

        max_q = max(
            q_values
        )

        best_actions = [
            action
            for action in self.actions
            if self.Q[
                state_key
            ][action] == max_q
        ]

        return random.choice(
            best_actions
        )

    # --------------------------------------------------------
    # MAXQ SUBTASK UPDATE
    # --------------------------------------------------------

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        state_key = self.state_key(
            state
        )

        next_key = self.state_key(
            next_state
        )

        current_value = self.Q[
            state_key
        ][action]

        if done:

            target = reward

        else:

            next_values = [
                self.Q[
                    next_key
                ][a]
                for a in self.actions
            ]

            target = (
                reward
                +
                self.gamma
                *
                max(
                    next_values
                )
            )

        # MAXQ-style value update
        self.Q[
            state_key
        ][action] = (
            current_value
            +
            self.alpha
            *
            (
                target
                -
                current_value
            )
        )

    # --------------------------------------------------------
    # DECAY EXPLORATION
    # --------------------------------------------------------

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon *
            self.epsilon_decay
        )


# ============================================================
# MAXQ HIERARCHY
# ============================================================

class MAXQHierarchy:

    def __init__(self):

        # ----------------------------------------------------
        # Level 0: Root task
        # ----------------------------------------------------

        self.root = [
            "GATHER",
            "BUILD",
            "DELIVER"
        ]

        # ----------------------------------------------------
        # Level 1: Subtasks
        # ----------------------------------------------------

        self.gather_agent = MAXQAgent(
            name="Gatherer",
            actions=[
                "gather",
                "wait"
            ]
        )

        self.build_agent = MAXQAgent(
            name="Builder",
            actions=[
                "build",
                "wait"
            ]
        )

        self.deliver_agent = MAXQAgent(
            name="Deliverer",
            actions=[
                "deliver",
                "wait"
            ]
        )

    # --------------------------------------------------------
    # EXECUTE GATHER SUBTASK
    # --------------------------------------------------------

    def execute_gather(
        self,
        environment,
        training=True
    ):

        state = environment.get_state()

        action = (
            self.gather_agent.choose_action(
                state,
                training
            )
        )

        (
            next_state,
            reward,
            done
        ) = environment.step(
            "Gatherer",
            action
        )

        if training:

            self.gather_agent.update(
                state,
                action,
                reward,
                next_state,
                done
            )

        return (
            next_state,
            reward,
            done
        )

    # --------------------------------------------------------
    # EXECUTE BUILD SUBTASK
    # --------------------------------------------------------

    def execute_build(
        self,
        environment,
        training=True
    ):

        state = environment.get_state()

        action = (
            self.build_agent.choose_action(
                state,
                training
            )
        )

        (
            next_state,
            reward,
            done
        ) = environment.step(
            "Builder",
            action
        )

        if training:

            self.build_agent.update(
                state,
                action,
                reward,
                next_state,
                done
            )

        return (
            next_state,
            reward,
            done
        )

    # --------------------------------------------------------
    # EXECUTE DELIVERY SUBTASK
    # --------------------------------------------------------

    def execute_deliver(
        self,
        environment,
        training=True
    ):

        state = environment.get_state()

        action = (
            self.deliver_agent.choose_action(
                state,
                training
            )
        )

        (
            next_state,
            reward,
            done
        ) = environment.step(
            "Deliverer",
            action
        )

        if training:

            self.deliver_agent.update(
                state,
                action,
                reward,
                next_state,
                done
            )

        return (
            next_state,
            reward,
            done
        )

    # --------------------------------------------------------
    # EXECUTE ROOT TASK
    # --------------------------------------------------------

    def execute_root(
        self,
        environment,
        training=True
    ):

        total_reward = 0

        # ====================================================
        # SUBTASK 1: GATHER
        # ====================================================

        while environment.resources < 2:

            (
                state,
                reward,
                done
            ) = self.execute_gather(
                environment,
                training
            )

            total_reward += reward

            if done:

                return (
                    total_reward,
                    False
                )

        # ====================================================
        # SUBTASK 2: BUILD
        # ====================================================

        if not environment.built:

            (
                state,
                reward,
                done
            ) = self.execute_build(
                environment,
                training
            )

            total_reward += reward

            if done:

                return (
                    total_reward,
                    False
                )

        # ====================================================
        # SUBTASK 3: DELIVER
        # ====================================================

        if environment.built:

            (
                state,
                reward,
                done
            ) = self.execute_deliver(
                environment,
                training
            )

            total_reward += reward

            if environment.delivered:

                return (
                    total_reward,
                    True
                )

        return (
            total_reward,
            False
        )

    # --------------------------------------------------------
    # DECAY ALL POLICIES
    # --------------------------------------------------------

    def decay_policies(self):

        self.gather_agent.decay_epsilon()

        self.build_agent.decay_epsilon()

        self.deliver_agent.decay_epsilon()


# ============================================================
# 1. CREATE ENVIRONMENT AND MAXQ SYSTEM
# ============================================================

environment = CooperativeEnvironment()

maxq = MAXQHierarchy()


# ============================================================
# 2. TRAINING
# ============================================================

episodes = 1000

training_rewards = []

successful_tasks = 0


print("=" * 70)
print(" MAXQ MULTI-AGENT COOPERATIVE TASK")
print("=" * 70)

print("\nHierarchy:")

print(
    "ROOT -> GATHER -> BUILD -> DELIVER"
)

print("\nTraining started...\n")


for episode in range(
    episodes
):

    environment.reset()

    total_reward, success = (
        maxq.execute_root(
            environment,
            training=True
        )
    )

    training_rewards.append(
        total_reward
    )

    if success:

        successful_tasks += 1

    maxq.decay_policies()

    if (
        (episode + 1)
        % 100
        == 0
    ):

        recent_success = np.mean(
            [
                1
                if x > 0
                else 0
                for x in
                training_rewards[-100:]
            ]
        ) * 100

        print(
            f"Episode {episode + 1:4d} | "
            f"Reward: {total_reward:7.2f} | "
            f"Resources: {environment.resources} | "
            f"Built: {environment.built} | "
            f"Delivered: {environment.delivered}"
        )


# ============================================================
# 3. TRAINING RESULTS
# ============================================================

print("\n" + "=" * 70)
print(" TRAINING RESULTS")
print("=" * 70)

print(
    "Training Episodes:",
    episodes
)

print(
    "Successful Tasks:",
    successful_tasks
)

print(
    "Gatherer States:",
    len(maxq.gather_agent.Q)
)

print(
    "Builder States:",
    len(maxq.build_agent.Q)
)

print(
    "Deliverer States:",
    len(maxq.deliver_agent.Q)
)

print(
    "Final Gatherer Epsilon:",
    round(
        maxq.gather_agent.epsilon,
        4
    )
)

print(
    "Final Builder Epsilon:",
    round(
        maxq.build_agent.epsilon,
        4
    )
)

print(
    "Final Deliverer Epsilon:",
    round(
        maxq.deliver_agent.epsilon,
        4
    )
)


# ============================================================
# 4. EVALUATION
# ============================================================

print("\n" + "=" * 70)
print(" EVALUATION")
print("=" * 70)

# Disable exploration
maxq.gather_agent.epsilon = 0
maxq.build_agent.epsilon = 0
maxq.deliver_agent.epsilon = 0

evaluation_episodes = 100

evaluation_success = 0

evaluation_rewards = []

evaluation_steps = []


for episode in range(
    evaluation_episodes
):

    environment.reset()

    total_reward = 0

    steps_before = environment.steps

    # --------------------------------------------------------
    # Gather
    # --------------------------------------------------------

    while environment.resources < 2:

        (
            state,
            reward,
            done
        ) = maxq.execute_gather(
            environment,
            training=False
        )

        total_reward += reward

        if done:

            break

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    if (
        not done
        and
        not environment.built
    ):

        (
            state,
            reward,
            done
        ) = maxq.execute_build(
            environment,
            training=False
        )

        total_reward += reward

    # --------------------------------------------------------
    # Deliver
    # --------------------------------------------------------

    if (
        not done
        and
        environment.built
    ):

        (
            state,
            reward,
            done
        ) = maxq.execute_deliver(
            environment,
            training=False
        )

        total_reward += reward

    steps_taken = (
        environment.steps
        - steps_before
    )

    evaluation_rewards.append(
        total_reward
    )

    evaluation_steps.append(
        steps_taken
    )

    if environment.delivered:

        evaluation_success += 1


# ============================================================
# 5. EVALUATION RESULTS
# ============================================================

success_rate = (
    evaluation_success /
    evaluation_episodes
    * 100
)


print(
    "Evaluation Episodes:",
    evaluation_episodes
)

print(
    "Successful Tasks:",
    evaluation_success
)

print(
    "Success Rate:",
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
    "Average Steps:",
    round(
        np.mean(
            evaluation_steps
        ),
        2
    )
)


# ============================================================
# 6. SHOW LEARNED POLICIES
# ============================================================

print("\n" + "=" * 70)
print(" LEARNED SUBTASK POLICIES")
print("=" * 70)


# ------------------------------------------------------------
# Gatherer policy
# ------------------------------------------------------------

print("\nGATHERER POLICY:")

for state in [
    (0, 0, 0),
    (1, 0, 0),
    (2, 0, 0)
]:

    action = (
        maxq.gather_agent.choose_action(
            state,
            training=False
        )
    )

    print(
        "State:",
        state,
        "-> Action:",
        action
    )


# ------------------------------------------------------------
# Builder policy
# ------------------------------------------------------------

print("\nBUILDER POLICY:")

for state in [
    (0, 0, 0),
    (1, 0, 0),
    (2, 0, 0)
]:

    action = (
        maxq.build_agent.choose_action(
            state,
            training=False
        )
    )

    print(
        "State:",
        state,
        "-> Action:",
        action
    )


# ------------------------------------------------------------
# Deliverer policy
# ------------------------------------------------------------

print("\nDELIVERER POLICY:")

for state in [
    (0, 0, 0),
    (0, 1, 0),
    (0, 1, 1)
]:

    action = (
        maxq.deliver_agent.choose_action(
            state,
            training=False
        )
    )

    print(
        "State:",
        state,
        "-> Action:",
        action
    )


# ============================================================
# 7. DEMONSTRATE ONE COOPERATIVE TASK
# ============================================================

print("\n" + "=" * 70)
print(" SAMPLE COOPERATIVE TASK")
print("=" * 70)

environment.reset()

print(
    "\nInitial State:",
    environment.get_state()
)


# Gatherer
while environment.resources < 2:

    (
        state,
        reward,
        done
    ) = maxq.execute_gather(
        environment,
        training=False
    )

    print(
        "Gatherer -> Gather Resource | "
        "Resources:",
        environment.resources,
        "| Reward:",
        reward
    )


# Builder
(
    state,
    reward,
    done
) = maxq.execute_build(
    environment,
    training=False
)

print(
    "Builder -> Build Unit/Product | "
    "Built:",
    environment.built,
    "| Reward:",
    reward
)


# Deliverer
(
    state,
    reward,
    done
) = maxq.execute_deliver(
    environment,
    training=False
)

print(
    "Deliverer -> Deliver Product | "
    "Delivered:",
    environment.delivered,
    "| Reward:",
    reward
)


if environment.delivered:

    print(
        "\nOVERALL TASK: SUCCESS"
    )

else:

    print(
        "\nOVERALL TASK: FAILED"
    )


# ============================================================
# 8. TRAINING REWARD GRAPH
# ============================================================

import matplotlib.pyplot as plt


plt.figure(
    figsize=(10, 5)
)

plt.plot(
    training_rewards
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Total Reward"
)

plt.title(
    "MAXQ Multi-Agent Training Performance"
)

plt.grid(True)

plt.show()

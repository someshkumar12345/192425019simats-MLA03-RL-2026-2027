import numpy as np
import random


# ============================================================
# TRAFFIC LIGHT OPTIMIZATION USING POLICY ITERATION
# ============================================================

# Traffic levels
LOW = 0
MEDIUM = 1
HIGH = 2

traffic_names = {
    LOW: "Low",
    MEDIUM: "Medium",
    HIGH: "High"
}

# Signal phases
NS_GREEN = 0
EW_GREEN = 1

signal_names = {
    NS_GREEN: "North-South Green",
    EW_GREEN: "East-West Green"
}

# Actions
KEEP = 0
SWITCH = 1

action_names = {
    KEEP: "Keep Signal",
    SWITCH: "Switch Signal"
}

# Number of states and actions
NUM_TRAFFIC_LEVELS = 3
NUM_PHASES = 2
NUM_STATES = NUM_TRAFFIC_LEVELS * NUM_PHASES
NUM_ACTIONS = 2

# Discount factor
GAMMA = 0.9

# Convergence threshold
THETA = 0.0001


# ============================================================
# STATE CONVERSION FUNCTIONS
# ============================================================

def state_to_index(traffic, phase):

    return traffic * NUM_PHASES + phase


def index_to_state(index):

    traffic = index // NUM_PHASES
    phase = index % NUM_PHASES

    return traffic, phase


# ============================================================
# TRAFFIC TRANSITION PROBABILITIES
# ============================================================

# Probability of moving from one traffic level
# to another traffic level.

traffic_transition = {

    LOW: {
        LOW: 0.70,
        MEDIUM: 0.30,
        HIGH: 0.00
    },

    MEDIUM: {
        LOW: 0.20,
        MEDIUM: 0.60,
        HIGH: 0.20
    },

    HIGH: {
        LOW: 0.00,
        MEDIUM: 0.40,
        HIGH: 0.60
    }
}


# ============================================================
# REWARD FUNCTION
# ============================================================

def get_reward(traffic, phase, action):

    # Estimated waiting vehicles
    waiting_vehicles = {
        LOW: 5,
        MEDIUM: 15,
        HIGH: 30
    }

    # Waiting-time penalty
    reward = -waiting_vehicles[traffic]

    # Small penalty for switching
    # because changing signals too frequently
    # is undesirable.

    if action == SWITCH:

        reward -= 2

    return reward


# ============================================================
# GET NEXT STATE PROBABILITIES
# ============================================================

def get_transition_probabilities(
    traffic,
    phase,
    action
):

    probabilities = np.zeros(
        NUM_STATES
    )

    # Determine next signal phase

    if action == KEEP:

        next_phase = phase

    else:

        next_phase = 1 - phase

    # Traffic changes according to
    # transition probabilities.

    for next_traffic in range(
        NUM_TRAFFIC_LEVELS
    ):

        probability = traffic_transition[
            traffic
        ][next_traffic]

        next_state = state_to_index(
            next_traffic,
            next_phase
        )

        probabilities[next_state] = (
            probability
        )

    return probabilities


# ============================================================
# CREATE MDP TRANSITION AND REWARD MATRICES
# ============================================================

P = np.zeros(
    (
        NUM_STATES,
        NUM_ACTIONS,
        NUM_STATES
    )
)

R = np.zeros(
    (
        NUM_STATES,
        NUM_ACTIONS
    )
)


for state in range(NUM_STATES):

    traffic, phase = index_to_state(
        state
    )

    for action in range(NUM_ACTIONS):

        # Transition probabilities
        P[state, action] = (
            get_transition_probabilities(
                traffic,
                phase,
                action
            )
        )

        # Reward
        R[state, action] = (
            get_reward(
                traffic,
                phase,
                action
            )
        )


# ============================================================
# POLICY EVALUATION
# ============================================================

def policy_evaluation(
    policy,
    value_function
):

    while True:

        delta = 0

        new_value_function = (
            value_function.copy()
        )

        for state in range(NUM_STATES):

            action = policy[state]

            value = 0

            for next_state in range(
                NUM_STATES
            ):

                probability = P[
                    state,
                    action,
                    next_state
                ]

                value += probability * (
                    R[state, action]
                    +
                    GAMMA *
                    value_function[next_state]
                )

            new_value_function[state] = value

            delta = max(
                delta,
                abs(
                    value -
                    value_function[state]
                )
            )

        value_function = (
            new_value_function
        )

        if delta < THETA:

            break

    return value_function


# ============================================================
# POLICY IMPROVEMENT
# ============================================================

def policy_improvement(
    value_function
):

    new_policy = np.zeros(
        NUM_STATES,
        dtype=int
    )

    policy_stable = True

    for state in range(NUM_STATES):

        action_values = []

        for action in range(NUM_ACTIONS):

            value = 0

            for next_state in range(
                NUM_STATES
            ):

                probability = P[
                    state,
                    action,
                    next_state
                ]

                value += probability * (
                    R[state, action]
                    +
                    GAMMA *
                    value_function[next_state]
                )

            action_values.append(
                value
            )

        best_action = np.argmax(
            action_values
        )

        new_policy[state] = (
            best_action
        )

    return new_policy


# ============================================================
# POLICY ITERATION
# ============================================================

def policy_iteration():

    # Start with arbitrary policy
    policy = np.zeros(
        NUM_STATES,
        dtype=int
    )

    value_function = np.zeros(
        NUM_STATES
    )

    iteration = 0

    while True:

        iteration += 1

        # --------------------------------------------
        # Policy Evaluation
        # --------------------------------------------

        value_function = (
            policy_evaluation(
                policy,
                value_function
            )
        )

        # --------------------------------------------
        # Policy Improvement
        # --------------------------------------------

        new_policy = (
            policy_improvement(
                value_function
            )
        )

        # --------------------------------------------
        # Check whether policy changed
        # --------------------------------------------

        if np.array_equal(
            policy,
            new_policy
        ):

            break

        policy = new_policy

    return (
        policy,
        value_function,
        iteration
    )


# ============================================================
# RUN POLICY ITERATION
# ============================================================

optimal_policy, optimal_values, iterations = (
    policy_iteration()
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 70)
print(" TRAFFIC LIGHT OPTIMIZATION")
print(" USING POLICY ITERATION")
print("=" * 70)

print(
    "\nPolicy Iteration Converged In:",
    iterations,
    "iterations"
)

print("\nOptimal Policy:")

print("-" * 70)

print(
    f"{'State':<8}"
    f"{'Traffic':<12}"
    f"{'Signal':<25}"
    f"{'Action':<20}"
    f"{'Value':<10}"
)

print("-" * 70)


for state in range(NUM_STATES):

    traffic, phase = index_to_state(
        state
    )

    print(
        f"{state:<8}"
        f"{traffic_names[traffic]:<12}"
        f"{signal_names[phase]:<25}"
        f"{action_names[optimal_policy[state]]:<20}"
        f"{optimal_values[state]:<10.2f}"
    )


# ============================================================
# SIMULATE TRAFFIC USING OPTIMAL POLICY
# ============================================================

def simulate_policy(
    policy,
    steps=100
):

    # Start with medium traffic
    traffic = MEDIUM

    # Start with North-South green
    phase = NS_GREEN

    total_waiting_time = 0

    switch_count = 0

    traffic_history = []

    for step in range(steps):

        state = state_to_index(
            traffic,
            phase
        )

        action = policy[state]

        # Record traffic
        traffic_history.append(
            traffic
        )

        # Waiting vehicles
        waiting_vehicles = {
            LOW: 5,
            MEDIUM: 15,
            HIGH: 30
        }

        total_waiting_time += (
            waiting_vehicles[traffic]
        )

        # Count signal changes
        if action == SWITCH:

            switch_count += 1

            phase = 1 - phase

        # Generate next traffic level
        probabilities = []

        for level in range(
            NUM_TRAFFIC_LEVELS
        ):

            probabilities.append(
                traffic_transition[
                    traffic
                ][level]
            )

        traffic = np.random.choice(
            [
                LOW,
                MEDIUM,
                HIGH
            ],
            p=probabilities
        )

    return (
        total_waiting_time,
        switch_count,
        traffic_history
    )


# ============================================================
# EVALUATE OPTIMAL POLICY
# ============================================================

waiting_time, switch_count, traffic_history = (
    simulate_policy(
        optimal_policy,
        steps=100
    )
)


print("\n" + "=" * 70)
print(" POLICY PERFORMANCE")
print("=" * 70)

print(
    "Simulation Steps       : 100"
)

print(
    "Total Vehicle Wait Time:",
    waiting_time
)

print(
    "Average Wait per Step  :",
    round(
        waiting_time / 100,
        2
    )
)

print(
    "Signal Switches        :",
    switch_count
)


# ============================================================
# BASELINE POLICY
# ============================================================

# Baseline policy:
# Always keep the current signal.

baseline_policy = np.zeros(
    NUM_STATES,
    dtype=int
)

baseline_wait, baseline_switches, _ = (
    simulate_policy(
        baseline_policy,
        steps=100
    )
)


print("\n" + "=" * 70)
print(" BASELINE VS OPTIMAL POLICY")
print("=" * 70)

print(
    f"{'Metric':<30}"
    f"{'Baseline':<15}"
    f"{'Optimal':<15}"
)

print("-" * 60)

print(
    f"{'Total Waiting Time':<30}"
    f"{baseline_wait:<15}"
    f"{waiting_time:<15}"
)

print(
    f"{'Signal Switches':<30}"
    f"{baseline_switches:<15}"
    f"{switch_count:<15}"
)

print(
    f"{'Average Waiting Time':<30}"
    f"{baseline_wait / 100:<15.2f}"
    f"{waiting_time / 100:<15.2f}"
)


# ============================================================
# IMPROVEMENT
# ============================================================

if baseline_wait != 0:

    improvement = (
        (baseline_wait - waiting_time)
        / baseline_wait
    ) * 100

else:

    improvement = 0


print(
    "\nWaiting-Time Improvement:",
    f"{improvement:.2f}%"
)


# ============================================================
# VISUALIZE TRAFFIC LEVELS
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    traffic_history,
    marker="o"
)

plt.yticks(
    [0, 1, 2],
    ["Low", "Medium", "High"]
)

plt.xlabel(
    "Time Step"
)

plt.ylabel(
    "Traffic Level"
)

plt.title(
    "Traffic Conditions During Optimal Policy Simulation"
)

plt.grid(True)

plt.show()

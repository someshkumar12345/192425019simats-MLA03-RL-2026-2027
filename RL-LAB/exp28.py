import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# ROBOT GRID NAVIGATION USING BELLMAN'S OPTIMALITY EQUATION
# ============================================================

# Grid:
#
# S = Start
# G = Goal
# X = Obstacle
# . = Free cell
#
# S . . . .
# . X X . .
# . . . . .
# . X . X .
# . . . . G

ROWS = 5
COLS = 5

START = (0, 0)
GOAL = (4, 4)

OBSTACLES = {
    (1, 1),
    (1, 2),
    (3, 1),
    (3, 3)
}

# ============================================================
# ACTIONS
# ============================================================

# Action : (row movement, column movement)

ACTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

ACTION_SYMBOLS = {
    "UP": "↑",
    "DOWN": "↓",
    "LEFT": "←",
    "RIGHT": "→"
}


# ============================================================
# PARAMETERS
# ============================================================

GAMMA = 0.9
THETA = 0.0001

# Reward structure
STEP_REWARD = -1
GOAL_REWARD = 100
OBSTACLE_REWARD = -10


# ============================================================
# ENVIRONMENT FUNCTIONS
# ============================================================

def is_valid_state(state):

    row, col = state

    return (
        0 <= row < ROWS
        and
        0 <= col < COLS
        and
        state not in OBSTACLES
    )


def get_next_state(state, action):

    row, col = state

    dr, dc = ACTIONS[action]

    new_state = (
        row + dr,
        col + dc
    )

    # If movement is invalid,
    # robot remains in current state.
    if not is_valid_state(new_state):

        return state

    return new_state


def get_reward(next_state):

    if next_state == GOAL:

        return GOAL_REWARD

    return STEP_REWARD


# ============================================================
# INITIALIZE VALUE FUNCTION
# ============================================================

V = np.zeros(
    (ROWS, COLS)
)

# Obstacles do not have a value.
for obstacle in OBSTACLES:

    V[obstacle] = np.nan


# ============================================================
# VALUE ITERATION
# BELLMAN'S OPTIMALITY EQUATION
# ============================================================

iteration = 0

while True:

    delta = 0

    new_V = V.copy()

    for row in range(ROWS):

        for col in range(COLS):

            state = (row, col)

            # Skip obstacles
            if state in OBSTACLES:

                continue

            # Goal state
            if state == GOAL:

                new_V[state] = 0

                continue

            action_values = []

            # Calculate value for every action
            for action in ACTIONS:

                next_state = get_next_state(
                    state,
                    action
                )

                reward = get_reward(
                    next_state
                )

                value = (
                    reward
                    +
                    GAMMA * V[next_state]
                )

                action_values.append(
                    value
                )

            # Bellman's optimality equation
            best_value = max(
                action_values
            )

            new_V[state] = best_value

            delta = max(
                delta,
                abs(
                    best_value - V[state]
                )
            )

    V = new_V

    iteration += 1

    # Convergence condition
    if delta < THETA:

        break


# ============================================================
# DISPLAY VALUE FUNCTION
# ============================================================

print("=" * 60)
print(" BELLMAN'S OPTIMALITY EQUATION")
print(" ROBOT GRID NAVIGATION")
print("=" * 60)

print("\nNumber of iterations:", iteration)

print("\nOptimal State-Value Function:")

for row in range(ROWS):

    for col in range(COLS):

        state = (row, col)

        if state in OBSTACLES:

            print("   X   ", end=" ")

        else:

            print(
                f"{V[state]:7.2f}",
                end=" "
            )

    print()


# ============================================================
# EXTRACT OPTIMAL POLICY
# ============================================================

policy = {}

for row in range(ROWS):

    for col in range(COLS):

        state = (row, col)

        if state in OBSTACLES:
            continue

        if state == GOAL:
            continue

        best_action = None
        best_value = -float("inf")

        for action in ACTIONS:

            next_state = get_next_state(
                state,
                action
            )

            reward = get_reward(
                next_state
            )

            action_value = (
                reward
                +
                GAMMA * V[next_state]
            )

            if action_value > best_value:

                best_value = action_value
                best_action = action

        policy[state] = best_action


# ============================================================
# DISPLAY OPTIMAL POLICY
# ============================================================

print("\n" + "=" * 60)
print(" OPTIMAL POLICY")
print("=" * 60)

for row in range(ROWS):

    for col in range(COLS):

        state = (row, col)

        if state == START:

            print(" S ", end=" ")

        elif state == GOAL:

            print(" G ", end=" ")

        elif state in OBSTACLES:

            print(" X ", end=" ")

        else:

            action = policy[state]

            print(
                f" {ACTION_SYMBOLS[action]} ",
                end=" "
            )

    print()


# ============================================================
# GENERATE OPTIMAL PATH
# ============================================================

current_state = START

path = [current_state]

max_path_length = ROWS * COLS * 2

for _ in range(max_path_length):

    if current_state == GOAL:

        break

    action = policy[current_state]

    next_state = get_next_state(
        current_state,
        action
    )

    # Prevent infinite loops
    if next_state == current_state:

        print(
            "\nRobot became stuck."
        )

        break

    path.append(next_state)

    current_state = next_state


# ============================================================
# DISPLAY OPTIMAL PATH
# ============================================================

print("\n" + "=" * 60)
print(" OPTIMAL PATH")
print("=" * 60)

print(
    " → ".join(
        str(state)
        for state in path
    )
)

print(
    "\nNumber of moves:",
    len(path) - 1
)

if path[-1] == GOAL:

    print(
        "Result: Goal reached successfully."
    )

else:

    print(
        "Result: Goal not reached."
    )


# ============================================================
# VISUALIZE GRID AND OPTIMAL PATH
# ============================================================

grid = np.zeros(
    (ROWS, COLS)
)

for obstacle in OBSTACLES:

    grid[obstacle] = -1

for state in path:

    if state not in OBSTACLES:

        grid[state] = 1


plt.figure(
    figsize=(7, 7)
)

plt.imshow(
    grid
)

# ------------------------------------------------------------
# Draw grid lines
# ------------------------------------------------------------

plt.xticks(
    np.arange(-0.5, COLS, 1)
)

plt.yticks(
    np.arange(-0.5, ROWS, 1)
)

plt.grid(
    True,
    linewidth=2
)

# ------------------------------------------------------------
# Mark obstacles
# ------------------------------------------------------------

for row, col in OBSTACLES:

    plt.text(
        col,
        row,
        "X",
        ha="center",
        va="center",
        fontsize=18
    )

# ------------------------------------------------------------
# Mark start
# ------------------------------------------------------------

plt.text(
    START[1],
    START[0],
    "S",
    ha="center",
    va="center",
    fontsize=18
)

# ------------------------------------------------------------
# Mark goal
# ------------------------------------------------------------

plt.text(
    GOAL[1],
    GOAL[0],
    "G",
    ha="center",
    va="center",
    fontsize=18
)

# ------------------------------------------------------------
# Draw optimal path
# ------------------------------------------------------------

path_rows = [
    state[0]
    for state in path
]

path_cols = [
    state[1]
    for state in path
]

plt.plot(
    path_cols,
    path_rows,
    marker="o",
    linewidth=3
)

plt.title(
    "Optimal Robot Navigation Using Bellman's Equation"
)

plt.xlabel(
    "Column"
)

plt.ylabel(
    "Row"
)

plt.show()

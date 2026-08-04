import numpy as np
import random

# Grid size
size = 4

# Q-table
Q = np.zeros((size, size, 4))

alpha = 0.1
gamma = 0.9
epsilon = 0.2

goal = (3, 3)

# Move function
def move(state, action):
    x, y = state
    if action == 0:      # Up
        x = max(0, x - 1)
    elif action == 1:    # Down
        x = min(size - 1, x + 1)
    elif action == 2:    # Left
        y = max(0, y - 1)
    else:                # Right
        y = min(size - 1, y + 1)
    return (x, y)

# Training
for episode in range(100):
    state = (0, 0)

    while state != goal:
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[state])

        next_state = move(state, action)

        reward = 100 if next_state == goal else -1

        Q[state][action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state][action]
        )

        state = next_state

print("Training Completed!")

# Show learned path
state = (0, 0)
print("\nOptimal Path:")
while state != goal:
    print(state)
    action = np.argmax(Q[state])
    state = move(state, action)

print(goal)

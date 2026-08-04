# States and actions
states = ["Start", "Shelf", "Goal"]

actions = {
    "Start": {"Move Right": 5, "Move Left": -2},
    "Shelf": {"Move Right": 10, "Move Left": -1}
}

print("Optimal Policy:")

for state in actions:
    best_action = max(actions[state], key=actions[state].get)
    print(state, "->", best_action)

states = ["Start", "Middle", "Win"]

actions = {
    "Start": ["Attack", "Defend"],
    "Middle": ["Attack"]
}

rewards = {
    ("Start", "Attack"): 10,
    ("Start", "Defend"): -5,
    ("Middle", "Attack"): 100
}

policy = {}

for state in actions:
    best = max(actions[state], key=lambda a: rewards[(state, a)])
    policy[state] = best

print("Optimal Policy:")
for state, action in policy.items():
    print(state, "->", action)

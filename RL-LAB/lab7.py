import numpy as np
states = ["Start", "Road", "Destination"]
actions = {
    "Start": ["Go"],
    "Road": ["Go", "Wait"],
    "Destination": []
}
transition = {
    ("Start", "Go"): ("Road", -1),
    ("Road", "Go"): ("Destination", 20),
    ("Road", "Wait"): ("Road", -2)
}
gamma = 0.9
V = {s: 0 for s in states}
for _ in range(20):
    for state in ["Start", "Road"]:
        values = []
        for action in actions[state]:
            next_state, reward = transition[(state, action)]
            values.append(reward + gamma * V[next_state])
        V[state] = max(values)
print("Optimal State Values")
for s in states:
    print(s, ":", round(V[s], 2))
print("\nOptimal Policy")
for state in ["Start", "Road"]:
    best = max(
        actions[state],
        key=lambda a: transition[(state, a)][1] +
        gamma * V[transition[(state, a)][0]]
    )
    print(state, "->", best)

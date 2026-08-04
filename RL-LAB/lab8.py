import random
states = ["Dirty", "Clean"]
actions = ["Clean", "Move"]
Q = {(s, a): 0 for s in states for a in actions}
returns = {(s, a): [] for s in states for a in actions}
episodes = 100
for _ in range(episodes):
    state = random.choice(states)
    action = random.choice(actions)
    if state == "Dirty" and action == "Clean":
        reward = 10
    elif state == "Clean" and action == "Move":
        reward = 5
    else:
        reward = -2
    returns[(state, action)].append(reward)
    Q[(state, action)] = sum(returns[(state, action)]) / len(
        returns[(state, action)]
    )
print("Action Values")
for key in Q:
    print(key, ":", round(Q[key], 2))

gamma = 0.9

rewards = [0, -1, -1, 10]
values = [0, 0, 0, 0]

for _ in range(5):
    values[2] = rewards[2] + gamma * values[3]
    values[1] = rewards[1] + gamma * values[2]
    values[0] = rewards[0] + gamma * values[1]

print("State Values:")
for i, v in enumerate(values):
    print("State", i, ":", round(v, 2))

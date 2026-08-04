import random

ads = ["Ad A", "Ad B", "Ad C"]
rewards = [0.4, 0.8, 0.6]
epsilon = 0.2

for _ in range(10):
    if random.random() < epsilon:
        choice = random.randint(0, 2)
    else:
        choice = rewards.index(max(rewards))

print("Best Advertisement:", ads[choice])

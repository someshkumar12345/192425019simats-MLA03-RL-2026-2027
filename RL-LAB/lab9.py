import numpy as np

Q = np.array([[1, 5],
              [2, 8],
              [3, 10]])

print("Best Actions")

for i in range(len(Q)):
    print("State", i, "-> Action", np.argmax(Q[i]))

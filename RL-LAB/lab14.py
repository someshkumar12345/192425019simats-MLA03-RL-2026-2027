import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
actor = Sequential([
    Dense(16, activation="relu", input_shape=(3,)),
    Dense(2, activation="softmax")
])
critic = Sequential([
    Dense(16, activation="relu", input_shape=(3,)),
    Dense(1)
])
state = np.array([[3, 1, 0]])
policy = actor.predict(state, verbose=0)
value = critic.predict(state, verbose=0)
print("Action Probabilities")
print(np.round(policy, 3))
print("\nState Value")
print(np.round(value, 3))

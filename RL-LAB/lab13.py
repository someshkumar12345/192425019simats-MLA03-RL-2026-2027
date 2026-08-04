import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
    Dense(16, activation="relu", input_shape=(4,)),
    Dense(3, activation="softmax")
])
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy"
)
state = np.array([[1, 0, 1, 1]])
prob = model.predict(state, verbose=0)
actions = ["Left", "Right", "Park"]
print("Action Probabilities")
for i in range(3):
    print(actions[i], ":", round(prob[0][i], 3))
print("\nSelected Action:", actions[np.argmax(prob)])

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
    Dense(32, activation="relu", input_shape=(4,)),
    Dense(32, activation="relu"),
    Dense(2, activation="linear")
])
model.compile(optimizer="adam", loss="mse")
# Example traffic state
state = np.array([[15, 8, 20, 5]])
q_values = model.predict(state, verbose=0)
print("Predicted Q-Values:")
print(np.round(q_values, 2))
action = np.argmax(q_values)
print("Selected Signal:", action)

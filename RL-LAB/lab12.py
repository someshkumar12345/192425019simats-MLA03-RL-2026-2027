import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
    Dense(16, activation="relu", input_shape=(3,)),
    Dense(2, activation="softmax")
])
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy"
)
state = np.array([[1, 0, 1]])
prob = model.predict(state, verbose=0)
print("Action Probabilities")
print(np.round(prob, 3))
action = np.argmax(prob)
if action == 0:
    print("Action : Pick")
else:
    print("Action : Place")

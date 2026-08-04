import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
    Dense(32, activation="relu", input_shape=(5,)),
    Dense(2, activation="softmax")
])
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy"
)
state = np.array([[1, 0, 1, 0, 1]])
prob = model.predict(state, verbose=0)
actions = ["Step Left", "Step Right"]
print("Action Probabilities")
print(np.round(prob, 3))
print("\nSelected Action:")
print(actions[np.argmax(prob)])

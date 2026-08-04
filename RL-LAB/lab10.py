from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(16, input_shape=(4,), activation="relu"))
model.add(Dense(2))

model.summary()

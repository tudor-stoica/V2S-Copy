from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
import numpy as np

# Define the expected input length
expected_input_length = 5

# Define the model with a single dense layer following the input layer
inputs = L.Input((expected_input_length,), name='input')
x = L.Dense(10)(inputs)  # Simple Dense layer
model = Model(inputs=inputs, outputs=x)

# Print the model summary to confirm the architecture
model.summary()

# Create test inputs with different lengths (3 and 7 instead of 5)
input_too_short = np.random.rand(1, 3)  # Input length of 3
input_too_long = np.random.rand(1, 7)   # Input length of 7

print(input_too_short)
print(input_too_long)

# Try to make predictions with different input sizes and catch errors
model.predict(input_too_short)
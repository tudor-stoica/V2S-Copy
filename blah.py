from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf

# Add these lines to list available devices
physical_devices = tf.config.list_physical_devices()
print("Available physical devices:")
for device in physical_devices:
    print(device)

# List and print available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Available GPU devices:")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU devices available.")
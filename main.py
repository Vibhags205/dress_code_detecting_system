import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model from the current directory
model = load_model("dress_code_detector.h5")

# Now you can use model.predict() just like in Colab
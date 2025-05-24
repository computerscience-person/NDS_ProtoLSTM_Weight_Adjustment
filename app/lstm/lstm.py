import tensorflow as tf
import numpy as np

timesteps = 5
input_features = 7

model = None

def load_model():
    global model
    model = tf.keras.models.load_model("./app/lstm/nds_0_0_3.keras")

def predict(input_data):
    if model is None:
        raise RuntimeError("Model is not loaded")
    if input_data.shape == (timesteps, input_features):
        input_data = tf.expand_dims(input_data, axis=1)
    assert input_data.ndim == 3
    assert input_data[1] == timesteps
    assert input_data[2] == input_features
    return model.predict(input_data)

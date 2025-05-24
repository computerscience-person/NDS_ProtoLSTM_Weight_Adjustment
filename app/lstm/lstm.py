import tensorflow as tf
import numpy as np

model = None

def load_model():
    global model
    model = tf.keras.models.load_model("./app/lstm/nds_0_0_3.keras")

def predict(input_data):
    if model is None:
        raise RuntimeError("Model is not loaded")
    return model.predict(input_data)

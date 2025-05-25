import tensorflow as tf
import keras.models as models
import numpy as np
from pathlib import Path
from ..schemas.models import ContextParameters

timesteps = 3
input_features = 7

model = None
prediction_buffer = None

def load_model(model_path: Path):
    global model
    model = models.load_model(model_path)
    print(model.name)

def predict(input_data):
    global model
    if model is None:
        raise RuntimeError("Model is not loaded")
    if len(input_data.shape) == 2:
        input_data = tf.expand_dims(input_data, axis=0)
    # assert input_data.ndim == 3
    # assert input_data[1] == timesteps
    assert input_data.shape[2] == input_features
    # print(input_data.shape)
    return model.predict(input_data)

def buffer_input(input: ContextParameters):
    structured_data = np.array([
        input.attacks_landed.lower,
        input.attacks_landed.upper,
        input.current_hp,
        input.defenses.crouching,
        input.defenses.standing,
        input.lower_hits,
        input.upper_hits
    ], dtype=np.float32)
    global prediction_buffer
    if prediction_buffer is None:
        prediction_buffer = np.expand_dims(structured_data, axis=(0))
    else:
        prediction_buffer = np.vstack((prediction_buffer, structured_data))
    print(prediction_buffer)

def get_buffer():
    global prediction_buffer
    return prediction_buffer

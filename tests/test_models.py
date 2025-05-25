import numpy as np
from .helpers.synthetic_generators import generate_game_timeseries

def test_generated_data_shape():
    data = generate_game_timeseries(timesteps=30)
    assert data.shape == (1, 30, 7), "Data shape should be (1, timesteps, 7)"

def test_hp_monotonic_decreasing():
    data = generate_game_timeseries()
    hp_values = data[0, :, 2]
    assert np.all(hp_values[:-1] >= hp_values[1:]), "HP should be monotonically non-increasing"

def test_hp_bounds():
    data = generate_game_timeseries()
    hp = data[0, :, 2]
    assert np.all((0 <= hp) & (hp <= 200)), "HP must stay within bounds 0 to 200"


import numpy as np

def generate_game_timeseries(timesteps: int = 50, max_hp: int = 200) -> np.ndarray:
    """
    Generate synthetic game time-series data for testing purposes.
    
    Returns:
        np.ndarray: Shape (1, timesteps, 7), with monotonically decreasing HP.
    """
    np.random.seed(42)
    data = np.zeros((timesteps, 7), dtype=np.float32)
    data[0, 2] = max_hp

    for t in range(1, timesteps):
        prev_hp = data[t - 1, 2]
        damage = np.random.randint(5, 41)  # HP loss per step

        data[t, 1] = np.random.randint(0, 6)  # hits landed
        data[t, 2] = max(prev_hp - damage, 0)  # HP
        data[t, 3] = np.random.randint(0, 4)  # defense
        data[t, 6] = np.random.randint(0, 6)  # extra action

    return np.expand_dims(data, axis=0)  # Shape: (1, timesteps, 7)


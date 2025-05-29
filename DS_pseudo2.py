import numpy as np
from typing import Dict

# === SIMULATION CONFIGURATION ===
SCRIPT_SIZE = 3
WMIN, WMAX = 0.0, 1.0
RMAX, PMAX = 0.4, 0.4
B = 0.5
TOTAL_HP = 10000

# Damage values for different attack types
UPPER_DAMAGE = 150  # Damage per upper body attack
LOWER_DAMAGE = 100  # Damage per lower body attack

# Initialize rulebase with 13 rules (weights between 0-1)
rulebase = np.full(13, 0.5)  # Start with mid-range weights

def generate_script(weights: np.ndarray, script_size: int = SCRIPT_SIZE) -> list[int]:
    """Select rules proportionally to their weights without replacement"""
    script = []
    available_indices = list(range(len(weights)))
    
    for _ in range(script_size):
        total_weight = np.sum(weights[available_indices])
        if total_weight <= 0:
            selected = np.random.choice(available_indices)
        else:
            pick = np.random.uniform(0, total_weight)
            cumulative = 0.0
            for i in available_indices:
                cumulative += weights[i]
                if cumulative >= pick:
                    selected = i
                    break
        script.append(selected)
        available_indices.remove(selected)
    return script

def calculate_fitness(metrics: Dict[str, int]) -> float:
    """Calculate fitness using new metrics format"""
    # Damage calculations
    bot_dmg = (
        UPPER_DAMAGE * metrics['parameters.upper_hits'] + 
        LOWER_DAMAGE * metrics['parameters.lower_hits']
    )
    
    player_dmg = (
        UPPER_DAMAGE * metrics['parameters.attacks_landed.upper'] +
        LOWER_DAMAGE * metrics['parameters.attacks_landed.lower']
    )
    
    # Score components
    dmg_score = (player_dmg - bot_dmg) / TOTAL_HP
    offensive = 0.002 * (metrics['parameters.attacks_landed.upper'] + 
                         metrics['parameters.attacks_landed.lower'])
    defense = 0.003 * (metrics['parameters.defenses.standing'] + 
                        metrics['parameters.defenses.crouching'])
    penalties = -0.005 * (metrics['parameters.upper_hits'] + 
                           metrics['parameters.lower_hits'])
    
    # Final calculation
    raw = B + 0.5 * dmg_score + offensive + defense + penalties
    return np.clip(raw, 0.0, 1.0)

def adjust_weights(
    script_indices: list[int],
    fitness: float,
    weights: np.ndarray
) -> np.ndarray:
    """Update weights with 0-1 bounds and new metric relationships"""
    # Calculate delta based on fitness
    if fitness < B:
        delta = -min(PMAX, (PMAX * (B - fitness)) / B)
    else:
        delta = min(RMAX, (RMAX * (fitness - B)) / (1 - B))
    
    # Apply weighted updates
    new_weights = weights.copy()
    used = np.random.rand(len(script_indices)) < 0.5  # 50% usage chance
    deltas = np.where(used, delta, 0.2 * delta)
    
    # Update script rules
    new_weights[script_indices] += deltas
    total_delta = np.sum(deltas)
    
    # Redistribute inverse delta to non-script rules
    non_script_mask = np.ones_like(new_weights, bool)
    non_script_mask[script_indices] = False
    non_script_count = np.sum(non_script_mask)
    
    if non_script_count > 0:
        redistribution = -total_delta / non_script_count
        new_weights[non_script_mask] += redistribution
    
    return np.clip(new_weights, WMIN, WMAX)

# Example usage
if __name__ == "__main__":
    # Sample metrics matching new format
    example_metrics = {
        'parameters.attacks_landed.lower': 8,
        'parameters.attacks_landed.upper': 12,
        'parameters.current_hp': 85,
        'parameters.defenses.crouching': 15,
        'parameters.defenses.standing': 20,
        'parameters.lower_hits': 5,
        'parameters.upper_hits': 3
    }
    
    script = generate_script(rulebase)
    print("Selected rules:", script)
    
    fitness = calculate_fitness(example_metrics)
    print(f"Fitness score: {fitness:.3f}")
    
    updated_weights = adjust_weights(script, fitness, rulebase)
    print("Updated weights:", np.round(updated_weights, 3))

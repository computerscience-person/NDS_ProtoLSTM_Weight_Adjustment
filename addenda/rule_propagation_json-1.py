import random
import json
from pathlib import Path

# Config
TOTAL_RULES = 50
BATCH_SIZE = 10
OUTPUT_FILE = Path(__file__).parent / "rulebase.json"

# Condition parameters
player_states = ["idle", "moving", "attacking", "blocking", "crouching", "crouch_block", "crouch_attack", "hurt"]
player_actions = [
    "idle", "punch", "kick", "block", "crouch",
    "moveForward", "moveBackward", "staggering",
    "crouch_block", "crouch_punch", "crouch_kick"
]
bot_actions = [
    "idle", "punch", "kick", "block", "crouch",
    "moveForward", "moveBackward",
    "crouch_block", "crouch_punch", "crouch_kick"
]
distance_range = [round(i * 3, 1) for i in range(5)]
hp_range = list(range(1000, 10000, 2000))

valid_actions = {
    "idle": ["idle"],
    "moving": ["moveForward", "moveBackward"],
    "attacking": ["punch", "kick"],
    "blocking": ["block"],
    "crouching": ["crouch"],
    "crouch_block": ["crouch_block"],
    "crouch_attack": ["crouch_punch", "crouch_kick"],
    "hurt": ["staggering"]
}

all_actions = list(set(player_actions))
all_bot_actions = list(set(bot_actions))

# Remove output file if exists
if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

print(f"\U0001F680 Generating {TOTAL_RULES} rules in batches of {BATCH_SIZE}...")

rule_id = 1
with open(OUTPUT_FILE, "w") as f:
    f.write("[")  # Start JSON array
    for batch_num in range(TOTAL_RULES // BATCH_SIZE):
        rules_batch = []
        for _ in range(BATCH_SIZE):
            player_state = random.choice(player_states)
            distance = random.choice(distance_range)
            player_hp = random.choice(hp_range)
            opponent_hp = random.choice(hp_range)
            distance_cond = random.choice(["<", ">"])
            player_hp_cond = random.choice(["<", ">"])
            opponent_hp_cond = random.choice(["<", ">"])

            # Action generation
            if random.random() < 0.2:  # combo type actions count
                combo_length = random.randint(2, 5)
                combo = random.sample(all_bot_actions, combo_length)
                action = " -> ".join(combo)
            else:
                action = random.choice(bot_actions)

            # Match player action based on state
            possible_player_actions = valid_actions.get(player_state, [])
            player_action = random.choice(possible_player_actions) if possible_player_actions else "idle"

            # Build rule object
            conditions = ", ".join([
                f"playerState = '{player_state}'",
                f"playerCurrentAction = '{player_action}'",
                f"distance {distance_cond} {distance}",
                f"playerHP {player_hp_cond} {player_hp}",
                f"opponentHP {opponent_hp_cond} {opponent_hp}"
            ])

            rule = {
                "ruleID": rule_id,
                "conditions": conditions,
                "action": action,
                "weight": 80,  # All rules start with 80 weight — Majchrzak style
                "wasUsed": 0
            }

            rules_batch.append(rule)
            rule_id += 1

        # Write batch
        json.dump(rules_batch, f, indent=4)
        if rule_id <= TOTAL_RULES:
            f.write(",\n")  # Comma between batches
        print(f"✅ Batch {batch_num + 1} written ({rule_id - 1} total rules so far)")

    f.write("]")  # End JSON array

# Sumweight note
total_sumweight = (rule_id - 1) * 80
print(f"\n📊 Final sumweight: {total_sumweight}")
print("📌 All rules initialized to 80. The actual values don't matter — the important thing is that all rules start at 80 and total weight is conserved, as in Majchrzak's implementation.")
